{-# LANGUAGE DeriveAnyClass    #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE TupleSections     #-}
module ML.ARAL.Proxy.RegressionNode
    ( Observation (..)
    , RegressionNode (..)
    , RegressionLayer (..)
    , randRegressionLayer
    , addGroundTruthValueLayer
    , trainRegressionLayer
    -- , trainBatchRegressionLayer
    , applyRegressionLayer
    , prettyRegressionNode
    , prettyRegressionLayer
    ) where

import           Control.Applicative
import           Control.Arrow                               (first)
import           Control.DeepSeq
import           Control.Monad
import           Data.Default
import           Data.List                                   (foldl', sortOn)
import qualified Data.Map.Strict                             as M
import           Data.Maybe                                  (fromMaybe)
import           Data.Ord                                    (Down (..), comparing)
import           Data.Serialize
import qualified Data.Vector                                 as VB
import           Data.Vector.Algorithms.Intro                as VB
import qualified Data.Vector.Storable                        as VS
import           Debug.Trace
import           EasyLogger
import           GHC.Generics
import           Numeric.AD
import           Numeric.Regression.Linear
import           Prelude                                     hiding ((<>))
import           Statistics.Sample.WelfordOnlineMeanVariance
import           System.IO
import           System.Random
import           Text.PrettyPrint
import           Text.Printf

import           ML.ARAL.NeuralNetwork.Normalisation
import           ML.ARAL.NeuralNetwork.Scaling
import           ML.ARAL.Types

-- | One `Observation` holds one input and expected output.
data Observation =
  Observation
    { obsPeriod              :: !Period
    , obsInputValues         :: !(VS.Vector Double)
    -- , obsReward      :: !Double
    -- , obsAction      :: !Int
    , obsExpectedOutputValue :: !Double
    }
  deriving (Eq, Show, Generic, Serialize, NFData)

data RegressionConfig = RegressionConfig
  { regConfigDataOutStepSize            :: !Double -- ^ Step size in terms of output value to group observation data.
  , regConfigDataMaxObservationsPerStep :: !Int    -- ^ Maximum number of data points per group.
  --  , regConfigFunction                   :: !RegFunction              -- ^ Regression function.
  , regConfigVerbose                    :: !Bool   -- ^ Verbose output
  } deriving (Eq, Show, Generic, NFData)

instance Serialize RegressionConfig where
  get = (RegressionConfig <$> get <*> get <*> get) <|> (RegressionConfig <$> get <*> get <*> pure False)


instance Default RegressionConfig where
  def = RegressionConfig 0.25 30 False

-- | Regression node that is aware of recent observations and coefficients.
data RegressionNode =
  RegressionNode
    { regNodeIndex        :: !Int                                 -- ^ Index of node in layer.
    , regNodeObservations :: !(M.Map Int (VB.Vector Observation)) -- ^ Key as `Int` for a scaled output value.
    , regNodeCoefficients :: !(VB.Vector Double)                  -- ^ Current coefficients.
    , regNodeOutWelford   :: !(WelfordExistingAggregate Double)   -- ^ Output scaling.
    , regNodeConfig       :: !RegressionConfig                    -- ^ Configuration.
    }
  deriving (Eq, Show, Generic, Serialize, NFData)

-- | A RegressionLayer holds one node for each action.
data RegressionLayer =
  RegressionLayer
    { regressionLayerActions :: !(VB.Vector RegressionNode)
    , regressionInpWelford   :: !(WelfordExistingAggregate (VS.Vector Double))
    , regressionStep         :: !Int
    }
  deriving (Eq, Show, Generic, Serialize, NFData)


prettyRegressionNode :: Maybe (WelfordExistingAggregate StateFeatures) -> RegressionNode -> Doc
prettyRegressionNode mWel (RegressionNode idx m coefs welOut cfg) =
  text "Node" <> nest nestCols (int idx) $+$
  text "Coefficients" <> nest nestCols (hcat $ punctuate comma $ map (prettyFractional 3) (VB.toList coefs)) $+$
  text "Observations" <> nest nestCols (int (M.size m) <> text " groups with " <> int (sum $ map VB.length (M.elems m))) $+$
  nest (nestCols + 10) (vcat $ map prettyObservationVector (M.toList m))
  where nestCols = 35
        prettyFractional :: (PrintfArg n, Fractional n) => Int -> n -> Doc
        prettyFractional commas = text . printf ("%+." ++ show commas ++ "f")
        prettyObservationVector :: (Int, VB.Vector Observation) -> Doc
        prettyObservationVector (nr, vec)
          | VB.null vec = headDoc <+> text "empty"
          | otherwise = vcat [headDoc <+> prettyObservation (VB.head vec), headDoc <+> prettyObservation (VB.last vec)]
          where headDoc = prettyFractional 4 (fromIntegral nr * stepSize) <+> parens (int nr) <> colon
        prettyObservation :: Observation -> Doc
        prettyObservation (Observation step inpVec out) =
          text "t=" <> int step <> comma <+>
          char '[' <> hcat (punctuate comma $ map (prettyFractional 3) (VS.toList inpVec)) <> char ']' <+>
          maybe mempty (\wel ->
          parens (char '[' <> hcat (punctuate comma $ map (prettyFractional 3) (VS.toList $ normaliseStateFeatureUnbounded wel inpVec)) <> char ']')) mWel <>
          colon <+> prettyFractional 3 out
        stepSize = regConfigDataOutStepSize cfg


prettyRegressionLayer :: RegressionLayer -> Doc
prettyRegressionLayer (RegressionLayer nodes wel _) = vcat (map (prettyRegressionNode (Just wel)) (VB.toList nodes))


-- | Create new regression node with provided config and given number of input values.
randRegressionNode :: RegressionConfig -> Int -> Int -> IO RegressionNode
randRegressionNode cfg nrInpVals nodeIndex = do
  coefs <- VB.fromList <$> replicateM (nrInpVals + 1) (randomRIO (-0.05, 0.05 :: Double))
  return $ RegressionNode nodeIndex M.empty coefs WelfordExistingAggregateEmpty cfg

-- | Create a new empty regression layer by providing the config, the number of nodes for the layer and the number of inputs.
randRegressionLayer :: Maybe RegressionConfig -> Int -> Int -> IO RegressionLayer
randRegressionLayer mCfg nrInput nrOutput = do
  nodes <- mapM (randRegressionNode (fromMaybe def mCfg) nrInput) [0 .. nrOutput - 1]
  return $ RegressionLayer (VB.fromList nodes) WelfordExistingAggregateEmpty 0


-- | Add ground truth value to specific node.
addGroundTruthValueNode :: Observation -> RegressionNode -> RegressionNode
addGroundTruthValueNode obs@(Observation _ _ out) (RegressionNode idx m coefs welOut cfg@(RegressionConfig step maxObs _)) = RegressionNode idx m' coefs welOut' cfg
  where
    key = floor (out * transf)
    transf = 1 / step
    m' = M.alter (Just . maybe (VB.singleton obs) (VB.take maxObs . (obs `VB.cons`))) key m
    welOut' = addValue welOut out


-- | Add ground truth values to the layer.
addGroundTruthValueLayer :: [(Observation, ActionIndex)] -> RegressionLayer -> RegressionLayer
addGroundTruthValueLayer obs (RegressionLayer ms welInp step) =
  RegressionLayer (foldl' (\acc (ob, aId) -> replaceIndex aId (addGroundTruthValueNode ob (acc VB.! aId)) acc) ms obs) welInp' (step + 1)
  where
    welInp' = foldl' addValue welInp (map (obsInputValues . fst) obs)
    replaceIndex idx x xs = xs VB.// [(idx, x)]
      -- VB.take idx xs VB.++ (x `VB.cons` VB.drop (idx + 1) xs)


data RegFunction = RegLinear
  deriving (Eq, Ord, Show, Generic, Serialize, NFData)

costFunction :: RegFunction -> VB.Vector Double -> VS.Vector Double -> Double
costFunction RegLinear coefs inps =
  VS.sum (VS.zipWith (*) (VB.convert coefs) inps) + VB.last coefs


-- | Train a Regression Node.
trainRegressionNode :: WelfordExistingAggregate (VS.Vector Double) -> Period -> RegressionNode -> RegressionNode
trainRegressionNode welInp period old@(RegressionNode idx m coefs welOut cfg)
  -- trace ("period: " ++ show (period)) $
  -- trace ("allObs len: " ++ show (length allObs)) $
 =
  if M.null m || VB.length allObs < observationsToUse --  || period `mod` ((VB.length coefs - 1) * observationsToUse `div` 10) /= 0 -- retrain every ~10% of change values
    then old
    else let reg = regress ys xs coefs :: [Model VB.Vector Double]
          in RegressionNode idx (periodicallyCleanObservations m) (untilThreshold (fromIntegral (VB.length coefs) * 5e-4) 1 coefs reg) welOut cfg
                -- regressStochastic ys xs coefs :: [Model VB.Vector Double]
       -- let reg = regress ys xs coefs :: [Model VB.Vector Double]
       -- in RegressionNode idx m (indexWithDefault (VB.length ys) coefs reg) welOut cfg
       -- trace ("trainRegressionNode\n" ++ show (prettyRegressionNode old))
       -- trace ("ys: " ++ show ys)
       -- trace ("auto: " ++ show (VB.map auto ys :: VB.Vector Double) )
       -- trace ("xs: " ++ show (zip (VB.toList xs) (VB.toList ys))) $
       -- trace ("cost: " ++ show (totalCost coefs (fmap auto ys) (fmap (fmap auto) xs)))
       -- RegressionNode idx m (indexWithDefault (VB.length ys) coefs reg) cfg
  where
    observationsToUse = VS.length $ obsInputValues $ VB.head $ snd $ M.findMax m
    untilThreshold :: Double -> Int -> VB.Vector Double -> [VB.Vector Double] -> VB.Vector Double
    untilThreshold _ idx old [] = $(pureLogPrintWarning) ("Default after " ++ show idx) old
    untilThreshold thresh idx old (new:rest)
      | VB.sum (VB.zipWith (\a b -> abs (a - b)) old new) <= thresh =
        if regConfigVerbose cfg
          then $(pureLogPrintWarning) ("Threshold reached. Steps: " ++ show idx) new
          else new
      | null rest = $(pureLogPrintWarning) ("No more models, but threshold not reached. Steps: " ++ show idx) new
      | idx >= 200 = $(pureLogPrintWarning) ("Threshold of " ++ show thresh ++ " never reached in 200 steps: " ++ show (VB.sum (VB.zipWith (\a b -> abs (a - b)) old new))) new
      | otherwise = untilThreshold thresh (idx + 1) new rest
    allObs :: VB.Vector Observation
    -- allObs = VB.filter ((>= maxPeriod - 3000) . obsPeriod) $ VB.concat (M.elems m)
    allObs = VB.take observationsToUse $ VB.modify (VB.sortBy (comparing (Down . obsPeriod))) $ VB.concat (M.elems m)
    xs :: VB.Vector (VB.Vector Double)
    xs = VB.map (VB.convert . normaliseStateFeatureUnbounded welInp . obsInputValues) allObs
    ys :: VB.Vector Double
    ys =
      VB.map
      -- (scaleMinMax (-5, 5) .
       -- normaliseUnbounded welOut .
        (obsExpectedOutputValue)
        allObs
    lastObsPeriod = obsPeriod $ VB.last allObs
    periodicallyCleanObservations m'
      | period `mod` 1000 == 0 = M.filter (not . VB.null) . M.map (VB.filter ((>= lastObsPeriod) . obsPeriod)) $ m'
      | otherwise = m'


-- | Train regression layger (= all nodes).
trainRegressionLayer :: Period -> RegressionLayer -> RegressionLayer
trainRegressionLayer period (RegressionLayer nodes wel step)
  | period < 100 = RegressionLayer nodes wel step
  | otherwise = RegressionLayer (VB.map (trainRegressionNode wel period) nodes) wel step


-- trainBatchRegressionNode :: WelfordExistingAggregate (VS.Vector Double) -> Period -> [((StateFeatures, ActionIndex, IsRandomAction), Double)] -> RegressionNode -> RegressionNode
-- trainBatchRegressionNode welInp period batchAllActions old@(RegressionNode idx m coefs welOut cfg) =
--   if VB.length xs < (3 + undefined)
--   then old
--   else
--     let reg = regressStochastic ys xs coefs :: [Model VB.Vector Double]
--     in RegressionNode idx m (indexWithDefault (VB.length ys) coefs reg) welOut cfg
--   where
--     ys :: VB.Vector Double
--     ys = VB.fromList $ map snd batchAllActions
--     xs :: VB.Vector (VB.Vector Double)
--     xs = VB.fromList $ map (VB.convert . fst3) $ filter ((== idx) . snd3) (map fst batchAllActions)
--     fst3 (x,_,_) = x
--     snd3 (_,x,_) = x
    -- headWithDefault def []  = $(pureLogPrintWarning) ("Could not solve regression in period " ++ show period) def
    -- headWithDefault _ (x:_) = x
    -- lastWithDefault def [] = $(pureLogPrintWarning) ("Could not solve regression in period " ++ show period) def
    -- lastWithDefault _ xs   = last xs
    -- indexWithDefault idx def [] = def
    -- indexWithDefault 0 _ (x:_) = x
    -- indexWithDefault idx def (x:xs)
    --   | null xs = x
    --   | otherwise = indexWithDefault (idx - 1) def xs
--     headWithDefault def []  = $(pureLogPrintWarning) ("Could not solve regression in period " ++ show period) def
--     headWithDefault _ (x:_) = x
--     lastWithDefault def [] = $(pureLogPrintWarning) ("Could not solve regression in period " ++ show period) def
--     lastWithDefault _ xs   = last xs
--     indexWithDefault idx def []     = def
--     indexWithDefault 0 _ (x:_)      = x
--     indexWithDefault idx def (x:xs)
--       | null xs = x
--       | otherwise = indexWithDefault (idx - 1) def xs

-- -- | Train regression layger (= all nodes).
-- trainBatchRegressionLayer :: Period  -> [((StateFeatures, ActionIndex, IsRandomAction), Double)] -> RegressionLayer -> RegressionLayer
-- trainBatchRegressionLayer period xs (RegressionLayer nodes wel step)
--   | period < 100 = RegressionLayer nodes wel step
--   | otherwise = RegressionLayer (VB.map (trainBatchRegressionNode wel period xs) nodes) wel step


-- | Apply a regression node.
applyRegressionNode :: RegressionNode -> VS.Vector Double -> Double
applyRegressionNode (RegressionNode idx _ coefs welOut _) inps
  | VB.length coefs - 1 /= VS.length inps = error $ "applyRegressionNode: Expected number of coefficients is not correct: " ++ show (VB.length coefs, VS.length inps)
  | otherwise = -- compute coefs (VB.convert inps)
    let res = VS.sum (VS.zipWith (*) (VB.convert coefs) inps) + VB.last coefs
    in
    -- trace ("idx: " ++ show idx ++ ", " ++ show inps ++ " res: " ++ show res)

    -- unscaleMinMax (-5, 5) $
    -- denormaliseUnbounded welOut $
    (VS.sum (VS.zipWith (*) (VB.convert coefs) inps) + VB.last coefs)


-- | Apply regression layer to given inputs
applyRegressionLayer :: RegressionLayer -> ActionIndex -> VS.Vector Double -> Double
applyRegressionLayer (RegressionLayer nodes wel _) actIdx stateFeat =
  -- trace ("applyRegressionLayer: " ++ show (VB.length regNodes, actIdx))
  applyRegressionNode (nodes VB.! actIdx) (normaliseStateFeature wel stateFeat)
  -- trace ("inps: " ++ show inps)


  -- -- let (RegressionNode _ _ _ coefs) = regNodes VB.! actIdx
  --  error "TODO"
