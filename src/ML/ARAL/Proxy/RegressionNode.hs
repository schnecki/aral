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
    , prettyRegressionLayerNoObs
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

import           ML.ARAL.Decay
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
  def = RegressionConfig 0.1 30 False

-- | Regression node that is aware of recent observations and coefficients.
data RegressionNode =
  RegressionNode
    { regNodeIndex        :: !Int                                 -- ^ Index of node in layer.
    , regNodeObservations :: !(M.Map Int (VB.Vector Observation)) -- ^ Key as `Int` for a scaled output value.
    , regNodeCoefficients :: !(VS.Vector Double)                  -- ^ Current coefficients.
    , regNodeHeatMap      :: !(VS.Vector Bool)                    -- ^ Coefficient enabled?
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


prettyRegressionNode :: Bool -> Maybe (WelfordExistingAggregate StateFeatures) -> RegressionNode -> Doc
prettyRegressionNode printObs mWel (RegressionNode idx m coefs heatMap welOut cfg) =
  text "Node"           $$ nest nestCols (int idx) $+$
  text "Coefficients"   $$ nest nestCols (hcat $ punctuate comma $ map (prettyFractional 3) (VS.toList coefs)) $+$
  text "Output scaling" $$ nest nestCols (text (show welOut)) $+$
  text "Observations"   $$ nest nestCols (int (M.size m) <> text " groups with " <> int (sum $ map VB.length (M.elems m))) $+$
  if printObs
     then nest (nestCols + 10) (vcat $ map prettyObservationVector (M.toList m))
     else mempty
  where nestCols = 40
        prettyFractional :: (PrintfArg n) => Int -> n -> Doc
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
prettyRegressionLayer (RegressionLayer nodes wel _) = vcat (map (prettyRegressionNode True (Just wel)) (VB.toList nodes))

prettyRegressionLayerNoObs :: RegressionLayer -> Doc
prettyRegressionLayerNoObs (RegressionLayer nodes wel _) = vcat (zipWith (\idx n -> text "Layer Node" <+> int idx $+$ prettyRegressionNode False (Just wel) n) [0..] (VB.toList nodes))


-- | Create new regression node with provided config and given number of input values.
randRegressionNode :: RegressionConfig -> Int -> Int -> IO RegressionNode
randRegressionNode cfg nrInpVals nodeIndex = do
  coefs <- VS.fromList <$> replicateM (nrInpVals + 1) (randomRIO (-0.05, 0.05 :: Double))
  return $ RegressionNode nodeIndex M.empty coefs (VS.replicate (nrInpVals + 1) True) WelfordExistingAggregateEmpty cfg

-- | Create a new empty regression layer by providing the config, the number of nodes for the layer and the number of inputs.
randRegressionLayer :: Maybe RegressionConfig -> Int -> Int -> IO RegressionLayer
randRegressionLayer mCfg nrInput nrOutput = do
  nodes <- mapM (randRegressionNode (fromMaybe def mCfg) nrInput) [0 .. nrOutput - 1]
  return $ RegressionLayer (VB.fromList nodes) WelfordExistingAggregateEmpty 0

-- | Filter out elements using heat map.
filterHeatMap :: VS.Vector Bool -> VS.Vector Double -> VS.Vector Double
filterHeatMap heatMap vec
  | VS.length heatMap /= VS.length vec = error $ "Length of heat map and vector do not coincide: " ++ show (VS.length heatMap, VS.length vec)
  | otherwise = VS.filter (/= 0) $ VS.zipWith (\act x -> fromAct act * x) heatMap vec
  where fromAct True  = 1
        fromAct False = 0

-- | Recreate original length of coefficients from heat map.
toCoefficients :: VS.Vector Bool -> VS.Vector Double -> VS.Vector Double
toCoefficients heatMap vec
  | VS.null vec && VS.null heatMap = VS.empty
  | VS.null heatMap || VS.null vec = error $ "Lengths of heat map and vector do not match: " ++ show (VS.length heatMap, VS.length vec)
  | VS.head heatMap = VS.head vec `VS.cons` toCoefficients (VS.tail heatMap) (VS.tail vec)
  | otherwise = 0 `VS.cons` toCoefficients (VS.tail heatMap) vec


-- | Add ground truth value to specific node.
addGroundTruthValueNode :: Period -> Observation -> RegressionNode -> RegressionNode
addGroundTruthValueNode period obs@(Observation _ _ out) (RegressionNode idx m coefs heatMap welOut cfg@(RegressionConfig step maxObs _)) = RegressionNode idx m' coefs heatMap welOut' cfg
  where
    key = floor (normaliseUnbounded welOut out * transf)
    transf = 1 / step
    m' = M.alter (Just . maybe (VB.singleton obs) (VB.take maxObs . (obs `VB.cons`))) key m
    welOut'
      | period < 30000 = addValue welOut out
      | otherwise = welOut


-- | Add ground truth values to the layer.
addGroundTruthValueLayer :: Period -> [(Observation, ActionIndex)] -> RegressionLayer -> RegressionLayer
addGroundTruthValueLayer period obs (RegressionLayer ms welInp step) =
  RegressionLayer (foldl' (\acc (ob, aId) -> replaceIndex aId (addGroundTruthValueNode period ob (acc VB.! aId)) acc) ms obs) welInp' (step + 1)
  where
    welInp'
      | period < 30000 = foldl' addValue welInp (map (obsInputValues . fst) obs)
      | otherwise = welInp
    replaceIndex idx x xs = xs VB.// [(idx, x)]
      -- VB.take idx xs VB.++ (x `VB.cons` VB.drop (idx + 1) xs)


data RegFunction = RegLinear
  deriving (Eq, Ord, Show, Generic, Serialize, NFData)

costFunction :: RegFunction -> VB.Vector Double -> VS.Vector Double -> Double
costFunction RegLinear coefs inps =
  VS.sum (VS.zipWith (*) (VB.convert coefs) inps) + VB.last coefs


-- | Train a Regression Node.
trainRegressionNode :: WelfordExistingAggregate (VS.Vector Double) -> Int -> Period -> RegressionNode -> RegressionNode
trainRegressionNode welInp nrNodes period old@(RegressionNode idx m coefs heatMap welOut cfg)
  -- trace ("period: " ++ show (period)) $
  -- trace ("allObs len: " ++ show (length allObs)) $
 =
  if M.null m || VB.length allObs < observationsToUse || period `mod` max 1 ((VS.length coefs - 1) * nrNodes `div` 10) /= 0 -- retrain every ~10% of change values. Note: This does not take number of worker agents into account!
    then old
    else
    let coefsFiltered = filterHeatMap heatMap coefs
        reg = map VS.convert $ regress ys xs (VB.convert coefsFiltered) :: [Model VS.Vector Double]
        alpha = decaySetup (ExponentialDecay (Just 1e-3) 0.8 30000) period 1
        fittedCoefs :: VS.Vector Double
        fittedCoefs = untilThreshold (fromIntegral (VS.length coefsFiltered) * 5e-4) 1 coefsFiltered reg
        coefs' = toCoefficients heatMap $ VS.zipWith (\oldVal newVal -> (1 - alpha) * oldVal + alpha * newVal) coefsFiltered fittedCoefs
        heatMap'
          | period > 10000 = VS.zipWith (\act v -> abs v >= 0.01 && act) heatMap coefs' VS.// [(VS.length heatMap - 1, True)]
          | otherwise = heatMap
          in RegressionNode idx (periodicallyCleanObservations m) coefs' heatMap' welOut cfg
          -- in RegressionNode idx (periodicallyCleanObservations m) coefs' welOut cfg
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
    untilThreshold :: Double -> Int -> VS.Vector Double -> [VS.Vector Double] -> VS.Vector Double
    untilThreshold _ idx old [] = $(pureLogPrintWarning) ("Default after " ++ show idx) old
    untilThreshold thresh idx old (new:rest)
      | VS.sum (VS.zipWith (\a b -> abs (a - b)) old new) <= thresh =
        if regConfigVerbose cfg
          then $(pureLogPrintWarning) ("Threshold reached. Steps: " ++ show idx) new
          else new
      | null rest = $(pureLogPrintWarning) ("No more models, but threshold not reached. Steps: " ++ show idx) new
      | idx >= 200 = $(pureLogPrintWarning) ("Threshold of " ++ show thresh ++ " never reached in 200 steps: " ++ show (VS.sum (VS.zipWith (\a b -> abs (a - b)) old new))) new
      | otherwise = untilThreshold thresh (idx + 1) new rest
    allObs :: VB.Vector Observation
    -- allObs = VB.filter ((>= maxPeriod - 3000) . obsPeriod) $ VB.concat (M.elems m)
    allObs = VB.take observationsToUse $ VB.modify (VB.sortBy (comparing (Down . obsPeriod))) $ VB.concat (M.elems m)
    xs :: VB.Vector (VB.Vector Double)
    xs = VB.map (VB.convert . filterHeatMap (VS.init heatMap) .
                 normaliseStateFeatureUnbounded welInp .  -- <- maybe breaks algorithm (different input scaling not preserved? e.g. for indicators?)
                 obsInputValues) allObs
    ys :: VB.Vector Double
    ys =
      VB.map
      -- (scaleMinMax (-5, 5) . obsExpectedOutputValue)
        ((10*) . normaliseUnbounded welOut . obsExpectedOutputValue)
        -- obsExpectedOutputValue
        allObs
    lastObsPeriod = obsPeriod $ VB.last allObs
    periodicallyCleanObservations m'
      | period `mod` 1000 == 0 = M.filter (not . VB.null) . M.map (VB.filter ((>= lastObsPeriod) . obsPeriod)) $ m'
      | otherwise = m'


-- | Train regression layger (= all nodes).
trainRegressionLayer :: Period -> RegressionLayer -> RegressionLayer
trainRegressionLayer period (RegressionLayer nodes wel step)
  | period < 100 = RegressionLayer nodes wel step
  | otherwise = RegressionLayer (VB.map (trainRegressionNode wel (VB.length nodes) period) nodes) wel step


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
applyRegressionNode (RegressionNode idx _ coefs heatMap welOut _) inps
  | VS.length coefs - 1 /= VS.length inps = error $ "applyRegressionNode: Expected number of coefficients is not correct: " ++ show (VS.length coefs, VS.length inps)
  | otherwise = -- compute coefs (VB.convert inps)
    -- let res = VS.sum (VS.zipWith3 (\act coef inp -> act * coef * inp) heatMap (VB.convert coefs) inps) + VS.last coefs
    -- in
    -- trace ("idx: " ++ show idx ++ ", " ++ show inps ++ " res: " ++ show res)

    -- unscaleMinMax (-5, 5) $
    denormaliseUnbounded welOut . (/10) $
    VS.sum (VS.zipWith3 (\act w i -> fromAct act * w * i) heatMap (VB.convert coefs) inps) + VS.last coefs
  where fromAct True  = 1
        fromAct False = 0


-- | Apply regression layer to given inputs
applyRegressionLayer :: RegressionLayer -> ActionIndex -> VS.Vector Double -> Double
applyRegressionLayer (RegressionLayer nodes wel _) actIdx stateFeat =
  -- trace ("applyRegressionLayer: " ++ show (VB.length regNodes, actIdx))
  applyRegressionNode (nodes VB.! actIdx) (normaliseStateFeatureUnbounded wel stateFeat) -- stateFeat
  -- trace ("inps: " ++ show inps)


  -- -- let (RegressionNode _ _ _ coefs) = regNodes VB.! actIdx
  --  error "TODO"
