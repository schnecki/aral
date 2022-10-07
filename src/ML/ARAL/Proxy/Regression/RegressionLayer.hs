{-# LANGUAGE DeriveAnyClass    #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE TupleSections     #-}
module ML.ARAL.Proxy.Regression.RegressionLayer
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
import           Control.Arrow                                    (first)
import           Control.DeepSeq
import           Control.Monad
import           Data.Default
import           Data.List                                        (foldl', intercalate, sortOn)
import qualified Data.Map.Strict                                  as M
import           Data.Maybe                                       (fromMaybe)
import           Data.Ord                                         (Down (..), comparing)
import           Data.Serialize
import qualified Data.Vector                                      as VB
import           Data.Vector.Algorithms.Intro                     as VB
import qualified Data.Vector.Storable                             as VS
import           Debug.Trace
import           EasyLogger
import           GHC.Generics
import           Numeric.AD
import           Numeric.Regression.Linear
import           Prelude                                          hiding ((<>))
import           Statistics.Sample.WelfordOnlineMeanVariance
import           System.IO
import           System.IO.Unsafe                                 (unsafePerformIO)
import           System.Random
import           Text.PrettyPrint
import           Text.Printf

import           ML.ARAL.Decay
import           ML.ARAL.NeuralNetwork.Normalisation
import           ML.ARAL.NeuralNetwork.Scaling
import           ML.ARAL.Proxy.Regression.VolatilityRegimeExpSmth
import           ML.ARAL.Types

-- | One `Observation` holds one input and expected output.
data Observation =
  Observation
    { obsPeriod              :: !Period
    , obsInputValues         :: !(VS.Vector Double)
    , obsVarianceRegimeValue :: !Double -- e.g. Reward in RL
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
    { regressionLayerActions :: !(VB.Vector RegressionNode, VB.Vector RegressionNode) -- ^ On set of actions for each regime.
    , regressionInpWelford   :: !(WelfordExistingAggregate (VS.Vector Double))
    , regressionStep         :: !Int
    , regressionRegime       :: !RegimeDetection    -- Low or High variance regime
    }
  deriving (Show, Generic, Serialize, NFData)


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
        prettyObservation (Observation step inpVec _ out) =
          text "t=" <> int step <> comma <+>
          char '[' <> hcat (punctuate comma $ map (prettyFractional 3) (VS.toList inpVec)) <> char ']' <+>
          maybe mempty (\wel ->
          parens (char '[' <> hcat (punctuate comma $ map (prettyFractional 3) (VS.toList $ normaliseStateFeatureUnbounded wel inpVec)) <> char ']')) mWel <>
          colon <+> prettyFractional 3 out
        stepSize = regConfigDataOutStepSize cfg


prettyRegressionLayer :: RegressionLayer -> Doc
prettyRegressionLayer (RegressionLayer (nodesLow, nodesHigh) welInp _ regime) =
  vcat (text "Low Regime" : map (prettyRegressionNode True (Just welInp)) (VB.toList nodesLow)) $+$
  vcat (text "High Regime" : map (prettyRegressionNode True (Just welInp)) (VB.toList nodesHigh))

prettyRegressionLayerNoObs :: RegressionLayer -> Doc
prettyRegressionLayerNoObs (RegressionLayer (nodesLow, nodesHigh) welInp _ regime) =
  vcat (text "Low Regime" : zipWith (\idx n -> text "Layer Node" <+> int idx $+$ prettyRegressionNode False (Just welInp) n) [0 ..] (VB.toList nodesLow)) $+$
  vcat (text "High Regime" : zipWith (\idx n -> text "Layer Node" <+> int idx $+$ prettyRegressionNode False (Just welInp) n) [0 ..] (VB.toList nodesHigh))


-- Regime helpers

overRegime :: Regime -> (VB.Vector RegressionNode -> VB.Vector RegressionNode) -> (VB.Vector RegressionNode, VB.Vector RegressionNode) -> (VB.Vector RegressionNode, VB.Vector RegressionNode)
overRegime Low f (nodesLow, nodesHigh)  = (f nodesLow, nodesHigh)
overRegime High f (nodesLow, nodesHigh) = (nodesLow, f nodesHigh)

withRegime :: Regime -> (VB.Vector RegressionNode -> a) -> (VB.Vector RegressionNode, VB.Vector RegressionNode) -> a
withRegime Low f (nodesLow, _)   = f nodesLow
withRegime High f (_, nodesHigh) = f nodesHigh

overBothRegimes ::(VB.Vector RegressionNode -> VB.Vector RegressionNode) -> (VB.Vector RegressionNode, VB.Vector RegressionNode) -> (VB.Vector RegressionNode, VB.Vector RegressionNode)
overBothRegimes f (nodesLow, nodesHigh) = (f nodesLow, f nodesHigh)


-- | Create new regression node with provided config and given number of input values.
randRegressionNode :: RegressionConfig -> Int -> Int -> IO RegressionNode
randRegressionNode cfg nrInpVals nodeIndex = do
  coefs <- VS.fromList <$> replicateM (nrInpVals + 1) (randomRIO (-0.05, 0.05 :: Double))
  return $ RegressionNode nodeIndex M.empty coefs (VS.replicate (nrInpVals + 1) True) WelfordExistingAggregateEmpty cfg

-- | Create a new empty regression layer by providing the config, the number of nodes for the layer and the number of inputs.
randRegressionLayer :: Maybe RegressionConfig -> Int -> Int -> IO RegressionLayer
randRegressionLayer mCfg nrInput nrOutput = do
  nodes <- mapM (randRegressionNode (fromMaybe def mCfg) nrInput) [0 .. nrOutput - 1]
  return $ RegressionLayer (VB.fromList nodes, VB.fromList nodes) WelfordExistingAggregateEmpty 0 def

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
addGroundTruthValueNode period obs@(Observation _ _ _ out) (RegressionNode idx m coefs heatMap welOut cfg@(RegressionConfig step maxObs _)) = RegressionNode idx m' coefs heatMap welOut' cfg
  where
    key = floor (normaliseUnbounded welOut out * transf)
    transf = 1 / step
    m' = M.alter (Just . maybe (VB.singleton obs) (VB.take maxObs . (obs `VB.cons`))) key m
    welOut'
      | True || period < 30000 = addValue welOut out
      | otherwise = welOut


-- | Add ground truth values from different workers to the layer.
addGroundTruthValueLayer :: Period -> [(Observation, ActionIndex)] -> RegressionLayer -> RegressionLayer
addGroundTruthValueLayer period [] lay = lay
addGroundTruthValueLayer period obs (RegressionLayer nodes welInp step regime) =
  let regExp = currentRegimeExp regime'
   in writeRegimeFile regExp `seq`
      RegressionLayer
    -- (foldl' (\acc (workerId, (ob, aId)) -> replaceIndex aId (addGroundTruthValueNode period ob (acc VB.! aId)) acc) nodes (zip [0..] obs))
        ((if period < 1000 then trace ("TODO: multiple regime detections!") else id) $ foldl' updateNodes nodes obs)
        welInp'
        (step + 1)
        regime' -- only on first one (others are from different workers!)
  where
    updateNodes nodes (ob, aId) = overRegime (currentRegimeExp regime) (\ns -> replaceIndex aId (addGroundTruthValueNode period ob (ns VB.! aId)) ns) nodes
    reward = obsVarianceRegimeValue $ fst $ head obs
    regime' = addValueToRegime regime reward
    welInp'
      | True || period < 30000 = foldl' addValue welInp (map (obsInputValues . fst) obs)
      | otherwise = welInp
    replaceIndex idx x xs = xs VB.// [(idx, x)]
      -- VB.take idx xs VB.++ (x `VB.cons` VB.drop (idx + 1) xs)
    getBorder
      | period < 10 = const 0
      | otherwise = (\(mean, _, x) -> mean + sqrt x) . finalize
    writeRegimeFile reg =
      unsafePerformIO $ do
        let txt =
              show period ++
              "\t" ++
              show (obsVarianceRegimeValue $ fst $ head obs) ++
              "\t" ++
              show (fromEnum reg) ++
              "\t" ++ show (regimeExpSmthFast regime) ++ "\t" ++ show (regimeExpSmthSlow regime) ++ "\t" ++ show (getBorder $ regimeWelfordAll regime) ++ "\n"
        when (period == 0) $ do writeFile "regime" $ "period\treward\tregime\tExpFast\tExpSlow\tBorder\n"
        appendFile "regime" txt


-- TODO
-- data RegFunction = RegLinear
--   deriving (Eq, Ord, Show, Generic, Serialize, NFData)
-- costFunction :: RegFunction -> VB.Vector Double -> VS.Vector Double -> Double
-- costFunction RegLinear coefs inps =
--   VS.sum (VS.zipWith (*) (VB.convert coefs) inps) + VB.last coefs


-- | Train a Regression Node.
trainRegressionNode :: WelfordExistingAggregate (VS.Vector Double) -> Int -> Period -> RegressionNode -> RegressionNode
trainRegressionNode welInp nrNodes period old@(RegressionNode idx m coefs heatMap welOut cfg) =
  if M.null m || VB.length allObs < observationsToUse || period `mod` max 1 ((VS.length coefs - 1) * nrNodes `div` 10) /= 0 -- retrain every ~10% of change values. Note: This does not take number of worker agents into account!
    then old
    else let coefsFiltered = filterHeatMap heatMap coefs
             reg = map VS.convert $ regress ys xs (VB.convert coefsFiltered) :: [Model VS.Vector Double]
             learnRate = decaySetup (ExponentialDecay (Just 1e-5) 0.8 30000) period 1
             fittedCoefs :: VS.Vector Double
             fittedCoefs = untilThreshold (fromIntegral (VS.length coefsFiltered) * 5e-4) 1 coefsFiltered reg
             coefs' = toCoefficients heatMap $ VS.zipWith (\oldVal newVal -> (1 - learnRate) * oldVal + learnRate * newVal) coefsFiltered fittedCoefs
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
    allObs = VB.take observationsToUse $ VB.modify (VB.sortBy (comparing (Down . obsPeriod))) $ VB.concat (M.elems m)
    xs :: VB.Vector (VB.Vector Double)
    xs =
      VB.map
        (VB.convert .
         filterHeatMap (VS.init heatMap) .
         normaliseStateFeatureUnbounded welInp . -- <- maybe breaks algorithm (different input scaling not preserved? e.g. for indicators?)
         obsInputValues)
        allObs
    ys :: VB.Vector Double
    ys =
      VB.map
      -- (scaleMinMax (-5, 5) . obsExpectedOutputValue)
         -- (*100) .
        (normaliseUnbounded welOut . obsExpectedOutputValue)
        -- obsExpectedOutputValue
        allObs
    lastObsPeriod = obsPeriod $ VB.last allObs
    periodicallyCleanObservations m'
      | period `mod` 1000 == 0 = M.filter (not . VB.null) . M.map (VB.filter ((>= lastObsPeriod) . obsPeriod)) $ m'
      | otherwise = m'


-- | Train regression layger (= all nodes).
trainRegressionLayer :: Period -> RegressionLayer -> RegressionLayer
trainRegressionLayer period (RegressionLayer nodes welInp step regime)
  | period < 1000 = RegressionLayer nodes welInp step regime -- only used for learning the normalization
  | otherwise = RegressionLayer (overBothRegimes (\ns -> VB.map (trainRegressionNode welInp (VB.length ns) period) ns) nodes) welInp step regime


-- | Apply a regression node.
applyRegressionNode :: RegressionNode -> VS.Vector Double -> Double
applyRegressionNode (RegressionNode idx _ coefs heatMap welOut _) inps
  | VS.length coefs - 1 /= VS.length inps = error $ "applyRegressionNode: Expected number of coefficients is not correct: " ++ show (VS.length coefs, VS.length inps)
  | otherwise = -- compute coefs (VB.convert inps)
    -- let res = VS.sum (VS.zipWith3 (\act coef inp -> act * coef * inp) heatMap (VB.convert coefs) inps) + VS.last coefs
    -- in
    -- trace ("idx: " ++ show idx ++ ", " ++ show inps ++ " res: " ++ show res)

    -- unscaleMinMax (-5, 5) $
    denormaliseUnbounded welOut $ -- . (/100) $
    VS.sum (VS.zipWith3 (\act w i -> fromAct act * w * i) heatMap (VB.convert coefs) inps) + VS.last coefs
  where fromAct True  = 1
        fromAct False = 0


-- | Apply regression layer to given inputs
applyRegressionLayer :: RegressionLayer -> ActionIndex -> VS.Vector Double -> Double
applyRegressionLayer (RegressionLayer nodes welInp _ regime) actIdx stateFeat =
  -- trace ("applyRegressionLayer: " ++ show (VB.length regNodes, actIdx))
  withRegime (currentRegimeExp regime) (\ns -> applyRegressionNode (ns VB.! actIdx) (normaliseStateFeatureUnbounded welInp stateFeat)) nodes


