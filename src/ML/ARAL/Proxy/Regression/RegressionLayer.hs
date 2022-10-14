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
    , regressionRegime       :: !(VB.Vector RegimeDetection)    -- Low or High variance regime
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
prettyRegressionLayer (RegressionLayer (nodesLow, nodesHigh) welInp _ _) =
  vcat (text "Low Regime" : map (prettyRegressionNode True (Just welInp)) (VB.toList nodesLow)) $+$ mempty $+$
  vcat (text "High Regime" : map (prettyRegressionNode True (Just welInp)) (VB.toList nodesHigh))

prettyRegressionLayerNoObs :: RegressionLayer -> Doc
prettyRegressionLayerNoObs (RegressionLayer (nodesLow, nodesHigh) welInp _ _) =
  vcat (text "Low Regime" : zipWith (\idx n -> text "Layer Node" <+> int idx $+$ prettyRegressionNode False (Just welInp) n) [0 ..] (VB.toList nodesLow)) $+$ mempty $+$
  vcat (text "High Regime" : zipWith (\idx n -> text "Layer Node" <+> int idx $+$ prettyRegressionNode False (Just welInp) n) [0 ..] (VB.toList nodesHigh))


-- Regime helpers

overRegime :: Int -> Regime -> (VB.Vector RegressionNode -> VB.Vector RegressionNode) -> (VB.Vector RegressionNode, VB.Vector RegressionNode) -> (VB.Vector RegressionNode, VB.Vector RegressionNode)
overRegime step _ f (nodesLow, nodesHigh)
  | step < periodsSharedRegime = (f nodesLow, f nodesHigh)
overRegime _ Low f (nodesLow, nodesHigh)  = (f nodesLow, nodesHigh)
overRegime _ High f (nodesLow, nodesHigh) = (nodesLow, f nodesHigh)

withRegime :: Int -> Regime -> (VB.Vector RegressionNode -> a) -> (VB.Vector RegressionNode, VB.Vector RegressionNode) -> a
withRegime step _ f (nodesLow, _)
  | step < periodsSharedRegime = f nodesLow
withRegime _ Low f (nodesLow, _)   = f nodesLow
withRegime _ High f (_, nodesHigh) = f nodesHigh

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
  return $ RegressionLayer (VB.fromList nodes, VB.fromList nodes) WelfordExistingAggregateEmpty 0 (VB.singleton def)

-- | Filter out elements using heat map.
filterHeatMap :: VS.Vector Bool -> VS.Vector Double -> VS.Vector Double
filterHeatMap heatMap vec
  | VS.length heatMap /= VS.length vec = error $ "filterHeatMap: Lengths of heat map and vector do not coincide: " ++ show (VS.length heatMap, VS.length vec)
  | otherwise = VS.imapMaybe (\idx x -> if heatMap VS.! idx then Just x else Nothing) vec

-- | Recreate original length of coefficients from heat map.
toCoefficients :: VS.Vector Bool -> VS.Vector Double -> VS.Vector Double
toCoefficients heatMap vec
  | VS.null vec && VS.null heatMap = VS.empty
  | VS.null heatMap || VS.null vec = error $ "toCoefficients: Lengths of heat map and vector do not match: " ++ show (VS.length heatMap, VS.length vec)
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
addGroundTruthValueLayer period obs (RegressionLayer nodes welInp step regime)
  | step > 0 && length obs /= VB.length regime =
    error $ "Regime length does ot fit number of observations: " ++ show (VB.length regime, length obs) ++ ". Number of parallel observations must be constant!"
  | otherwise =
    let regExp = currentRegimeExp (VB.head regime')
     in writeRegimeFile regExp `seq` RegressionLayer (foldl' updateNodes nodes (zip [0 ..] obs)) welInp' (step + 1) regime'
  where
    updateNodes nds (regId, (ob, aId)) = overRegime step (currentRegimeExp (regime' VB.! regId)) (\ns -> replaceIndex aId (addGroundTruthValueNode period ob (ns VB.! aId)) ns) nds
    -- reward = obsVarianceRegimeValue $ fst $ head obs
    regime0
      | VB.length regime == length obs = regime
      | step == 0 = regime VB.++ VB.replicate (length obs - VB.length regime) (VB.head regime)
      | otherwise = error "addGroundTruthValueNode: Should not happen!"
    regime' = VB.zipWith (\reg -> addValueToRegime reg . obsVarianceRegimeValue . fst) regime0 (VB.fromList obs)
    welInp'
      | True || step < 30000 = foldl' addValue welInp (map (obsInputValues . fst) obs)
      | otherwise = welInp
    replaceIndex idx x xs = xs VB.// [(idx, x)]
    getBorder
      | step < 10 = const 0
      | otherwise = (\(mean, _, x) -> mean + sqrt x) . finalize
    writeRegimeFile reg =
      unsafePerformIO $ do
        let txt =
              show step ++
              "\t" ++
              show (obsVarianceRegimeValue $ fst $ head obs) ++
              "\t" ++
              show (fromEnum reg) ++
              "\t" ++
              show (regimeExpSmthFast $ VB.head regime') ++
              "\t" ++ show (regimeExpSmthSlow $ VB.head regime') ++ "\t" ++ show (getBorder $ regimeWelfordAll $ VB.head regime') ++ "\n"
        when (step == 0) $ do writeFile "regime" $ "period\treward\tregime\tExpFast\tExpSlow\tBorder\n"
        appendFile "regime" txt


-- TODO
-- data RegFunction = RegLinear
--   deriving (Eq, Ord, Show, Generic, Serialize, NFData)
-- costFunction :: RegFunction -> VB.Vector Double -> VS.Vector Double -> Double
-- costFunction RegLinear coefs inps =
--   VS.sum (VS.zipWith (*) (VB.convert coefs) inps) + VB.last coefs

-- | Periods after which the heat map will be activated and, hence, unimportant features automatically turned off.
periodsHeatMapActive :: Int
periodsHeatMapActive = 10000

-- | Starting periods in which the there is no regime seperation. To prevent divergence the same regime is used until the specified number of data points are collected per regime.
periodsSharedRegime :: Int
periodsSharedRegime = 10000

-- | Earliest possible training period when training starts.
periodsTrainStart :: Int
periodsTrainStart = 1000

gradDecentMaxIterations :: Int
gradDecentMaxIterations = 500

-- | Train a Regression Node.
trainRegressionNode :: WelfordExistingAggregate (VS.Vector Double) -> Int -> Period -> RegressionNode -> RegressionNode
trainRegressionNode welInp nrNodes period old@(RegressionNode idx m coefs heatMap welOut cfg) =
  if M.null m || VB.length allObs < observationsToUse || period `mod` max 1 ((VS.length coefs - 1) * nrNodes `div` 10) /= 0 -- retrain every ~10% of change values. Note: This does not take number of worker agents into account!
    then old
    else let coefsFiltered = filterHeatMap heatMap coefs
             models = map VS.convert $ regress ys xs (VB.convert coefsFiltered) :: [Model VS.Vector Double]
             learnRate = decaySetup (ExponentialDecay (Just 1e-5) 0.8 30000) period 1
             eiFittedCoefs :: Either RegressionNode (VS.Vector Double)
             eiFittedCoefs = untilThreshold (fromIntegral (VS.length coefsFiltered) * 5e-4) 1 coefsFiltered models
          in case eiFittedCoefs of
               Left regNode -> regNode -- in case we reanable previously disabled features
               Right fittedCoefs ->
                 let coefs' = toCoefficients heatMap $ VS.zipWith (\oldVal newVal -> (1 - learnRate) * oldVal + learnRate * newVal) coefsFiltered fittedCoefs
                     heatMap'
                       | period > periodsHeatMapActive = VS.zipWith (\act v -> abs v >= 0.01 && act) heatMap coefs' VS.// [(VS.length heatMap - 1, True)]
                       | otherwise = heatMap
                  in RegressionNode idx (periodicallyCleanObservations m) coefs' heatMap' welOut cfg
  where
    observationsToUse = VS.length $ obsInputValues $ VB.head $ snd $ M.findMax m
    modelError oldModel newModel = VS.sum $ VS.zipWith (\a b -> abs (a - b)) oldModel newModel
    nonZero x
      | x == 0 = 0.01
      | otherwise = x
    untilThreshold :: Double -> Int -> VS.Vector Double -> [VS.Vector Double] -> Either RegressionNode (VS.Vector Double)
    untilThreshold _ iter lastModel [] = $(pureLogPrintWarning) ("No more models. Default after " ++ show iter) (Right lastModel)
    untilThreshold thresh iter lastModel (new:rest)
      | modelError lastModel new <= thresh =
        if regConfigVerbose cfg
          then $(pureLogPrintWarning) ("ThresholdModel reached. Steps: " ++ show iter) (Right new)
          else Right new
      | null rest = $(pureLogPrintWarning) ("No more models, but threshlastModel not reached. Steps: " ++ show iter) (Right new)
      | iter >= gradDecentMaxIterations =
        if period > 2 * periodsHeatMapActive && VS.any (== False) heatMap
          then $(pureLogPrintWarning)
                 ("Reactivating all features. ThresholdModel of " ++ show thresh ++ " never reached in " ++ show gradDecentMaxIterations ++ " steps: " ++ show (modelError lastModel new))
                 (Left $ trainRegressionNode welInp nrNodes period (RegressionNode idx m (VS.map nonZero coefs) (VS.map (const True) heatMap) welOut cfg))
          else $(pureLogPrintWarning)
                 ("Period " ++ show period ++ ": ThresholdModel of " ++ show thresh ++ " never reached in " ++ show gradDecentMaxIterations ++ " steps: " ++ show (modelError lastModel new))
                 (Right new)
      | otherwise = untilThreshold thresh (iter + 1) new rest
    allObs :: VB.Vector Observation
    allObs = VB.take observationsToUse $ VB.modify (VB.sortBy (comparing (Down . obsPeriod))) $ VB.concat (M.elems m)
    xs :: VB.Vector (VB.Vector Double)
    xs = VB.map (VB.convert . filterHeatMap (VS.init heatMap) . normaliseStateFeatureUnbounded welInp . obsInputValues) allObs -- <- maybe breaks algorithm (different input scaling not preserved? e.g. for indicators?)
    ys :: VB.Vector Double
    ys = VB.map (normaliseUnbounded welOut . obsExpectedOutputValue) allObs
    lastObsPeriod = obsPeriod $ VB.last allObs
    periodicallyCleanObservations m'
      | period `mod` 1000 == 0 = M.filter (not . VB.null) . M.map (VB.filter ((>= lastObsPeriod) . obsPeriod)) $ m'
      | otherwise = m'


-- | Train regression layger (= all nodes).
trainRegressionLayer :: Period -> RegressionLayer -> RegressionLayer
trainRegressionLayer period (RegressionLayer nodes welInp step regime)
  | step < periodsTrainStart = RegressionLayer nodes welInp step regime -- only used for learning the normalization
  | otherwise = RegressionLayer (overBothRegimes (\ns -> VB.map (trainRegressionNode welInp (VB.length ns) period) ns) nodes) welInp step regime


-- | Apply a regression node.
applyRegressionNode :: RegressionNode -> VS.Vector Double -> Double
applyRegressionNode (RegressionNode idx _ coefs heatMap welOut _) inps
  | VS.length coefs - 1 /= VS.length inps = error $ "applyRegressionNode: Expected number of coefficients is not correct: " ++ show (VS.length coefs, VS.length inps)
  | otherwise = denormaliseUnbounded welOut $ VS.sum (VS.zipWith3 (\act w i -> fromAct act * w * i) heatMap (VB.convert coefs) inps) + VS.last coefs
  where
    fromAct True  = 1
    fromAct False = 0


-- | Apply regression layer to given inputs
applyRegressionLayer :: Int -> RegressionLayer -> ActionIndex -> VS.Vector Double -> Double
applyRegressionLayer regId (RegressionLayer nodes welInp step regime) actIdx stateFeat =
  -- trace ("applyRegressionLayer: " ++ show (VB.length regNodes, actIdx))
  withRegime step (currentRegimeExp (regime VB.! regId') ) (\ns -> applyRegressionNode (ns VB.! actIdx) (normaliseStateFeatureUnbounded welInp stateFeat)) nodes
  where regId'
          | regId < VB.length regime = regId -- might no be available on first iteration(s)
          | otherwise = 0
