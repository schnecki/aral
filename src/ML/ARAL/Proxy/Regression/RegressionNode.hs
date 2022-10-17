{-# LANGUAGE DeriveAnyClass    #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE TupleSections     #-}
module ML.ARAL.Proxy.Regression.RegressionNode
    ( RegressionNode (..)
    , RegressionConfig (..)
    , randRegressionNode
    , addGroundTruthValueNode
    , trainRegressionNode
    , applyRegressionNode
    , prettyRegressionNode
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
import           Data.Reflection                                  hiding (int)
import           Data.Serialize
import qualified Data.Vector                                      as VB
import           Data.Vector.Algorithms.Intro                     as VB
import qualified Data.Vector.Storable                             as VS
import           Debug.Trace
import           EasyLogger
import           GHC.Generics
import           Numeric.AD
import           Numeric.Regression.Generic
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
import           ML.ARAL.Proxy.Regression.Observation
import           ML.ARAL.Proxy.Regression.RegressionConfig
import           ML.ARAL.Proxy.Regression.RegressionModel
import           ML.ARAL.Proxy.Regression.VolatilityRegimeExpSmth
import           ML.ARAL.Types


-- | Periods after which the heat map will be activated and, hence, unimportant features automatically turned off.
periodsHeatMapActive :: Int
periodsHeatMapActive = 10000

-- | Maximum number of gradient update iterations per step.
gradDecentMaxIterations :: Int
gradDecentMaxIterations = 500


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


prettyRegressionNode :: Bool -> Maybe (WelfordExistingAggregate (VS.Vector Double)) -> RegressionNode -> Doc
prettyRegressionNode printObs mWelInp (RegressionNode idx m coefs heatMap welOut cfg) =
  text "Node" $$ nest nestCols (int idx) $+$ text "Coefficients" $$ nest nestCols (hcat $ punctuate comma $ map (prettyFractional 3) (VS.toList coefs)) $+$ text "Output scaling" $$
  nest nestCols (text (show welOut)) $+$
  text "Observations" $$
  nest nestCols (int (M.size m) <> text " groups with " <> int (sum $ map VB.length (M.elems m))) $+$
  if printObs
    then nest (nestCols + 10) (vcat $ map prettyObservationVector (M.toList m))
    else mempty
  where
    nestCols = 40
    prettyFractional :: (PrintfArg n) => Int -> n -> Doc
    prettyFractional commas = text . printf ("%+." ++ show commas ++ "f")
    prettyObservationVector :: (Int, VB.Vector Observation) -> Doc
    prettyObservationVector (nr, vec)
      | VB.null vec = headDoc <+> text "empty"
      | otherwise = vcat [headDoc <+> prettyObservation mWelInp (VB.head vec), headDoc <+> prettyObservation mWelInp (VB.last vec)]
      where
        headDoc = prettyFractional 4 (fromIntegral nr * stepSize) <+> parens (int nr) <> colon
    stepSize = regConfigDataOutStepSize cfg


-- Regime helpers


-- | Create new regression node with provided config and given number of input values.
randRegressionNode :: RegressionConfig -> Int -> Int -> IO RegressionNode
randRegressionNode cfg nrInpVals nodeIndex = do
  coefs <- VS.fromList <$> replicateM (regressionNrCoefficients regFun nrInpVals) (randomRIO (-0.05, 0.05 :: Double))
  return $ RegressionNode nodeIndex M.empty coefs (VS.replicate (regressionNrCoefficients regFun nrInpVals) True) WelfordExistingAggregateEmpty cfg
  where regFun = regConfigModel cfg


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
addGroundTruthValueNode period obs@(Observation _ _ _ out) (RegressionNode idx m coefs heatMap welOut cfg@(RegressionConfig step maxObs _ _)) = RegressionNode idx m' coefs heatMap welOut' cfg
  where
    key = floor (normaliseUnbounded welOut out * transf)
    transf = 1 / step
    m' = M.alter (Just . maybe (VB.singleton obs) (VB.take maxObs . (obs `VB.cons`))) key m
    welOut'
      | True || period < 30000 = addValue welOut out
      | otherwise = welOut


-- | Train a Regression Node.
trainRegressionNode :: WelfordExistingAggregate (VS.Vector Double) -> Int -> Period -> RegressionNode -> RegressionNode
trainRegressionNode welInp nrNodes period old@(RegressionNode idx m coefs heatMap welOut cfg) =
  if M.null m || VB.length allObs < observationsToUse || period `mod` max 1 ((VS.length coefs - 1) * nrNodes `div` 10) /= 0 -- retrain every ~10% of change values. Note: This does not take number of worker agents into account!
    then old
    else let coefsFiltered = filterHeatMap heatMap coefs
             -- models = map VS.convert $ regress ys xs (VB.convert coefsFiltered) :: [Model VS.Vector Double]
             regFun = regConfigModel cfg
             models = map VS.convert $ regressOn regFun ys xs (VB.convert coefsFiltered) :: [Model VS.Vector Double]
             learnRate = decaySetup (ExponentialDecay (Just 1e-5) 0.8 30000) period 1
             eiFittedCoefs :: Either RegressionNode (VS.Vector Double)
             eiFittedCoefs = untilThreshold (fromIntegral (VS.length coefsFiltered) * 5e-4) 1 coefsFiltered models
          in case eiFittedCoefs of
               Left regNode -> regNode -- in case we reanable previously disabled features
               -- Right fittedCoefs | VS.length fittedCoefs /= regressionNrCoefficients regFun nrInpVals
               Right fittedCoefs ->
                 let coefs'
                       | VS.length coefsFiltered /= VS.length fittedCoefs = error $ "coefs': Length of filtered and fitted coefs do not coincide: " ++ show (VS.length coefsFiltered, VS.length fittedCoefs)
                       | otherwise = toCoefficients heatMap $ VS.zipWith (\oldVal newVal -> (1 - learnRate) * oldVal + learnRate * newVal) coefsFiltered fittedCoefs
                     heatMap'
                       | period > periodsHeatMapActive = VS.zipWith (\act v -> abs v >= 0.01 && act) heatMap coefs' VS.// [(VS.length heatMap - 1, True)]
                       | otherwise = heatMap
                  in RegressionNode idx (periodicallyCleanObservations m) coefs' heatMap' welOut cfg
  where
    observationsToUse = VS.length $ obsInputValues $ VB.head $ snd $ M.findMax m
    modelError oldModel newModel
      | VS.length oldModel /= VS.length newModel =
        error $ "modelError: Error in length of old and new model: " ++ show (VS.length oldModel, VS.length newModel) ++ " period: " ++ show period
      | otherwise = VS.sum $ VS.zipWith (\a b -> abs (a - b)) oldModel newModel
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
                 ("Reactivating all features. ThresholdModel of " ++
                  show thresh ++ " never reached in " ++ show gradDecentMaxIterations ++ " steps: " ++ show (modelError lastModel new))
                 (Left $ trainRegressionNode welInp nrNodes period (RegressionNode idx m (VS.map nonZero coefs) (VS.map (const True) heatMap) welOut cfg))
          else $(pureLogPrintWarning)
                 ("Period " ++
                  show period ++ ": ThresholdModel of " ++ show thresh ++ " never reached in " ++ show gradDecentMaxIterations ++ " steps: " ++ show (modelError lastModel new))
                 (Right new)
      | otherwise = untilThreshold thresh (iter + 1) new rest
    allObs :: VB.Vector Observation
    allObs = VB.take observationsToUse $ VB.modify (VB.sortBy (comparing (Down . obsPeriod))) $ VB.concat (M.elems m)
    xs :: VB.Vector (VB.Vector Double)
    xs = VB.map (VB.convert . **ERROR: filterHeatMap (VS.init heatMap) . normaliseStateFeatureUnbounded welInp . obsInputValues) allObs -- <- maybe breaks algorithm (different input scaling not preserved? e.g. for indicators?)
    ys :: VB.Vector Double
    ys = VB.map (normaliseUnbounded welOut . obsExpectedOutputValue) allObs
    lastObsPeriod = obsPeriod $ VB.last allObs
    periodicallyCleanObservations m'
      | period `mod` 1000 == 0 = M.filter (not . VB.null) . M.map (VB.filter ((>= lastObsPeriod) . obsPeriod)) $ m'
      | otherwise = m'


-- | Apply a regression node.
applyRegressionNode :: RegressionNode -> VS.Vector Double -> Double
applyRegressionNode (RegressionNode idx _ coefs heatMap welOut cfg) inps
  --  | VS.length coefs /= regressionNrCoefficients regModels = error $ "applyRegressionNode: Expected number of coefficients is not correct: " ++ show (VS.length coefs, VS.length inps)
  | otherwise = denormaliseUnbounded welOut $ compute regModels (VB.convert $ VS.zipWith (\active x -> fromAct active * x) heatMap coefs :: VB.Vector Double) (VB.convert inps)

    -- VS.sum (VS.zipWith3 (\act w i -> fromAct act * w * i) heatMap (VB.convert coefs) inps) + VS.last coefs
  where
    regModels = regFun cfg
    fromAct True  = 1
    fromAct False = 0
