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
    , regNodeOutWelford   :: !(WelfordExistingAggregate Double)   -- ^ Output scaling.
    , regNodeConfig       :: !RegressionConfig                    -- ^ Configuration.
    }
  deriving (Eq, Show, Generic, Serialize, NFData)


prettyRegressionNode :: Bool -> Maybe (WelfordExistingAggregate (VS.Vector Double)) -> RegressionNode -> Doc
prettyRegressionNode printObs mWelInp (RegressionNode idx m coefs welOut cfg) =
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
  coefs <- VS.fromList <$> replicateM nrCoefs (randomRIO (-xavier, xavier :: Double))
  return $ RegressionNode nodeIndex M.empty coefs WelfordExistingAggregateEmpty cfg
  where
    nrCoefs = nrCoefsRegressionModels regFun nrInpVals
    regFun = regConfigModel cfg
    -- W_l,i ~ U(-sqrt (6/(n_l+n_{l+1})),sqrt (6/(n_l+n_{l+1}))) where n_l is the number of nodes in layer l
    xavier = sqrt 6 / sqrt (fromIntegral nrInpVals + fromIntegral nrCoefs)


-- | Add ground truth value to specific node.
addGroundTruthValueNode :: Period -> Observation -> RegressionNode -> RegressionNode
addGroundTruthValueNode period obs@(Observation _ _ _ out) (RegressionNode idx m coefs welOut cfg) = RegressionNode idx m' coefs welOut' cfg
  where
    key = round (normaliseUnbounded welOut out * transf)
    transf = 1 / regConfigDataOutStepSize cfg
    maxObs = regConfigDataMaxObservationsPerStep cfg
    m' = M.alter (Just . maybe (VB.singleton obs) (VB.take maxObs . (obs `VB.cons`))) key m
    welOut'
      | period < 30000 = addValue welOut out
      | otherwise = welOut


-- | Train a Regression Node.
trainRegressionNode :: WelfordExistingAggregate (VS.Vector Double) -> Int -> Period -> RegressionNode -> RegressionNode
trainRegressionNode welInp nrNodes period old@(RegressionNode idx m coefs welOut cfg) =
  if M.null m || VB.length allObs < nrObservationsToUse || period `mod` max 1 ((VS.length coefs - 1) * nrNodes `div` 10) /= 0 -- retrain every ~10% of change values. Note: This does not take number of worker agents into account!
    then old
    else let models = map VS.convert $ regressOn (computeModels regModels) ys xs (VB.convert coefs) :: [Model VS.Vector Double]
             learnRate = decaySetup (regConfigLearnRateDecay cfg) period (regConfigLearnRate0 cfg)
             threshold = decaySetup (ExponentialDecay (Just $ fromIntegral (VS.length coefs) * 1e-4) 0.8 30000) period (fromIntegral (VS.length coefs) * 5e-4)
             minCorr = regConfigMinCorrelation cfg
             eiFittedCoefs :: Either RegressionNode (VS.Vector Double)
             eiFittedCoefs = untilThreshold threshold 1 coefs models
          in case eiFittedCoefs of
               Left regNode -> regNode -- in case we reanable previously disabled features
               Right fittedCoefs ->
                 let coefs' = VS.zipWith (\oldVal newVal -> (1 - learnRate) * oldVal + learnRate * newVal) coefs fittedCoefs
                     coefs''
                       | period > periodsHeatMapActive = VS.map (\v -> if abs v >= minCorr then v else 0) (VS.init coefs') `VS.snoc` VS.last coefs'
                       | otherwise = coefs'
                  in RegressionNode idx (periodicallyCleanObservations m) coefs'' welOut cfg
  where
    regModels = regConfigModel cfg
    lenInp = VS.length (welfordMean welInp)
    nrObservationsToUse = lenInp `div` 3
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
        -- if False && period > 2 * periodsHeatMapActive && VS.any (== 0) coefs
        --   then $(pureLogPrintWarning)
        --          ("Reactivating all features. ThresholdModel of " ++
        --           show thresh ++ " never reached in " ++ show gradDecentMaxIterations ++ " steps: " ++ show (modelError lastModel new))
        --          (Left $ trainRegressionNode welInp nrNodes period (RegressionNode idx m (VS.map nonZero coefs) welOut cfg))
        --   else
          $(pureLogPrintWarning)
                 ("Period " ++
                  show period ++ ": ThresholdModel of " ++ show thresh ++ " never reached in " ++ show gradDecentMaxIterations ++ " steps: " ++ show (modelError lastModel new))
                 (Right new)
      | otherwise = untilThreshold thresh (iter + 1) new rest
    lastObs = VB.take nrObservationsToUse (VB.modify (VB.sortBy (comparing (Down . obsPeriod))) $ VB.concat (M.elems m))
    obsCache = M.toList m
    invTransf = regConfigDataOutStepSize cfg
    obsRandIdx :: [VB.Vector Int]
    obsRandIdx = unsafePerformIO $ mapM (fmap VB.fromList . randomPick) obsCache
    randomPick (key, vec) = do
      r <- randomRIO (0, 1::Double) -- ensure that we don't always feed all the data
      if r < 0.67
        then return []
        else randomRIO (1, maxNr key vec) >>= \nr -> replicateM nr (randomRIO (0, VB.length vec - 1))
    maxNr key vec = min (VB.length vec) . max 1 $ round $ fromIntegral (abs key) * invTransf
    allObs :: VB.Vector Observation
    allObs = lastObs VB.++ VB.concat (zipWith (\(_, obs) -> VB.map (obs VB.!)) obsCache obsRandIdx)
    xs :: VB.Vector (VB.Vector Double)
    xs = VB.map (VB.convert . normaliseStateFeatureUnbounded welInp . obsInputValues) allObs -- <- maybe breaks algorithm (different input scaling not preserved? e.g. for indicators?)
    ys :: VB.Vector Double
    ys = VB.map (normaliseUnbounded welOut . obsExpectedOutputValue) allObs
    lastObsPeriod = obsPeriod $ VB.last lastObs
    headObsPeriod = obsPeriod $ VB.head lastObs
    obsPeriods = headObsPeriod - lastObsPeriod
    periodicallyCleanObservations m'
      --  | period `mod` 1000 == 0 = M.filter (not . VB.null) . M.map (VB.filter ((>= 10 * obsPeriods) . obsPeriod)) $ m'
      | otherwise = m'


-- | Apply a regression node.
applyRegressionNode :: RegressionNode -> VS.Vector Double -> Double
applyRegressionNode (RegressionNode _ _ coefs welOut cfg) inps = denormaliseUnbounded welOut $ computeModels regModels (VB.convert coefs :: VB.Vector Double) (VB.convert inps)
  where
    regModels = regConfigModel cfg
