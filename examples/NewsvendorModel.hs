{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveAnyClass             #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs               #-}
{-# LANGUAGE MultiParamTypeClasses      #-}
{-# LANGUAGE OverloadedLists            #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE TupleSections              #-}
{-# LANGUAGE TypeFamilies               #-}
module Main where

import           ML.ARAL                as B
import           SolveLp

import           Experimenter

import           Data.Default
import           Helper

import           Control.Arrow          (first, second, (***))
import           Control.DeepSeq        (NFData)
import           Control.Lens
import           Control.Lens           (set, (^.))
import           Control.Monad          (foldM, liftM, unless, when)
import           Control.Monad.IO.Class (liftIO)
import           Data.Default
import           Data.Function          (on)
import           Data.List              (elemIndex, genericLength, groupBy,
                                         sort, sortBy)
import qualified Data.Map.Strict        as M
import           Data.Serialize
import qualified Data.Text              as T
import           Data.Text.Encoding     as E
import qualified Data.Vector.Storable   as V
import           GHC.Generics
import           GHC.Int                (Int32, Int64)
import           GHC.TypeLits
import           Prelude                hiding (Left, Right)
import           System.IO
import           System.Random

import           Debug.Trace

data Demand = Uniform
  deriving (Show, Eq, Ord)

-- see: Qin, Yan, et al. "The newsvendor problem: Review and directions for future research." European Journal of Operational Research 213.2 (2011): 361-374.


purchasePrice, salvagePrice, lostSalesPrice, retailPrice :: Double
purchasePrice = 0.5 -- v
salvagePrice = 0.25 -- g
lostSalesPrice = 2 -- B
retailPrice = 2.5 -- p

payAtEnd :: Bool -- pay
payAtEnd = False

useEpisodeEnd :: Bool
useEpisodeEnd = True

minDemand, maxDemand :: Int
minDemand = 3
maxDemand = 8

-- Set uniform demand
demand = Uniform


optimum :: (Double -> Double) -> Double
optimum f' = f' ((retailPrice - purchasePrice + lostSalesPrice) / (retailPrice - salvagePrice + lostSalesPrice))

fUniform :: Double -> Double
fUniform x = fromIntegral minDemand + (fromIntegral maxDemand - fromIntegral minDemand) * x

optimumUniform :: Double
optimumUniform = optimum fUniform


-- PROBLEM DESCRIPTION

-- State
newtype St = St Int deriving (Ord, Show, Enum, Eq, NFData, Generic, Serialize)

instance Bounded St where
  minBound = St 0
  maxBound = St (2*maxDemand)


-- Actions
data Act = Buy | Stop
  deriving (Eq, Ord, Enum, Bounded, Generic, NFData, Serialize)

instance Show Act where
  show Buy  = "buy "
  show Stop = "stop"

actions :: [Act]
actions = [Buy, Stop]


actionFun :: ARAL St Act -> AgentType -> St -> [Act] -> IO (Reward St, St, EpisodeEnd)
actionFun _ tp (St nr) [Buy]
  | payAtEnd = return (Reward 0, St (nr + 1), False)
  | otherwise = return (Reward (-purchasePrice), St (nr + 1), False)
actionFun _ tp (St qInt) [Stop] = do
  xInt <-
    case demand of
      Uniform -> randomRIO (minDemand, maxDemand)
  let q = fromIntegral qInt :: Double -- stock
  let x = fromIntegral xInt :: Double -- demand

  -- calc costs
  let purchaseCosts
        | payAtEnd = purchasePrice * q
        | otherwise = 0
  if qInt >= xInt
    then do
      let reward = retailPrice * x - purchaseCosts + salvagePrice * (q - x)
      return (Reward reward, St 0, useEpisodeEnd)
    else do
      let reward = retailPrice * q - purchaseCosts - lostSalesPrice * (x - q)
      return (Reward reward, St 0, useEpisodeEnd)
actionFun _ _ _ xs        = error $ "Multiple/Unexpected actions received in actionFun: " ++ show xs

actFilter :: St -> [V.Vector Bool]
actFilter st = [V.fromList [True, True]]


-- Experementer instance


expSetup :: ARAL St Act -> ExperimentSetting
expSetup borl =
  ExperimentSetting
    { _experimentBaseName = "newsvendor"
    , _experimentInfoParameters = [isNN]
    , _experimentRepetitions = 40
    , _preparationSteps = 500000
    , _evaluationWarmUpSteps = 0
    , _evaluationSteps = 10000
    , _evaluationReplications = 1
    , _evaluationMaxStepsBetweenSaves = Nothing
    }
  where
    isNN = ExperimentInfoParameter "Is neural network" (isNeuralNetwork (borl ^. proxies . v))

evals :: [StatsDef s]
evals =
  [
    Name "Exp Mean of Repl. Mean Reward" $ Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Stats $ Sum OverPeriods (Of "reward")
  , Name "Repl. Mean Reward" $ Mean OverReplications $ Stats $ Sum OverPeriods (Of "reward")
  , Name "Exp StdDev of Repl. Mean Reward" $ StdDev OverExperimentRepetitions $ Stats $ Mean OverReplications $ Stats $ Sum OverPeriods (Of "reward")
  -- , Mean OverExperimentRepetitions $ Stats $ StdDev OverReplications $ Stats $ Sum OverPeriods (Of "reward")
  , Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Last (Of "avgRew")
  , Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Last (Of "avgEpisodeLength")
  , Name "Exp Mean of Repl. Mean Steps to Goal" $ Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Last (Of "avgEpisodeLength")
  , Name "Repl. Mean Steps to Goal" $ Mean OverReplications $ Last (Of "avgEpisodeLength")
  , Name "Exp StdDev of Repl. Mean Steps to Goal" $ StdDev OverExperimentRepetitions $ Stats $ Mean OverReplications $ Last (Of "avgEpisodeLength")
  -- , Mean OverExperimentRepetitions $ Stats $ StdDev OverReplications $ Last (Of "avgEpisodeLength")
  ]
  -- ++
  -- concatMap
  --   (\s -> map (\a -> Mean OverReplications $ First (Of $ E.encodeUtf8 $ T.pack $ show (s, a))) (filteredActionIndexes actions actFilter s))
  --   (sort [(minBound :: St) .. maxBound])
  -- ++
  -- [ Id $ EveryXthElem 10 $ Of "avgRew"
  -- , Mean OverReplications $ EveryXthElem 100 (Of "avgRew")
  -- , StdDev OverReplications $ EveryXthElem 100 (Of "avgRew")
  -- , Mean OverReplications (Stats $ Mean OverPeriods (Of "avgRew"))
  -- , Mean OverReplications $ EveryXthElem 100 (Of "psiRho")
  -- , StdDev OverReplications $ EveryXthElem 100 (Of "psiRho")
  -- , Mean OverReplications $ EveryXthElem 100 (Of "psiV")
  -- , StdDev OverReplications $ EveryXthElem 100 (Of "psiV")
  -- , Mean OverReplications $ EveryXthElem 100 (Of "psiW")
  -- , StdDev OverReplications $ EveryXthElem 100 (Of "psiW")]

  -- where
  --   filterRow f = filter (f . fst . getCurrentIdx)


instance RewardFuture St where
  type StoreType St = ()

-- instance BorlLp St Act where
--   lpActions _ = actions
--   lpActionFunction = actionFun
--   lpActionFilter _ = head . actFilter

-- policy :: Policy St Act
-- policy s a = undefined

-- fakeEpisodes :: ARAL St Act -> ARAL St Act -> ARAL St Act
-- fakeEpisodes rl rl'
--   | rl ^. s == goal && rl ^. episodeNrStart == rl' ^. episodeNrStart = episodeNrStart %~ (\(nr, t) -> (nr + 1, t + 1)) $ rl'
--   | otherwise = episodeNrStart %~ (\(nr, t) -> (nr, t + 1)) $ rl'


-- instance ExperimentDef (ARAL St Act)
--   -- type ExpM (ARAL St Act) = TF.SessionT IO
--                                           where
--   type ExpM (ARAL St Act) = IO
--   type InputValue (ARAL St Act) = ()
--   type InputState (ARAL St Act) = ()
--   type Serializable (ARAL St Act) = ARALSerialisable St Act
--   serialisable = toSerialisable
--   deserialisable :: Serializable (ARAL St Act) -> ExpM (ARAL St Act) (ARAL St Act)
--   deserialisable = fromSerialisable actionFun actFilter tblInp
--   generateInput _ _ _ _ = return ((), ())
--   runStep phase rl _ _ = do
--     rl' <- stepM rl
--     let inverseSt | isAnn rl = Just mInverseSt
--                   | otherwise = Nothing
--     when (rl' ^. t `mod` 10000 == 0) $ liftIO $ prettyARALHead True inverseSt rl' >>= print
--     let (eNr, eSteps) = rl ^. episodeNrStart
--         eLength = fromIntegral eSteps / max 1 (fromIntegral eNr)
--         p = Just $ fromIntegral $ rl' ^. t
--         val l = realToFrac $ head $ fromValue (rl' ^?! l)
--         results | phase /= EvaluationPhase =
--                   [ StepResult "reward" p (realToFrac (rl' ^?! lastRewards._head))
--                   , StepResult "avgEpisodeLength" p eLength
--                   ]
--                 | otherwise =
--                   [ StepResult "reward" p (realToFrac $ rl' ^?! lastRewards._head)
--                   , StepResult "avgRew" p (realToFrac $ V.head (rl' ^?! proxies . rho . proxyScalar))
--                   , StepResult "psiRho" p (val $ psis . _1)
--                   , StepResult "psiV" p (val $ psis . _2)
--                   , StepResult "psiW" p (val $ psis . _3)
--                   , StepResult "avgEpisodeLength" p eLength
--                   , StepResult "avgEpisodeLengthNr" (Just $ fromIntegral eNr) eLength
--                   ] -- ++
--                   -- concatMap
--                   --   (\s ->
--                   --      map (\a -> StepResult (T.pack $ show (s, a)) p (M.findWithDefault 0 (tblInp s, a) (rl' ^?! proxies . r1 . proxyTable))) (filteredActionIndexes actions actFilter s))
--                   --   (sort [(minBound :: St) .. maxBound])
--     return (results, fakeEpisodes rl rl')
--   parameters _ =
--     [ParameterSetup "algorithm" (set algorithm) (view algorithm) (Just $ const $ return
--                                                                   [ AlgARAL 0.8 1.0 ByStateValues
--                                                                   , AlgARAL 0.8 0.999 ByStateValues
--                                                                   , AlgARAL 0.8 0.99 ByStateValues
--                                                                   -- , AlgDQN 0.99 EpsilonSensitive
--                                                                   -- , AlgDQN 0.5 EpsilonSensitive
--                                                                   , AlgDQN 0.999 Exact
--                                                                   , AlgDQN 0.99 Exact
--                                                                   , AlgDQN 0.50 Exact
--                                                                   ]) Nothing Nothing Nothing]
--   beforeEvaluationHook _ _ _ _ rl = return $ set episodeNrStart (0, 0) $ set (B.parameters . exploration) 0.00 $ set (B.settings . disableAllLearning) True rl

nnConfig :: NNConfig
nnConfig =
  NNConfig
    { _replayMemoryMaxSize = 1000
    , _replayMemoryStrategy = ReplayMemorySingle
    , _trainBatchSize = 8
    , _trainingIterations = 1
    , _grenadeLearningParams = OptAdam 0.005 0.9 0.999 1e-8 1e-3
    , _grenadeSmoothTargetUpdate = 0.01
    , _grenadeSmoothTargetUpdatePeriod = 100
    , _learningParamsDecay = ExponentialDecay Nothing 0.05 100000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsRewardAlg (AlgARAL 0.8 1.0 ByStateValues) False 6
    , _scaleOutputAlgorithm = ScaleMinMax
    , _cropTrainMaxValScaled = Just 0.98
    , _grenadeDropoutFlipActivePeriod = 10000
    , _grenadeDropoutOnlyInactiveAfter = 10^5
    , _clipGradients = ClipByGlobalNorm 0.01
    , _autoNormaliseInput = True
    }

borlSettings :: Settings
borlSettings = def
  { _workersMinExploration = [] -- [0.3, 0.2, 0.1]
  , _nStep = 1
  , _independentAgents = 1
  }


-- | ARAL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha               = 0.01
    , _alphaRhoMin = 2e-5
    , _beta                = 0.01
    , _delta               = 0.005
    , _gamma               = 0.01
    , _epsilon             = 0.05

    , _exploration         = 1.0
    , _learnRandomAbove    = 0.7 -- 0 -- 1.0
    , _zeta                = 0.03
    , _xi                  = 0.005

    }

-- | Decay function of parameters.
decay :: ParameterDecaySetting
decay =
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-5) 0.5 75000  -- 5e-4
      , _alphaRhoMin      = NoDecay
      , _beta             = ExponentialDecay (Just 1e-4) 0.5 150000
      , _delta            = ExponentialDecay (Just 5e-4) 0.5 150000
      , _gamma            = ExponentialDecay (Just 1e-3) 0.5 150000 -- 1e-3
      , _zeta             = ExponentialDecay (Just 0) 0.5 150000
      , _xi               = NoDecay
      -- Exploration
      , _epsilon          = [NoDecay] -- ExponentialDecay (Just 5.0) 0.5 150000
      , _exploration      = ExponentialDecay (Just 0.01) 0.50 75000
      , _learnRandomAbove = NoDecay
      }

initVals :: InitValues
initVals = InitValues (-100) (-100) 0 0 0 0

main :: IO ()
main = do
  let opt =
        case demand of
          Uniform -> optimumUniform
  putStrLn $ "\n\nOptimum: " ++ show opt ++ "\n"
  putStr "Experiment or user mode [User mode]? Enter e for experiment mode, l for lp mode, and u for user mode: " >> hFlush stdout
  l <- getLine
  case l of
    "l"   -> lpMode
    "e"   -> experimentMode
    "exp" -> experimentMode
    _     -> usermode


experimentMode :: IO ()
experimentMode = undefined
  -- do
  -- let databaseSetup = DatabaseSetting "host=192.168.1.110 dbname=ARADRL user=experimenter password=experimenter port=5432" 10
  -- ---
  -- rl <- mkUnichainTabular algARAL (liftInitSt initState) tblInp actionFun actFilter params decay borlSettings (Just initVals)
  -- (changed, res) <- runExperiments liftIO databaseSetup expSetup () rl
  -- let runner = liftIO
  -- ---
  -- putStrLn $ "Any change: " ++ show changed
  -- evalRes <- genEvalsConcurrent 6 runner databaseSetup res evals
  --    -- print (view evalsResults evalRes)
  -- writeAndCompileLatex databaseSetup evalRes
  -- writeCsvMeasure databaseSetup res NoSmoothing ["reward", "avgEpisodeLength"]


lpMode :: IO ()
lpMode = do
  putStrLn "I am solving the system using linear programming to provide the optimal solution...\n"
  undefined
  -- lpRes <- runBorlLpInferWithRewardRepet 100000 policy mRefState
  -- print lpRes
  -- mkStateFile 0.65 False True lpRes
  -- mkStateFile 0.65 False False lpRes
  -- putStrLn "NOTE: Above you can see the solution generated using linear programming. Bye!"


mRefState :: Maybe (St, ActionIndex)
mRefState = Nothing
-- mRefState = Just (fromIdx (goalX, goalY), 0)


usermode :: IO ()
usermode = do

  -- Approximate all fucntions using a single neural network
  -- rl <- mkUnichainHasktorch alg (liftInitSt initState) netInp actionFun actFilter params decay modelBuilderHasktorch nnConfig borlSettings (Just initVals)

  -- Use a table to approximate the function (tabular version)
  alg <- chooseAlg Nothing
  rl <- mkUnichainTabular alg (liftInitSt initState) tblInp actionFun actFilter params decay borlSettings (Just initVals)
  let inverseSt | isAnn rl = Just mInverseSt
                | otherwise = Nothing

  askUser inverseSt True usage cmds [] rl -- maybe increase learning by setting estimate of rho
  where
    cmds = zipWith (\u a -> (fst u, maybe [0] return (elemIndex a actions))) usage [Buy, Stop]
    usage = [("b", "Buy"), ("s", "Stop")]


modelBuilderHasktorch :: Integer -> (Integer, Integer) -> MLPSpec
modelBuilderHasktorch lenIn (lenActs, cols) = MLPSpec [lenIn, 20, 10, 10, lenOut] (HasktorchActivation HasktorchRelu []) (Just HasktorchTanh)
  where
    lenOut = lenActs * cols


netInp :: St -> V.Vector Double
netInp (St nr) = V.fromList [scaleMinMax (0, fromIntegral maxDemand) $ fromIntegral nr]

allStateInputs :: M.Map NetInputWoAction St
allStateInputs = M.fromList $ zip (map netInp [minBound..maxBound]) [minBound..maxBound]

mInverseSt :: NetInputWoAction -> Maybe (Either String St)
mInverseSt xs = return <$> M.lookup xs allStateInputs


tblInp :: St -> V.Vector Double
tblInp (St nr) = V.fromList [fromIntegral nr]

initState :: St
initState = St 0
