-- This is the example used by Mahadevan, S. (1996, March). Sensitive discount optimality: Unifying discounted and
-- average reward reinforcement learning. In ICML (pp. 328-336).

{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveAnyClass             #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs               #-}
{-# LANGUAGE OverloadedLists            #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE TupleSections              #-}
{-# LANGUAGE TypeFamilies               #-}
module Main where

import           ML.BORL                as B
import           SolveLp

import           Experimenter

import           Helper

import           Control.Arrow          (first, second)
import           Control.DeepSeq        (NFData)
import           Control.Lens
import           Control.Lens           (set, (^.))
import           Control.Monad          (foldM, liftM, unless, when)
import           Control.Monad.IO.Class (liftIO)
import           Data.List              (genericLength, sort)
import qualified Data.Map.Strict        as M
import           Data.Serialize
import           Data.Text              (Text)
import qualified Data.Text              as T
import           Data.Text.Encoding     as E
import qualified Data.Vector.Storable   as V
import           GHC.Generics
import           GHC.Int                (Int32, Int64)
import           Grenade
import           System.Directory
import           System.IO
import           System.Random

import qualified HighLevelTensorflow    as TF


import           Debug.Trace

----------------------------------------
---------------- Setup -----------------
----------------------------------------

-- Maximum Queue Size
maxQueueSize :: Int
maxQueueSize = 5 -- was 20

-- Setup as in Mahadevan, S. (1996, March). Sensitive discount optimality: Unifying discounted and average reward reinforcement learning. In ICML (pp. 328-336).
lambda, mu, fixedPayoffR, c :: Float
lambda = 5                      -- arrival rate/time
mu = 5                          -- service rate/time
fixedPayoffR = 12               -- fixed payoff
c = 1                           -- holding cost per order

costFunctionF :: Int -> IO Float
costFunctionF j = -- do
  -- x <- randomRIO (0.25, 1.75)
  return $ c * fromIntegral j -- (j+1) - fromIntegral j * x -- holding cost function

----------------------------------------


-- Uniformized M/M/1 State

type CurrentQueueSize = Int
type OrderArrival = Bool

data St = St CurrentQueueSize OrderArrival deriving (Show, Ord, Eq, NFData, Generic, Serialize)

getQueueLength :: St -> CurrentQueueSize
getQueueLength (St len _) = len

instance Enum St where
  fromEnum (St sz arr)      = 2*sz + fromEnum arr
  toEnum x = St (x `div` 2) (toEnum $ x `mod` 2)

instance Bounded St where
  minBound = toEnum 0
  maxBound = toEnum (2 * maxQueueSize + 1)

-- Experiment Setup

expSetup :: BORL St -> ExperimentSetting
expSetup borl =
  ExperimentSetting
    { _experimentBaseName         = "queuing-system M/M/1 eps=5 phase-aware 4" -- "queuing-system M/M/1 eps=5 phase-aware neu"
    , _experimentInfoParameters   = [iMaxQ, iLambda, iMu, iFixedPayoffR, iC, isNN, isTf]
    , _experimentRepetitions      = 40
    , _preparationSteps           = 1000000
    , _evaluationWarmUpSteps      = 0
    , _evaluationSteps            = 100000
    , _evaluationReplications     = 1
    , _maximumParallelEvaluations = 1
    }

  where
    iMaxQ = ExperimentInfoParameter "Maximum queue size" maxQueueSize
    iLambda = ExperimentInfoParameter "Lambda" lambda
    iMu = ExperimentInfoParameter "Mu" mu
    iFixedPayoffR = ExperimentInfoParameter "FixedPayoffR" fixedPayoffR
    iC = ExperimentInfoParameter "Holding Cost c" c
    isNN = ExperimentInfoParameter "Is neural network" (isNeuralNetwork (borl ^. proxies . v))
    isTf = ExperimentInfoParameter "Is tensorflow network" (isTensorflow (borl ^. proxies . v))

evals :: [StatsDef s]
evals =
  -- [ Name "Exp Mean of Repl. Mean Reward" $ Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Stats $ Sum OverPeriods (Of "reward")
  -- , Name "Exp StdDev of Repl. Mean Reward" $ StdDev OverExperimentRepetitions $ Stats $ Mean OverReplications $ Stats $ Sum OverPeriods (Of "reward")
  -- , Name "Average Reward" $ Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Last (Of "avgRew")
  -- , Name "Exp Mean of Repl. Mean QueueLength" $ Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Last (Of "queueLength")
  -- , Name "Exp StdDev of Repl. Mean QueueLength" $ StdDev OverExperimentRepetitions $ Stats $ Mean OverReplications $ Last (Of "queueLength")
  -- ]
  [ Name "Exp Mean of Repl. Mean Reward" $ Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Stats $ Sum OverPeriods (Of "reward")
  , Name "Exp StdDev of Repl. Mean Reward" $ StdDev OverExperimentRepetitions $ Stats $ Mean OverReplications $ Stats $ Sum OverPeriods (Of "reward")
  , Name "Average Reward" $ Mean OverReplications $ First (Of "avgRew")
  , Name "Exp Mean of Average Reward" $ Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ First (Of "avgRew")
  , Name "Exp Mean of Repl. Mean QueueLength" $ Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Stats $ Mean OverPeriods (Of "queueLength")
  -- , Name "Repl. Mean QueueLength" $ Mean OverReplications $ Of "queueLength"
  , Name "Repl. Mean Reward" $ Mean OverReplications $ Stats $ Sum OverPeriods (Of "reward")
  , Name "Repl. Mean QueueLength" $ Mean OverReplications $ Stats $ Mean OverPeriods (Of "queueLength")
  , Name "Exp StdDev of Repl. Mean QueueLength" $ StdDev OverExperimentRepetitions $ Stats $ Mean OverPeriods (Of "queueLength")
  ]
  -- ++
  -- concatMap
  --   (\s -> map (\a ->
  --                 Mean OverReplications $ First (Of $ E.encodeUtf8 $ T.pack $ show (s, a))) (filteredActionIndexes actions actFilter s))
  --   (sort $ take 9 [(minBound :: St) .. maxBound])

instance RewardFuture St where
  type StoreType St = ()

instance BorlLp St where
  lpActions = actions
  lpActionFilter = actFilter

policy :: Int -> Policy St
policy maxAdmit (St s incoming) act
  | not incoming && act == rejectAct =
    [((St (max 0 (s - 1)) False, rejectAct), pMu), ((St s True, condAdmit s), pAdmit s * pLambda), ((St s True, condAdmit s), pReject s * pLambda)]
  | incoming && act == rejectAct = [((St s True, condAdmit s), pAdmit s * pLambda), ((St s True, rejectAct), pReject s * pLambda)] ++ [((St (max 0 (s - 1)) False, rejectAct), pMu)]
  | incoming && act == admitAct =
    [((St (s + 1) True, condAdmit (s + 1)), pAdmit (s + 1) * pLambda), ((St (s + 1) True, rejectAct), pReject (s + 1) * pLambda)] ++ [((St s False, rejectAct), pMu)]
  | otherwise = error "unexpected case in policy"
  where
    pAdmit s
      | s >= maxAdmit = 0
      | otherwise = 1
    pReject s
      | pAdmit s == 1 = 0
      | otherwise = 1
    pMu = realToFrac $ mu / (lambda + mu)
    pLambda = realToFrac $ lambda / (lambda + mu)
    condAdmit s =
      if pAdmit s == 1
        then admitAct
        else rejectAct
    admitAct = actions !! 1
    rejectAct = head actions

instance ExperimentDef (BORL St) where
  type ExpM (BORL St) = IO
  -- type ExpM (BORL St) = TF.SessionT IO
  type InputValue (BORL St) = ()
  type InputState (BORL St) = ()
  type Serializable (BORL St) = BORLSerialisable St
  serialisable = toSerialisable
  deserialisable :: Serializable (BORL St) -> ExpM (BORL St) (BORL St)
  deserialisable = fromSerialisable actions actFilter decay tblInp modelBuilder
  generateInput _ _ _ _ = return ((), ())
  runStep phase rl _ _ = do
      rl' <- stepM rl
      when (rl' ^. t `mod` 10000 == 0) $ liftIO $ prettyBORLHead True (Just mInverseSt) rl' >>= print
      let p = Just $ fromIntegral $ rl' ^. t
          val l = realToFrac (rl' ^?! l)
          results =
            [
              StepResult "queueLength" p (fromIntegral $ getQueueLength $ rl' ^. s)
            , StepResult "reward" p (val $ lastRewards._head)
            ]
            ++
            [ StepResult "avgRew" p (val $ proxies . rho . proxyScalar)
            -- , StepResult "psiRho" p (val $ psis . _1)
            -- , StepResult "psiV" p   (val $ psis . _2)
            -- , StepResult "psiW" p   (val $ psis . _3)
            | phase == EvaluationPhase
            ]
            ++
            concatMap
              (\s ->
                 map (\a -> StepResult (T.pack $ show (s, a)) p (realToFrac $ M.findWithDefault 0 (tblInp s, a) (rl' ^?! proxies . r1 . proxyTable))) (filteredActionIndexes actions actFilter s))
                 (sort $ take 9 $ filter (const (phase == EvaluationPhase))[(minBound :: St) .. maxBound ])
      return (results, rl')
  parameters _ =
    [ ParameterSetup
        "algorithm"
        (set algorithm)
        (view algorithm)
        (Just $ const $ return [ -- AlgDQNAvgRewAdjusted 0.8 0.99  ByStateValues
                               -- ,
                               --   AlgDQNAvgRewAdjusted 0.8 0.999 ByStateValues
                               -- , AlgDQNAvgRewAdjusted 0.8 1.0 ByStateValues
                               -- , AlgDQN 0.99 EpsilonSensitive
                               -- , AlgDQN 0.99 Exact
                               -- , AlgDQN 0.5  EpsilonSensitive
                                AlgDQN 0.5  Exact
                               , AlgDQN 0.999  Exact
                               , AlgDQNAvgRewAdjusted 0.8 0.99 ByStateValues
                               ])
        Nothing
        Nothing
        Nothing
    ]
  beforeEvaluationHook _ _ _ _ rl = return $ set episodeNrStart (0, 0) $ set (B.parameters . exploration) 0.00 $ set (B.settings . disableAllLearning) True rl

nnConfig :: NNConfig
nnConfig =
  NNConfig
    { _replayMemoryMaxSize = 10000
    , _replayMemoryStrategy = ReplayMemorySingle
    , _trainBatchSize = 8
    , _grenadeLearningParams = LearningParameters 0.01 0.0 0.0001
    , _learningParamsDecay = ExponentialDecay (Just 1e-4) 0.05 150000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = ScalingNetOutParameters (-400) 400 (-5000) 5000 (-300) 300 (-300) 300
    , _stabilizationAdditionalRho = 0
    , _stabilizationAdditionalRhoDecay = ExponentialDecay Nothing 0.05 100000
    , _updateTargetInterval = 1 -- 300
    , _updateTargetIntervalDecay = NoDecay


    , _workersMinExploration = []
    }


-- | BORL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha               = 0.01
    , _alphaRhoMin = 2e-5
    , _beta                = 0.01
    , _delta               = 0.005
    , _gamma               = 0.01
    , _epsilon             = 5.0 -- 10.0

    , _exploration         = 1.0
    , _learnRandomAbove    = 1.5
    , _zeta                = 0.03
    , _xi                  = 0.005

    }

-- | Decay function of parameters.
decay :: Decay
decay =
  decaySetupParameters
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-5) 0.5 50000  -- 5e-4
      , _alphaRhoMin = NoDecay
      , _beta             = ExponentialDecay (Just 1e-4) 0.5 150000
      , _delta            = ExponentialDecay (Just 5e-4) 0.5 150000
      , _gamma            = ExponentialDecay (Just 1e-2) 0.5 150000 -- 1e-3
      , _zeta             = ExponentialDecay (Just 0) 0.5 150000
      , _xi               = NoDecay
      -- Exploration
      , _epsilon          = [NoDecay] -- ExponentialDecay (Just 5.0) 0.5 150000
      , _exploration      = ExponentialDecay (Just 0.10) 0.50 100000
      , _learnRandomAbove = NoDecay
      }

initVals :: InitValues
initVals = InitValues 0 0 0 0 0 0

main :: IO ()
main = do
  putStr "Experiment or user mode [User mode]? Enter e for experiment mode, l for lp mode, and u for user mode: " >> hFlush stdout
  l <- getLine
  case l of
    "l"   -> lpMode
    "e"   -> experimentMode
    "exp" -> experimentMode
    _     -> usermode

experimentMode :: IO ()
experimentMode = do
  let databaseSetup = DatabaseSetting "host=192.168.1.110 dbname=ARADRL user=experimenter password=experimenter port=5432" 10
  ---
  rl <- mkUnichainTabular algBORL (liftInitSt initState) tblInp actions actFilter params decay (Just initVals)
  (changed, res) <- runExperiments runMonadBorlIO databaseSetup expSetup () rl
  let runner = runMonadBorlIO
  ---
  -- let mkInitSt = mkUnichainTensorflowCombinedNet alg (liftInitSt initState) netInp actions actFilter params decay modelBuilder nnConfig  (Just initVals)
  -- (changed, res) <- runExperimentsM runMonadBorlTF databaseSetup expSetup () mkInitSt
  -- let runner = runMonadBorlTF
  putStrLn $ "Any change: " ++ show changed
  evalRes <- genEvals runner databaseSetup res evals
     -- print (view evalsResults evalRes)
  writeAndCompileLatex databaseSetup evalRes

  where removeFileIfExists :: FilePath -> IO ()
        removeFileIfExists fp = do
          exists <- doesFileExist fp
          when exists $ removeFile fp
          writeFile fp  ""


lpMode :: IO ()
lpMode = do
  putStrLn "I am solving the system using linear programming to provide the optimal solution...\n"
  l <- putStr "Enter integer! L=" >> hFlush stdout >> read <$> getLine
  lpRes <- runBorlLpInferWithRewardRepetWMax 10 1000 (policy l) mRefStateAct
  print lpRes
  mkStateFile 0.80 True True lpRes
  mkStateFile 0.80 False False lpRes
  putStrLn "NOTE: Above you can see the solution generated using linear programming. Bye!"


mRefStateAct :: Maybe (St, ActionIndex)
-- mRefStateAct = Just (initState, fst $ head $ zip [0..] (actFilter initState))
mRefStateAct = Nothing

alg :: Algorithm St
alg =
        -- AlgDQN 0.99  Exact -- EpsilonSensitive
        -- AlgDQN 0.99 EpsilonSensitive
        -- AlgDQN 0.50  EpsilonSensitive
  AlgDQNAvgRewAdjusted 0.8 1.0 ByStateValues
                -- AlgBORLVOnly ByStateValues mRefStateAct
        -- AlgDQNAvgRewAdjusted 0.8 0.99 ByReward
        -- AlgDQNAvgRewAdjusted 0.8 0.99 ByStateValues
        -- AlgDQNAvgRewAdjusted 0.8 0.99 (ByStateValuesAndReward 1.0 (ExponentialDecay (Just 0.6) 0.9 100000))
        -- AlgBORL 0.5 0.65 ByStateValues mRefStateAct
        -- AlgBORL 0.5 0.65 (Fixed 30) mRefStateAct

allStateInputs :: M.Map NetInputWoAction St
allStateInputs = M.fromList $ zip (map netInp [minBound..maxBound]) [minBound..maxBound]

mInverseSt :: NetInputWoAction -> Maybe (Either String St)
mInverseSt xs = return <$> M.lookup xs allStateInputs

usermode :: IO ()
usermode = do
  writeFile queueLenFilePath "Queue Length\n"

  -- rl <- (randomNetworkInitWith UniformInit :: IO NN) >>= \nn -> mkUnichainGrenade alg (liftInitSt initState) netInp actions actFilter params decay nn nnConfig (Just initVals)
  rl <- do
    case alg of
      AlgBORL{} -> (randomNetworkInitWith UniformInit :: IO NNCombined) >>= \nn -> mkUnichainGrenadeCombinedNet alg (liftInitSt initState) netInp actions actFilter params decay nn nnConfig (Just initVals)
      AlgDQNAvgRewAdjusted{} -> (randomNetworkInitWith UniformInit :: IO NNCombinedAvgFree) >>= \nn -> mkUnichainGrenadeCombinedNet alg (liftInitSt initState) netInp actions actFilter params decay nn nnConfig (Just initVals)
      _ ->  (randomNetworkInitWith UniformInit :: IO NN) >>= \nn -> mkUnichainGrenadeCombinedNet alg (liftInitSt initState) netInp actions actFilter params decay nn nnConfig (Just initVals)

  -- rl <- (randomNetworkInitWith UniformInit :: IO NN) >>= \nn -> mkUnichainGrenade alg (liftInitSt initState) netInp actions actFilter params decay nn nnConfig (Just initVals)
  -- rl <- mkUnichainTensorflow alg (liftInitSt initState) netInp actions actFilter params decay modelBuilder nnConfig  (Just initVals)
  -- rl <- mkUnichainTensorflowCombinedNet alg (liftInitSt initState) netInp actions actFilter params decay modelBuilder nnConfig  (Just initVals)
  -- rl <- mkUnichainTabular alg (liftInitSt initState) tblInp actions actFilter params decay (Just initVals)
  askUser (Just mInverseSt) True usage cmds [] rl
  where cmds = []
        usage = []

type NN = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 2, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 2, 'D1 2]
type NNCombined = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 40, Relu, FullyConnected 40 30, Relu, FullyConnected 30 12, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 40, 'D1 40, 'D1 30, 'D1 30, 'D1 12, 'D1 12]
type NNCombinedAvgFree = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 4, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 4, 'D1 4]


modelBuilder :: TF.ModelBuilderFunction
modelBuilder colOut =
  TF.buildModel $
  TF.inputLayer1D inpLen >>
  TF.fullyConnected [3*(inpLen + nrActs)] TF.relu' >>
  TF.fullyConnected [2*nrActs] TF.relu' >>
  TF.fullyConnected [2*nrActs] TF.relu' >>
  TF.fullyConnected [nrActs, colOut] TF.tanh' >>
  TF.trainingByAdamWith TF.AdamConfig {TF.adamLearningRate = 0.01, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}
  -- trainingByGradientDescent 0.01
  where inpLen = fromIntegral $ V.length (netInp initState)
        nrActs = genericLength actions

netInp :: St -> V.Vector Float
netInp (St len arr) = V.fromList [scaleNegPosOne (0, fromIntegral maxQueueSize) $ fromIntegral len, scaleNegPosOne (0, 1) $ fromIntegral $ fromEnum arr]

tblInp :: St -> V.Vector Float
tblInp (St len arr)        = V.fromList [fromIntegral len, fromIntegral $ fromEnum arr]


names :: [Text]
names = ["reject", "admit "]

initState :: St
initState = St 0 False

queueLenFilePath :: FilePath
queueLenFilePath = "queueLength"

-- Actions
actions :: [Action St]
actions = zipWith Action (map appendQueueLenFile [reject, admit]) names
  where
    appendQueueLenFile f tp st@(St len _) = do
      when (tp == MainAgent) $ appendFile queueLenFilePath (show len ++ "\n")
      f tp st


actFilter :: St -> V.Vector Bool
actFilter (St _ False)  = V.fromList [True, False]
actFilter (St len True) | len >= maxQueueSize  = V.fromList [True, False]
actFilter _             = V.fromList [True, True]

rewardFunction :: St -> ChosenAction -> IO (Reward  St)
rewardFunction (St 0 _) Reject = return $ Reward 0
rewardFunction (St s True) Admit = do
  costFunRes <- costFunctionF (s + 1)
  return $ Reward $ (fixedPayoffR - costFunRes) * (lambda + mu)
rewardFunction (St s _) Reject = do
  costFunRes <- costFunctionF s
  return $ Reward ((0 - costFunRes) * (lambda + mu))


data ChosenAction = Reject | Admit
  deriving (Eq)

reject :: AgentType -> St -> IO (Reward St, St, EpisodeEnd)
reject _ st@(St len True) = do
  reward <- rewardFunction st Reject
  r <- randomRIO (0, 1 :: Float)
  return $ if r <= lambda / (lambda + mu)
    then (reward, St len True, False)              -- new arrival with probability lambda/(lambda+mu)
    else (reward, St (max 0 (len-1)) False, False) -- no new arrival with probability: mu / (lambda+mu)
reject _ st@(St len False) = do
  reward <- rewardFunction st Reject
  r <- randomRIO (0, 1 :: Float) -- case for continue (only the reject action is allowed)
  return $ if r <= lambda / (lambda + mu)
    then (reward,St len True, False)              -- new arrival with probability lambda/(lambda+mu)
    else (reward,St (max 0 (len-1)) False, False) -- processing finished with probability: mu / (lambda+mu)


admit :: AgentType -> St -> IO (Reward St, St, EpisodeEnd)
admit _ st@(St len True) = do
  reward <- rewardFunction st Admit
  r <- randomRIO (0, 1 :: Float)
  return $ if r <= lambda / (lambda + mu)
    then (reward, St (len+1) True, False)  -- admit + new arrival
    else (reward, St len False, False)     -- admit + no new arrival
admit _ _ = error "admit function called with no arrival available. This function is only to be called when an order arrived."
