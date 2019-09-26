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

import           ML.BORL
import           SolveLp

import           Experimenter

import           Helper

import           Control.Arrow          (first, second)
import           Control.DeepSeq        (NFData)
import           Control.Lens
import           Control.Lens           (set, (^.))
import           Control.Monad          (foldM, liftM, unless, when)
import           Control.Monad.IO.Class (liftIO)
import           Data.List              (genericLength)
import           Data.Serialize
import           Data.Text              (Text)
import           GHC.Generics
import           GHC.Int                (Int32, Int64)
import           Grenade
import           System.IO
import           System.Random


import qualified TensorFlow.Build       as TF (addNewOp, evalBuildT, explicitName, opDef,
                                               opDefWithName, opType, runBuildT, summaries)
import qualified TensorFlow.Core        as TF hiding (value)
-- import qualified TensorFlow.GenOps.Core                         as TF (square)
import qualified TensorFlow.GenOps.Core as TF (abs, add, approximateEqual,
                                               approximateEqual, assign, cast,
                                               getSessionHandle, getSessionTensor,
                                               identity', lessEqual, matMul, mul,
                                               readerSerializeState, relu, relu', shape,
                                               square, sub, tanh, tanh', truncatedNormal)
import qualified TensorFlow.Minimize    as TF
-- import qualified TensorFlow.Ops                                 as TF (abs, add, assign,
--                                                                        cast, identity',
--                                                                        matMul, mul, relu,
--                                                                        sub,
--                                                                        truncatedNormal)
import qualified TensorFlow.Ops         as TF (initializedVariable, initializedVariable',
                                               placeholder, placeholder', reduceMean,
                                               reduceSum, restore, save, scalar, vector,
                                               zeroInitializedVariable,
                                               zeroInitializedVariable')
import qualified TensorFlow.Session     as TF
import qualified TensorFlow.Tensor      as TF (Ref (..), collectAllSummaries,
                                               tensorNodeName, tensorRefFromName,
                                               tensorValueFromName)


import           Debug.Trace

----------------------------------------
---------------- Setup -----------------
----------------------------------------

-- Maximum Queue Size
maxQueueSize :: Int
maxQueueSize = 5

-- Setup as in Mahadevan, S. (1996, March). Sensitive discount optimality: Unifying discounted and average reward reinforcement learning. In ICML (pp. 328-336).
lambda, mu, fixedPayoffR, c :: Double
lambda = 5                      -- arrival rate/time
mu = 5                          -- service rate/time
fixedPayoffR = 12               -- fixed payoff
c = 1                           -- holding cost per order

costFunctionF :: Int -> Double
costFunctionF j = c * fromIntegral j -- holding cost function

----------------------------------------


-- Uniformized M/M/1 State

type CurrentQueueSize = Int
type OrderArrival = Bool

data St = St CurrentQueueSize OrderArrival deriving (Show, Ord, Eq, NFData, Generic, Serialize)

instance Enum St where
  fromEnum (St sz arr) = 2*sz + fromEnum arr
  toEnum x = St (x `div` 2) (toEnum $ x `mod` 2)

instance Bounded St where
  minBound = toEnum 0
  maxBound = toEnum (2*maxQueueSize + 1)


expSetup :: BORL St -> ExperimentSetting
expSetup borl =
  ExperimentSetting
    { _experimentBaseName         = "queuing-system M/M/1"
    , _experimentInfoParameters   = [iMaxQ, iLambda, iMu, iFixedPayoffR, iC, isNN, isTf]
    , _experimentRepetitions      = 1
    , _preparationSteps           = 500000
    , _evaluationWarmUpSteps      = 0
    , _evaluationSteps            = 10000
    , _evaluationReplications     = 3
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
  [ Id $ EveryXthElem 10 $ Of "avgRew"
  , Mean OverReplications $ EveryXthElem 100 (Of "avgRew")
  , StdDev OverReplications $ EveryXthElem 100 (Of "avgRew")
  , Mean OverReplications (Stats $ Mean OverPeriods (Of "avgRew"))
  , Mean OverReplications $ EveryXthElem 100 (Of "psiRho")
  , StdDev OverReplications $ EveryXthElem 100 (Of "psiRho")
  , Mean OverReplications $ EveryXthElem 100 (Of "psiV")
  , StdDev OverReplications $ EveryXthElem 100 (Of "psiV")
  , Mean OverReplications $ EveryXthElem 100 (Of "psiW")
  , StdDev OverReplications $ EveryXthElem 100 (Of "psiW")
  , Mean OverReplications $ Last (Of "avgEpisodeLength")
  , StdDev OverReplications $ Last (Of "avgEpisodeLength")
  ]

instance RewardFuture St where
  type StoreType St = ()

instance BorlLp St where
  lpActions = actions
  lpActionFilter = actFilter

policy :: Int -> Policy St
policy maxAdmit (St s incoming) act
  | not incoming && act == rejectAct =
    [((St (max 0 (s - 1)) False, rejectAct), pMu)] ++ [((St s True, condAdmit s), pAdmit s * pLambda), ((St s True, condAdmit s), pReject s * pLambda)]
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
    pMu = mu / (lambda + mu)
    pLambda = lambda / (lambda + mu)
    condAdmit s =
      if pAdmit s == 1
        then admitAct
        else rejectAct
    admitAct = actions !! 1
    rejectAct = head actions


instance ExperimentDef (BORL St)
  -- type ExpM (BORL St) = TF.SessionT IO
                                          where
  type ExpM (BORL St) = IO
  type InputValue (BORL St) = ()
  type InputState (BORL St) = ()
  type Serializable (BORL St) = BORLSerialisable St
  serialisable = toSerialisable
  deserialisable :: Serializable (BORL St) -> ExpM (BORL St) (BORL St)
  deserialisable = fromSerialisable actions actFilter decay netInp netInp modelBuilder
  generateInput _ _ _ _ = return ((), ())
  runStep rl _ _ =
    liftIO $ do
      rl' <- stepM rl
      when (rl' ^. t `mod` 10000 == 0) $ liftSimple $ prettyBORLHead True rl' >>= print
      let (eNr, eStart) = rl ^. episodeNrStart
          eLength = fromIntegral eStart / fromIntegral eNr
          results =
            [ StepResult "avgRew" (Just $ fromIntegral $ rl' ^. t) (rl' ^?! proxies . rho . proxyScalar)
            , StepResult "psiRho" (Just $ fromIntegral $ rl' ^. t) (rl' ^?! psis . _1)
            , StepResult "psiV" (Just $ fromIntegral $ rl' ^. t) (rl' ^?! psis . _2)
            , StepResult "psiW" (Just $ fromIntegral $ rl' ^. t) (rl' ^?! psis . _3)
            , StepResult "avgEpisodeLength" (Just $ fromIntegral $ rl' ^. t) eLength
            , StepResult "avgEpisodeLengthNr" (Just $ fromIntegral eNr) eLength
            ]
      return (results, rl')
  parameters _ =
    [ ParameterSetup
        "algorithm"
        (set algorithm)
        (view algorithm)
        (Just $ const $
         return
           [ AlgBORL defaultGamma0 defaultGamma1 (ByMovAvg 3000) Normal False Nothing
           , AlgBORL defaultGamma0 defaultGamma1 (ByMovAvg 3000) Normal True Nothing
           , AlgBORLVOnly (ByMovAvg 3000) Nothing
           ])
        Nothing
        Nothing
        Nothing
    ]

-- | BORL Parameters.
params :: Parameters
params =
  Parameters
    { _alpha              = 0.03
    , _alphaANN           = 0.5
    , _beta               = 0.02
    , _betaANN            = 1
    , _delta              = 0.01
    , _deltaANN           = 1
    , _gamma              = 0.01
    , _gammaANN           = 1
    , _epsilon            = 0.2
    , _exploration        = 0.8
    , _learnRandomAbove   = 0.0
    , _zeta               = 0.0
    , _xi                 = 0.05 -- 75 -- 0.1
    , _disableAllLearning = False
    }

-- | Decay function of parameters.
decay :: Decay
decay t = exponentialDecay (Just minValues) 0.50 100000 t
  where
    minValues =
      Parameters
        { _alpha = 0.0001
        , _alphaANN = 0.5
        , _beta = 0.0001
        , _betaANN = 1.0
        , _delta = 0.0001
        , _deltaANN = 1.0
        , _gamma = 0.001
        , _gammaANN = 1.0
        , _epsilon = 5
        , _exploration = 0.005
        , _learnRandomAbove = 0.0
        , _zeta = 0.0
        , _xi = 0.05
        , _disableAllLearning = False
        }

initVals :: InitValues
initVals = InitValues 30 0 0 0 0

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
  let databaseSetup = DatabaseSetting "host=localhost dbname=experimenter2 user=experimenter password= port=5432" 10
  let rl = mkUnichainTabular algBORL initState netInp actions actFilter params decay Nothing
  (changed, res) <- runExperiments runMonadBorlIO databaseSetup expSetup () rl
  let runner = runMonadBorlIO
  -- let mkInitSt = mkUnichainTensorflowM algBORL initState netInp actions actFilter params decay modelBuilder nnConfig (Just initVals)
  -- (changed, res) <- runExperimentsM runMonadBorlTF databaseSetup expSetup () mkInitSt
  -- let runner = runMonadBorlTF
  putStrLn $ "Any change: " ++ show changed
  evalRes <- genEvals runner databaseSetup res evals
     -- print (view evalsResults evalRes)
  writeAndCompileLatex evalRes


lpMode :: IO ()
lpMode = do
  putStrLn "I am solving the system using linear programming to provide the optimal solution beforehand...\n"
  l <- putStr "Enter integer! L=" >> hFlush stdout >> read <$> getLine
  runBorlLpInferWithRewardRepet 1000 (policy l) >>= print
  putStrLn "NOTE: Above you can see the solution generated using linear programming. Bye!"


usermode :: IO ()
usermode = do
  writeFile queueLenFilePath "Queue Length\n"
  let algorithm =
        -- AlgDQN 0.99             -- does not work
        -- AlgDQN 0.50             -- does work
        -- AlgBORLVOnly (ByMovAvg 5000) (Just (initState, fst $ head $ zip [0..] (actFilter initState)))

        AlgBORL 0.5 0.8 (ByMovAvg 5000) Normal False (Just (initState, fst $ head $ zip [0..] (actFilter initState)))

  -- nn <- randomNetworkInitWith UniformInit :: IO NN
  -- rl <- mkUnichainGrenade algorithm initState netInp actions actFilter params decay nn nnConfig (Just initVals)
  -- rl <- mkUnichainTensorflow algorithm initState netInp actions actFilter params decay modelBuilder nnConfig  (Just initVals)
  let rl = mkUnichainTabular algorithm initState tblInp actions actFilter params decay (Just initVals)
  askUser True usage cmds rl
  where cmds = []
        usage = []

type NN = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 2, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 2, 'D1 2]

modelBuilder :: (TF.MonadBuild m) => m TensorflowModel
modelBuilder =
  buildModel $
  inputLayer1D inpLen >> fullyConnected1D (5*inpLen) TF.relu' >> fullyConnected1D (3*inpLen) TF.relu' >> fullyConnected1D (2*inpLen) TF.relu' >> fullyConnected1D (genericLength actions) TF.tanh' >>
  trainingByAdam1DWith TF.AdamConfig {TF.adamLearningRate = 0.005, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}
  where inpLen = genericLength (netInp initState)


nnConfig :: NNConfig
nnConfig = NNConfig
  { _replayMemoryMaxSize  = 10000
  , _trainBatchSize       = 32
  , _grenadeLearningParams = LearningParameters 0.01 0.9 0.0001
  , _prettyPrintElems     = map netInp ([minBound .. maxBound] :: [St])
  , _scaleParameters      = scalingByMaxAbsReward False 6
  , _updateTargetInterval = 3000
  , _trainMSEMax          = Just 0.03
  }

netInp :: St -> [Double]
netInp (St len arr) = [scaleNegPosOne (0, fromIntegral maxQueueSize) $ fromIntegral len, scaleNegPosOne (0, 1) $ fromIntegral $ fromEnum arr]

tblInp :: St -> [Double]
tblInp (St len arr) = [fromIntegral len, fromIntegral $ fromEnum arr]


names :: [Text]
names = ["reject", "admit "]

initState :: St
initState = St 0 False

queueLenFilePath :: FilePath
queueLenFilePath = "queueLength"

-- Actions
actions :: [Action St]
actions = zipWith Action (map appendQueueLenFile [reject, admit]) names

  where appendQueueLenFile f st@(St len _) = do
          appendFile queueLenFilePath (show len ++ "\n")
          f st


actFilter :: St -> [Bool]
actFilter (St _ False)  = [True, False]
actFilter (St len True) | len >= maxQueueSize = [True, False]
actFilter _             = [True, True]

rewardFunction :: St -> ChosenAction -> Reward  St
rewardFunction (St 0 _) Reject = Reward 0
rewardFunction (St s True) Admit = Reward $ (fixedPayoffR - costFunctionF (s + 1)) * (lambda + mu)
rewardFunction (St s _) Reject = Reward ((0 - costFunctionF(s)) * (lambda + mu))


data ChosenAction = Reject | Admit
  deriving (Eq)

reject :: St -> IO (Reward St, St, EpisodeEnd)
reject st@(St len True) = do
  let reward = rewardFunction st Reject
  r <- randomRIO (0, 1 :: Double)
  return $ if r <= lambda / (lambda + mu)
    then (reward, St len True, False)              -- new arrival with probability lambda/(lambda+mu)
    else (reward, St (max 0 (len-1)) False, False) -- no new arrival with probability: mu / (lambda+mu)
reject st@(St len False) = do
  let reward = rewardFunction st Reject
  r <- randomRIO (0, 1 :: Double) -- case for continue (only the reject action is allowed)
  return $ if r <= lambda / (lambda + mu)
    then (reward,St len True, False)              -- new arrival with probability lambda/(lambda+mu)
    else (reward,St (max 0 (len-1)) False, False) -- processing finished with probability: mu / (lambda+mu)

admit :: St -> IO (Reward St, St, EpisodeEnd)
admit st@(St len True) = do
  let reward = rewardFunction st Admit
  r <- randomRIO (0, 1 :: Double)
  return $ if r <= lambda / (lambda + mu)
    then (reward, St (len+1) True, False) -- admit + new arrival
    else (reward, St len False, False)    -- admit + no new arrival
admit _ = error "admit function called with no arrival available. This function is only to be called when an order arrived."