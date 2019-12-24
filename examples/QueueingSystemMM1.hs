-- This is the example used by Mahadevan, S. (1996, March). Sensitive discount optimality: Unifying discounted and
-- average reward reinforcement learning. In ICML (pp. 328-336).

{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveAnyClass             #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs               #-}
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
import qualified Data.Map.Strict        as M
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
maxQueueSize = 4

-- Setup as in Mahadevan, S. (1996, March). Sensitive discount optimality: Unifying discounted and average reward reinforcement learning. In ICML (pp. 328-336).
lambda, mu, fixedPayoffR, c :: Double
lambda = 5                      -- arrival rate/time
mu = 5                          -- service rate/time
fixedPayoffR = 12               -- fixed payoff
c = 1                           -- holding cost per order

costFunctionF :: Int -> IO Double
costFunctionF j = do
  -- x <- randomRIO (0.25, 1.75)
  return $ c * fromIntegral j -- (j+1) - fromIntegral j * x -- holding cost function

----------------------------------------


-- Uniformized M/M/1 State

type CurrentQueueSize = Int
type OrderArrival = Bool

data St = St CurrentQueueSize OrderArrival deriving (Show, Ord, Eq, NFData, Generic, Serialize)

instance Enum St where
  fromEnum (St sz arr)      = 2*sz + fromEnum arr
  toEnum x = St (x `div` 2) (toEnum $ x `mod` 2)

instance Bounded St where
  minBound = toEnum 0
  maxBound = toEnum (2 * maxQueueSize + 1)


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
    pMu = mu / (lambda + mu)
    pLambda = lambda / (lambda + mu)
    condAdmit s =
      if pAdmit s == 1
        then admitAct
        else rejectAct
    admitAct = actions !! 1
    rejectAct = head actions

instance ExperimentDef (BORL St)
                                          where
  type ExpM (BORL St) = IO
  -- type ExpM (BORL St) = TF.SessionT IO
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
      when (rl' ^. t `mod` 10000 == 0) $ liftIO $ prettyBORLHead True (Just mInverseSt) rl' >>= print
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
           [ AlgBORL defaultGamma0 defaultGamma1 (ByMovAvg 3000)  Nothing
           , AlgBORL defaultGamma0 defaultGamma1 (ByMovAvg 3000) Nothing
           , AlgBORLVOnly (ByMovAvg 3000) Nothing
           ])
        Nothing
        Nothing
        Nothing
    ]

nnConfig :: NNConfig
nnConfig =
  NNConfig
    { _replayMemoryMaxSize = 10000
    , _trainBatchSize = 8
    , _grenadeLearningParams = LearningParameters 0.01 0.0 0.0001
    , _learningParamsDecay = ExponentialDecay (Just 1e-4) 0.05 150000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = ScalingNetOutParameters (-400) 400 (-5000) 5000 (-300) 300 (-300) 300
    , _stabilizationAdditionalRho = 0
    , _stabilizationAdditionalRhoDecay = ExponentialDecay Nothing 0.05 100000
    , _updateTargetInterval = 1 -- 300
    , _trainMSEMax = Nothing    -- Just 0.05
    , _setExpSmoothParamsTo1 = True
    }


-- | BORL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha              = 0.01
    , _beta               = 0.03
    , _delta              = 0.03
    , _gamma              = 0.005
    , _epsilon            = 5
    , _explorationStrategy = EpsilonGreedy
    , _exploration        = 1.0
    , _learnRandomAbove   = 0.10
    , _zeta               = 0.10
    , _xi                 = 5e-3
    , _disableAllLearning = False
    -- ANN
    , _alphaANN           = 0.5 -- only used for multichain
    , _betaANN            = 1
    , _deltaANN           = 1
    , _gammaANN           = 1

    }

-- | Decay function of parameters.
decay :: Decay
decay =
  decaySetupParameters
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-4) 0.15 10000
      , _beta             = ExponentialDecay (Just 1e-4) 0.5 150000
      , _delta            = ExponentialDecay (Just 5e-4) 0.5 150000
      , _gamma            = ExponentialDecay (Just 1e-3) 0.5 150000
      , _zeta             = NoDecay -- ExponentialDecay (Just 0) 0.5 150000
      , _xi               = NoDecay
      -- Exploration
      , _epsilon          = NoDecay
      , _exploration      = ExponentialDecay (Just 0.10) 0.5 150000
      , _learnRandomAbove = NoDecay
      -- ANN
      , _alphaANN         = ExponentialDecay Nothing 0.75 150000
      , _betaANN          = ExponentialDecay Nothing 0.75 150000
      , _deltaANN         = ExponentialDecay Nothing 0.75 150000
      , _gammaANN         = ExponentialDecay Nothing 0.75 150000
      }
  --  | isAlgDqnAvgRewardFree alg || isAlgBorlVOnly alg = overrideDecayParameters t
  --  -   [ (alpha, 0.5, 30000, 0.00001)
  --     , (alphaANN, 0.5, 30000, 0.00001)
  --     , (beta, 0.5, 30000, 0.0001)
  --     , (betaANN, 0.5, 30000, 0.0001)
  --     , (delta, 0.5, 30000, 0.0001)
  --     , (deltaANN, 0.5, 30000, 0.0001)
  --     , (gamma, 0.5, 30000, 0.0001)
  --     , (gammaANN, 0.5, 30000, 0.0001)
  --     , (exploration, 0.5, 30000, 0.01)
  --     ] p
  --   -- overrideDecayParameters t [(beta, beta, 0.5, 150000, 1e-4), (delta, delta, 0.5, 150000, 1e-4)] p $
  --   exponentialDecayParameters (Just minValues) 0.75 150000 t p
  -- where
  --   m = 1e-3
  --   minValues =

initVals :: InitValues
initVals = InitValues 0 0 0 0 0

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
  writeAndCompileLatex databaseSetup evalRes


lpMode :: IO ()
lpMode = do
  putStrLn "I am solving the system using linear programming to provide the optimal solution...\n"
  l <- putStr "Enter integer! L=" >> hFlush stdout >> read <$> getLine
  runBorlLpInferWithRewardRepet 1000 (policy l) mRefStateAct >>= print
  putStrLn "NOTE: Above you can see the solution generated using linear programming. Bye!"


mRefStateAct :: Maybe (St, ActionIndex)
-- mRefStateAct = Just (initState, fst $ head $ zip [0..] (actFilter initState))
mRefStateAct = Nothing

alg :: Algorithm St
alg =
        -- AlgDQN 0.99
        -- AlgDQN 0.50
        -- AlgDQNAvgRewardFree 0.8 0.995 (ByStateValuesAndReward 1.0 (ExponentialDecay (Just 0.8) 0.99 100000)) -- ByReward -- (Fixed 30)
        -- AlgBORLVOnly ByStateValues mRefStateAct
        AlgBORL 0.5 0.65 ByStateValues mRefStateAct
        -- (ByStateValuesAndReward 1.0 (ExponentialDecay Nothing 0.5 100000))

allStateInputs :: M.Map [Double] St
allStateInputs = M.fromList $ zip (map netInp [minBound..maxBound]) [minBound..maxBound]

mInverseSt :: NetInputWoAction -> Maybe (Either String St)
mInverseSt xs = return <$> M.lookup xs allStateInputs

usermode :: IO ()
usermode = do
  writeFile queueLenFilePath "Queue Length\n"

  -- rl <- (randomNetworkInitWith UniformInit :: IO NN) >>= \nn -> mkUnichainGrenade alg initState netInp actions actFilter params decay nn nnConfig (Just initVals)
  rl <- do
    case alg of
      AlgBORL{} -> (randomNetworkInitWith UniformInit :: IO NNCombined) >>= \nn -> mkUnichainGrenadeCombinedNet alg initState netInp actions actFilter params decay nn nnConfig (Just initVals)
      AlgDQNAvgRewardFree{} -> (randomNetworkInitWith UniformInit :: IO NNCombinedAvgFree) >>= \nn -> mkUnichainGrenadeCombinedNet alg initState netInp actions actFilter params decay nn nnConfig (Just initVals)
      AlgDQN{} ->  (randomNetworkInitWith UniformInit :: IO NN) >>= \nn -> mkUnichainGrenadeCombinedNet alg initState netInp actions actFilter params decay nn nnConfig (Just initVals)
  -- rl <- (randomNetworkInitWith UniformInit :: IO NN) >>= \nn -> mkUnichainGrenade alg initState netInp actions actFilter params decay nn nnConfig (Just initVals)
  -- rl <- mkUnichainTensorflow alg initState netInp actions actFilter params decay modelBuilder nnConfig  (Just initVals)
  rl <- mkUnichainTensorflowCombinedNet alg initState netInp actions actFilter params decay modelBuilder nnConfig  (Just initVals)
  -- let rl = mkUnichainTabular alg initState tblInp actions actFilter params decay (Just initVals)
  askUser (Just mInverseSt) True usage cmds rl
  where cmds = []
        usage = []

type NN = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 2, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 2, 'D1 2]
type NNCombined = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 40, Relu, FullyConnected 40 30, Relu, FullyConnected 30 12, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 40, 'D1 40, 'D1 30, 'D1 30, 'D1 12, 'D1 12]
type NNCombinedAvgFree = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 4, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 4, 'D1 4]


modelBuilder :: ModelBuilderFunction
modelBuilder colOut =
  buildModel $
  inputLayer1D inpLen >> fullyConnected [20] TF.relu' >> fullyConnected [10] TF.relu' >> fullyConnected [10] TF.relu' >> fullyConnected [genericLength actions, colOut] TF.tanh' >>
  trainingByAdamWith TF.AdamConfig {TF.adamLearningRate = 0.01, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}
  -- trainingByGradientDescent 0.01
  where inpLen = genericLength (netInp initState)

netInp :: St -> [Double]
netInp (St len arr) = [scaleNegPosOne (0, fromIntegral maxQueueSize) $ fromIntegral len, scaleNegPosOne (0, 1) $ fromIntegral $ fromEnum arr]

tblInp :: St -> [Double]
tblInp (St len arr)        = [fromIntegral len, fromIntegral $ fromEnum arr]


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
    appendQueueLenFile f st@(St len _) = do
      appendFile queueLenFilePath (show len ++ "\n")
      f st


actFilter :: St -> [Bool]
actFilter (St _ False)  = [True, False]
actFilter (St len True) | len >= maxQueueSize = [True, False]
actFilter _             = [True, True]

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

reject :: St -> IO (Reward St, St, EpisodeEnd)
reject st@(St len True) = do
  reward <- rewardFunction st Reject
  r <- randomRIO (0, 1 :: Double)
  return $ if r <= lambda / (lambda + mu)
    then (reward, St len True, False)              -- new arrival with probability lambda/(lambda+mu)
    else (reward, St (max 0 (len-1)) False, False) -- no new arrival with probability: mu / (lambda+mu)
reject st@(St len False) = do
  reward <- rewardFunction st Reject
  r <- randomRIO (0, 1 :: Double) -- case for continue (only the reject action is allowed)
  return $ if r <= lambda / (lambda + mu)
    then (reward,St len True, False)              -- new arrival with probability lambda/(lambda+mu)
    else (reward,St (max 0 (len-1)) False, False) -- processing finished with probability: mu / (lambda+mu)


admit :: St -> IO (Reward St, St, EpisodeEnd)
admit st@(St len True) = do
  reward <- rewardFunction st Admit
  r <- randomRIO (0, 1 :: Double)
  return $ if r <= lambda / (lambda + mu)
    then (reward, St (len+1) True, False)  -- admit + new arrival
    else (reward, St len False, False)     -- admit + no new arrival
admit _ = error "admit function called with no arrival available. This function is only to be called when an order arrived."
