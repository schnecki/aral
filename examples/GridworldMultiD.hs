-- This is a xD world, where each dimension is steered by one agent. All of them can go left, right, up down. The goal is at (0,0,...)
--
--

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

import           ML.ARAL              as B

import           Experimenter

import           Helper

import           Control.DeepSeq      (NFData)
import           Control.Lens         ((^.))

import           Control.Monad        (when)
import           Data.Default
import           Data.List            (elemIndex, genericLength)
import qualified Data.Map.Strict      as M
import           Data.Serialize
import qualified Data.Vector.Storable as V
import           GHC.Generics
import           Grenade
import           Prelude              hiding (Left, Right)
import           System.Random

import           Debug.Trace

dim, cubeSize, goal :: Int
dim = 3               -- dimensions, this is also the number of independentAgents. Each agent steers one dimension
cubeSize = 5          -- size of each dim
goal = 0              -- goal is at [goal, goal, ...]

goalSt :: St
goalSt = St $ replicate dim goal

expSetup :: ARAL St Act -> ExperimentSetting
expSetup borl =
  ExperimentSetting
    { _experimentBaseName         = "gridworld-multi 28.1."
    , _experimentInfoParameters   = [isNN]
    , _experimentRepetitions      = 40
    , _preparationSteps           = 500000
    , _evaluationWarmUpSteps      = 0
    , _evaluationSteps            = 10000
    , _evaluationReplications     = 1
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


nnConfig :: NNConfig
nnConfig =
  NNConfig
    { _replayMemoryMaxSize = 10000
    , _replayMemoryStrategy = ReplayMemoryPerAction -- ReplayMemorySingle
    , _trainBatchSize = 4
    , _trainingIterations = 3
    , _grenadeLearningParams = OptAdam 0.001 0.9 0.999 1e-8 1e-3
    , _grenadeSmoothTargetUpdate = 0.01
    , _grenadeSmoothTargetUpdatePeriod = 100
    , _learningParamsDecay = NoDecay -- ExponentialDecay (Just 1e-6) 0.75 10000
    , _prettyPrintElems = take 250 $ map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsRewardAlg alg False 10
    , _scaleOutputAlgorithm = ScaleMinMax
    , _cropTrainMaxValScaled = Just 0.98
    , _grenadeDropoutFlipActivePeriod = 10000
    , _grenadeDropoutOnlyInactiveAfter = 10^5
    , _clipGradients = NoClipping -- ClipByGlobalNorm 0.01
    }

borlSettings :: Settings
borlSettings = def
  { _workersMinExploration = replicate 7 0.01 -- ++ [0.1, 0.2, 0.3]
  , _nStep = 4
  , _independentAgents = dim
  , _overEstimateRho = False -- True
  , _independentAgentsSharedRho = True
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
    , _epsilon             = 0.25

    , _exploration         = 1.0
    , _learnRandomAbove    = 0.7
    , _zeta                = 0.03
    , _xi                  = 0.005

    }


-- | Decay function of parameters.
decay :: ParameterDecaySetting
decay =
    Parameters
      { _alpha            = ExponentialDecay (Just 5e-5) 0.5 25000  -- 5e-4
      , _alphaRhoMin      = NoDecay
      , _beta             = ExponentialDecay (Just 1e-4) 0.5 50000
      , _delta            = ExponentialDecay (Just 5e-4) 0.5 50000
      , _gamma            = ExponentialDecay (Just 1e-3) 0.5 50000
      , _zeta             = ExponentialDecay (Just 0) 0.5 150000
      , _xi               = NoDecay
      -- Exploration
      , _epsilon          = [NoDecay] -- [ExponentialDecay (Just 0.050) 0.05 150000]
      , _exploration      = ExponentialDecay (Just 0.01) 0.50 25000
      , _learnRandomAbove = NoDecay
      }

-- -- | Decay function of parameters.
-- decay :: ParameterDecaySetting
-- decay =
--     Parameters
--       { _alpha            = ExponentialDecay (Just 1e-5) 0.5 50000  -- 5e-4
--       , _alphaRhoMin      = NoDecay
--       , _beta             = ExponentialDecay (Just 1e-4) 0.5 150000
--       , _delta            = ExponentialDecay (Just 5e-4) 0.5 150000
--       , _gamma            = ExponentialDecay (Just 1e-3) 0.5 150000 -- 1e-3
--       , _zeta             = ExponentialDecay (Just 0) 0.5 150000
--       , _xi               = NoDecay
--       -- Exploration
--       , _epsilon          = [NoDecay] -- ExponentialDecay (Just 5.0) 0.5 150000
--       , _exploration      = ExponentialDecay (Just 0.01) 0.50 100000
--       , _learnRandomAbove = NoDecay
--       }

initVals :: InitValues
initVals = InitValues 0 0 0 0 0 0

main :: IO ()
main = usermode


mRefState :: Maybe (St, ActionIndex)
mRefState = Nothing
-- mRefState = Just (fromIdx (goalX, goalY), 0)

alg :: Algorithm St
alg =

  -- AlgARALVOnly ByStateValues Nothing
        -- AlgDQN 0.99  EpsilonSensitive
        -- AlgDQN 0.50  EpsilonSensitive            -- does work
        -- algDQNAvgRewardFree
        AlgDQNAvgRewAdjusted 0.8 1.0 ByStateValues
  -- AlgARAL 0.5 0.8 ByStateValues mRefState

usermode :: IO ()
usermode = do

  -- Approximate all fucntions using a single neural network
  rl <- mkUnichainGrenadeCombinedNet alg (liftInitSt initState) netInp actionFun actFilter params decay modelBuilderGrenade nnConfig borlSettings (Just initVals)
  -- rl <- mkUnichainGrenade alg (liftInitSt initState) netInp actionFun actFilter params decay modelBuilderGrenade nnConfig borlSettings (Just initVals)

  -- Use a table to approximate the function (tabular version)
  -- rl <- mkUnichainTabular alg (liftInitSt initState) tblInp actionFun actFilter params decay borlSettings (Just initVals)
  let inverseSt | isAnn rl = Just mInverseSt
                | otherwise = Nothing

  askUser inverseSt True usage cmds [] rl -- maybe increase learning by setting estimate of rho
  where
    cmds | dim == 1 = map (\(s, a) -> (fst s, maybe [0] return (elemIndex a actions))) (zip usage [Inc, Dec])
         | otherwise = []
    usage | dim == 1 = [("i", "Move up"), ("k", "Move down")]
          | otherwise = []


-- | The definition for a feed forward network using the dynamic module. Note the nested networks. This network clearly is over-engeneered for this example!
modelBuilderGrenade :: Integer -> (Integer, Integer) -> IO SpecConcreteNetwork
modelBuilderGrenade lenIn (lenActs, cols) =
  buildModelWith (def { cpuBackend = BLAS }) def $
  inputLayer1D lenIn >>
  fullyConnected (3*lenIn) >> relu >>
  fullyConnected (1*lenIn) >> relu >>
  fullyConnected (lenIn `div` 2) >> relu >>

  -- fullyConnected 36 >> relu >> -- dropout 0.90 >>
  -- fullyConnected 24 >> relu >>
  -- -- fullyConnected 24 >> relu >>
  fullyConnected lenOut >> reshape (lenActs, cols, 1) >> tanhLayer
  where
    lenOut = lenActs * cols


netInp :: St -> V.Vector Double
netInp (St st) =
  V.fromList $ map (scaleMinMax (0, fromIntegral cubeSize) . fromIntegral) st

tblInp :: St -> V.Vector Double
tblInp (St st) = V.fromList (map fromIntegral st)

initState :: St
initState = St $ replicate dim (cubeSize `div` 2)

-- State
newtype St =
  St [Int] -- ^ Number in cubeSize for each dimension.
  deriving (Eq, Ord, Show, NFData, Generic, Serialize)

instance Enum St where
  fromEnum (St st) = Prelude.sum $ zipWith (\i v -> v * cubeSize ^ i) [0 .. (dim - 1)] (reverse st)
  toEnum nr = St $ fromEnum' (dim - 1) nr
    where
      fromEnum' 0 n = [n]
      fromEnum' i n = n `div` cubeSize ^ i : fromEnum' (i - 1) (n `mod` cubeSize ^ i)

instance Bounded St where
  minBound = St $ replicate dim 0
  maxBound = St $ replicate dim (cubeSize-1)


-- Actions
data Act = NoOp | Random | Inc | Dec
  deriving (Eq, Ord, Enum, Bounded, Generic, NFData, Serialize)

instance Show Act where
  show Random = "random"
  show NoOp   = "noop  "
  show Inc    = "inc.  "
  show Dec    = "dec.  "

actions :: [Act]
actions = [minBound..maxBound]

actionFun :: AgentType -> St -> [Act] -> IO (Reward St, St, EpisodeEnd)
actionFun tp s@(St st) acts
  | all (== Random) acts = moveRand tp s
  | otherwise = do
    stepRew <- randomRIO (0, 8 :: Double)
    return (Reward $ stepRew + (Prelude.sum rews / fromIntegral (length rews)), St s', False)
  where
    (rews, s') = unzip $ zipWith moveX acts [0 ..]
    moveX :: Act -> Int -> (Double, Int)
    moveX NoOp   = \nr -> (0, st !! nr)
    moveX Inc    = moveInc tp s
    moveX Dec    = moveDec tp s
    moveX Random = error "Unexpected Random in actionFun.moveX. Check the actionFilter!"


actFilter :: St -> [V.Vector Bool]
actFilter st
  | st == goalSt = replicate (borlSettings ^. independentAgents) (False `V.cons` ( True `V.cons` V.replicate (length actions - 2) False))
actFilter _ = replicate (borlSettings ^. independentAgents) (True `V.cons` (False `V.cons` V.replicate (length actions - 2) True))
actFilter _ = error "Unexpected setup in actFilter in GridworldMini.hs"


moveRand :: AgentType -> St -> IO (Reward St, St, EpisodeEnd)
moveRand tp st = do
  when (st /= goalSt) $ error $ "moveRand in non-goal state: " ++ show st
  xs <- sequenceA (replicate dim $ randomRIO (0, cubeSize :: Int))
  return (Reward 10, St xs, True)

moveDec :: AgentType -> St -> Int -> (Double, Int)
moveDec _ (St st) agNr
  | m == 0 = (-1, 0)
  | otherwise = (0, m - 1)
  where
    m = st !! agNr

moveInc :: AgentType -> St -> Int -> (Double, Int)
moveInc _ (St st) agNr
  | m == cubeSize = (-1, cubeSize)
  | otherwise = (0, m + 1)
  where
    m = st !! agNr


allStateInputs :: M.Map NetInputWoAction St
allStateInputs = M.fromList $ zip (map netInp [minBound..maxBound]) [minBound..maxBound]

mInverseSt :: NetInputWoAction -> Maybe (Either String St)
mInverseSt xs = return <$> M.lookup xs allStateInputs
