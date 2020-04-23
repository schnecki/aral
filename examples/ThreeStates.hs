{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
-- This is example is a three-state MDP from Mahedevan 1996, Average Reward Reinforcement Learning - Foundations...
-- (Figure 2, p.166).

-- The provided solution is that a) the average reward rho=1 and b) the bias values are

-- when selection action a1 (A->B)
-- V(B) = -0.5
-- V(A) = 0.5
-- V(C) = 1.5

-- when selecting action a2 (A->C)
-- V(B) = -1.5
-- V(A) = -0.5
-- V(C) = 0.5

-- Thus the policy selecting a1 (going Left) is preferable.

module Main where


import           Control.DeepSeq      (NFData)
import           Control.Lens
import           Data.Default
import           Data.Int             (Int64)
import           Data.List            (genericLength)
import           Data.Text            (Text)
import qualified Data.Vector.Storable as V
import           GHC.Exts             (fromList)
import           GHC.Generics
import           Grenade              hiding (train)

import qualified HighLevelTensorflow  as TF


import           ML.BORL              hiding (actionFilter)
import           SolveLp

import           Helper

import           Debug.Trace


type NN = Network '[ FullyConnected 1 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 2, Tanh] '[ 'D1 1, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 2, 'D1 2]

nnConfig :: NNConfig
nnConfig =
  NNConfig
    { _replayMemoryMaxSize = 10000
    , _replayMemoryStrategy = ReplayMemorySingle
    , _trainBatchSize = 32
    , _grenadeLearningParams = OptAdam 0.001 0.9 0.999 1e-8 -- OptSGD 0.01 0.9 0.0001
    , _learningParamsDecay = ExponentialDecay Nothing 0.5 100000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsReward False 2
    , _stabilizationAdditionalRho = 0.25
    , _stabilizationAdditionalRhoDecay = ExponentialDecay Nothing 0.05 100000
    , _updateTargetInterval = 1
    , _updateTargetIntervalDecay = NoDecay
    }


borlSettings :: Settings
borlSettings = def {_workersMinExploration = [], _nStep = 1}


netInp :: St -> V.Vector Float
netInp st = V.singleton (scaleNegPosOne (minVal, maxVal) (fromIntegral $ fromEnum st))

maxVal :: Float
maxVal = fromIntegral $ fromEnum (maxBound :: St)

minVal :: Float
minVal = fromIntegral $ fromEnum (minBound :: St)

numActions :: Int64
numActions = genericLength actions

numInputs :: Int64
numInputs = fromIntegral $ V.length $ netInp initState

modelBuilder :: TF.ModelBuilderFunction
modelBuilder cols =
  TF.buildModel $
  TF.inputLayer1D numInputs >>
  TF.fullyConnected [20] TF.relu' >>
  TF.fullyConnected [10] TF.relu' >>
  TF.fullyConnected [numActions, cols] TF.tanh' >>
  TF.trainingByAdamWith TF.AdamConfig {TF.adamLearningRate = 0.001, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}

instance RewardFuture St where
  type StoreType St = ()


instance BorlLp St where
  lpActions = actions
  lpActionFilter = actionFilter


policy :: Policy St
policy s a
  | (s, a) == (A, left)  = [((B, right), 1.0)]
  | (s, a) == (B, right) = [((A, left), 1.0)]
  | (s, a) == (A, right) = [((C, left), 1.0)]
  | (s, a) == (C, left)  = [((A, left), 1.0)]
  | otherwise = []

mRefState :: Maybe (St, ActionIndex)
-- mRefState = Just (initState, 0)
mRefState = Nothing

alg :: Algorithm St
alg =
        AlgBORL defaultGamma0 defaultGamma1 ByStateValues mRefState
        -- algDQNAvgRewardFree
        -- AlgDQNAvgRewAdjusted Nothing 0.8 0.999 ByStateValues
        -- AlgBORLVOnly (Fixed 1) Nothing
        -- AlgDQN 0.99 EpsilonSensitive -- need to change epsilon accordingly to not have complete random!!!
        -- AlgDQN 0.99 Exact

main :: IO ()
main = do


  runBorlLp policy mRefState >>= print
  putStr "NOTE: Above you can see the solution generated using linear programming."

  nn <- randomNetworkInitWith HeEtAl :: IO NN

  -- rl <- mkUnichainGrenade alg (liftInitSt initState) netInp actions actionFilter params decay nn nnConfig borlSettings Nothing
  rl <- mkUnichainTensorflowCombinedNet alg (liftInitSt initState) netInp actions actionFilter params decay modelBuilder nnConfig borlSettings Nothing
  -- let rl = mkUnichainTabular alg (liftInitSt initState) (return . fromIntegral . fromEnum) actions actionFilter params decay settings Nothing
  askUser Nothing True usage cmds qlCmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []
        qlCmds = []


initState :: St
initState = A


-- | BORL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha = 0.005
    , _alphaRhoMin = 2e-5
    , _beta = 0.01
    , _delta = 0.01
    , _gamma = 0.01
    , _epsilon = 0.1

    , _exploration = 1.0
    , _learnRandomAbove = 0.5
    , _zeta = 0.15
    , _xi = 0.001

    }

-- | Decay function of parameters.
decay :: ParameterDecaySetting
decay =
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-3) 0.05 100000
      , _alphaRhoMin         = NoDecay
      , _beta             = ExponentialDecay (Just 1e-3) 0.05 100000
      , _delta            = ExponentialDecay (Just 1e-3) 0.05 100000
      , _gamma            = ExponentialDecay (Just 1e-2) 0.05 100000
      , _zeta             = ExponentialDecay (Just 1e-3) 0.5 150000
      , _xi               = ExponentialDecay (Just 1e-3) 0.5 150000
        -- Exploration
      , _epsilon          = [NoDecay]
      , _exploration      = ExponentialDecay (Just 10e-2) 0.01 10000
      , _learnRandomAbove = NoDecay
      }


-- State
data St = B | A | C
  deriving (Ord, Eq, Show, Enum, Bounded,NFData,Generic)

-- Actions
actions :: [Action St]
actions = [left, right]

left,right :: Action St
left = Action moveLeft "left "
right = Action moveRight "right"

actionFilter :: St -> V.Vector Bool
actionFilter A = V.fromList [True, True]
actionFilter B = V.fromList [False, True]
actionFilter C = V.fromList [True, False]


moveLeft :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveLeft _ s =
  return $
  case s of
    A -> (Reward 2, B, False)
    B -> error "not allowed"
    C -> (Reward 2, A, False)

moveRight :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveRight _ s =
  return $
  case s of
    A -> (Reward 0, C, False)
    B -> (Reward 0, A, False)
    C -> error "not allowed"


