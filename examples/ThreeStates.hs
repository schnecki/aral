{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLists       #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
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


import           Control.DeepSeq
import           Control.Lens
import           Data.Default
import           Data.Int             (Int64)
import           Data.List            (genericLength)
import qualified Data.Map.Strict      as M
import           Data.Serialize
import           Data.Text            (Text)
import qualified Data.Vector.Storable as V
import           GHC.Exts             (fromList)
import           GHC.Generics
import           Grenade              hiding (train)
import           Prelude              hiding (Left, Right)

import           ML.ARAL              hiding (actionFilter)
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
    , _trainingIterations = 1
    , _grenadeLearningParams = OptAdam 0.001 0.9 0.999 1e-8 1e-3
    , _grenadeSmoothTargetUpdate = 0.01
    , _grenadeSmoothTargetUpdatePeriod = 1
    , _learningParamsDecay = ExponentialDecay Nothing 0.5 100000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsReward False 2
    , _scaleOutputAlgorithm = ScaleMinMax
    , _cropTrainMaxValScaled = Just 0.98
    , _grenadeDropoutFlipActivePeriod = 0
    , _grenadeDropoutOnlyInactiveAfter = 0
    , _clipGradients = ClipByGlobalNorm 0.01
    }


borlSettings :: Settings
borlSettings = def {_workersMinExploration = [], _nStep = 1}


netInp :: St -> V.Vector Double
netInp st = V.singleton (scaleMinMax (minVal, maxVal) (fromIntegral $ fromEnum st))

tblInp :: St -> V.Vector Double
tblInp st = V.singleton (fromIntegral $ fromEnum st)


maxVal :: Double
maxVal = fromIntegral $ fromEnum (maxBound :: St)

minVal :: Double
minVal = fromIntegral $ fromEnum (minBound :: St)

numActions :: Int64
numActions = genericLength actions

numInputs :: Int64
numInputs = fromIntegral $ V.length $ netInp initState


instance RewardFuture St where
  type StoreType St = ()


instance BorlLp St Act where
  lpActions _ = actions
  lpActionFilter _ = head . actionFilter
  lpActionFunction = actionFun


policy :: Policy St Act
policy s a
  | (s, a) == (A, Left)  = [((B, Right), 1.0)]
  | (s, a) == (B, Right) = [((A, Left), 1.0)]
  | (s, a) == (A, Right) = [((C, Left), 1.0)]
  | (s, a) == (C, Left)  = [((A, Left), 1.0)]
  | otherwise = []

mRefState :: Maybe (St, ActionIndex)
-- mRefState = Just (initState, 0)
mRefState = Nothing

alg :: Algorithm St
alg =
        -- AlgARAL defaultGamma0 defaultGamma1 ByStateValues mRefState
        -- algDQNAvgRewardFree
        AlgARAL 0.8 0.999 ByStateValues
        -- AlgARAL 0.8 0.999 (Fixed 1)
        -- AlgARALVOnly (Fixed 1) Nothing
        -- AlgDQN 0.99 EpsilonSensitive -- need to change epsilon accordingly to not have complete random!!!
        -- AlgDQN 0.99 Exact


main :: IO ()
main = do

  -- runBorlLpInferWithRewardRepetWMax 13 80000 policy mRefState >>= print
  runBorlLp policy mRefState >>= print
  putStr "NOTE: Above you can see the solution generated using linear programming."

  -- rl <- mkUnichainGrenade alg (liftInitSt initState) netInp actionFun actionFilter params decay (modelBuilderGrenade actions initState) nnConfig borlSettings Nothing
  rl <- mkUnichainTabular alg (liftInitSt initState) netInp actionFun actionFilter params decay borlSettings Nothing
  let inverseSt | isAnn rl = mInverseSt
                | otherwise = mInverseStTbl

  askUser (Just inverseSt) True usage cmds qlCmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []
        qlCmds = []


mInverseSt :: NetInputWoAction -> Maybe (Either String St)
mInverseSt xs = return <$> M.lookup xs allStateInputs

mInverseStTbl :: NetInputWoAction -> Maybe (Either String St)
mInverseStTbl xs = return <$> M.lookup xs allStateInputsTbl

allStateInputs :: M.Map NetInputWoAction St
allStateInputs = M.fromList $ zip (map netInp [minBound..maxBound]) [minBound..maxBound]

allStateInputsTbl :: M.Map NetInputWoAction St
allStateInputsTbl = M.fromList $ zip (map tblInp [minBound..maxBound]) [minBound..maxBound]

-- | The definition for a feed forward network using the dynamic module. Note the nested networks. This network clearly is over-engeneered for this example!
modelBuilderGrenade :: [Action a] -> St -> Integer -> IO SpecConcreteNetwork
modelBuilderGrenade acts initSt cols =
  buildModel $
  inputLayer1D lenIn >>
  fullyConnected 6 >> relu >>
  fullyConnected lenOut >> reshape (lenActs, cols, 1) >> tanhLayer
  where
    lenOut = lenActs * cols
    lenIn = fromIntegral $ V.length (netInp initSt)
    lenActs = genericLength acts


initState :: St
initState = A

-- | ARAL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha = 0.07
    , _alphaRhoMin = 2e-5
    , _beta = 0.07
    , _delta = 0.07
    , _gamma = 0.07
    , _epsilon = 1.5

    , _exploration = 1.0
    , _learnRandomAbove = 0.1
    , _zeta = 0.15
    , _xi = 0.001

    }

-- | Decay function of parameters.
decay :: ParameterDecaySetting
decay =
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-1) 0.05 10000
      , _alphaRhoMin      = NoDecay
      , _beta             = ExponentialDecay (Just 0.07) 0.05 100000
      , _delta            = ExponentialDecay (Just 0.07) 0.05 100000
      , _gamma            = ExponentialDecay (Just 0.01) 0.05 10000
      , _zeta             = ExponentialDecay (Just 1e-3) 0.5 150000
      , _xi               = ExponentialDecay (Just 1e-3) 0.5 150000
        -- Exploration
      , _epsilon          = [ExponentialDecay (Just 0.2) 0.05 10000]
      , _exploration      = ExponentialDecay (Just 1e-2) 0.05 100000
      , _learnRandomAbove = NoDecay
      }


-- State
data St
  = B
  | A
  | C
  deriving (Ord, Eq, Show, Enum, Bounded, NFData, Generic, Serialize)

-- Actions
data Act
  = Left
  | Right
  deriving (Eq, Ord, Enum, Bounded, Generic, NFData, Serialize)

instance Show Act where
  show Left  = "left  "
  show Right = "right "

actions :: [Act]
actions = [Left, Right]

actionFun :: AgentType -> St -> [Act] -> IO (Reward St, St, EpisodeEnd)
actionFun tp s [Left]  = moveLeft tp s
actionFun tp s [Right] = moveRight tp s
actionFun _ _ xs       = error $ "Multiple actions received in actionFun: " ++ show xs

actionFilter :: St -> [V.Vector Bool]
actionFilter A = [V.fromList [True, True]]
actionFilter B = [V.fromList [False, True]]
actionFilter C = [V.fromList [True, False]]


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
