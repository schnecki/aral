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

import           ML.BORL              hiding (actionFilter)

import           Helper

import           Control.DeepSeq      (NFData)
import           Control.Lens
import           Data.Default
import           Data.Int             (Int64)
import           Data.Text            (Text)
import qualified Data.Vector.Storable as V
import           GHC.Exts             (fromList)
import           GHC.Generics
import           Grenade              hiding (train)
import           Prelude              hiding (Left, Right)

import qualified HighLevelTensorflow  as TF


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
    , _learningParamsDecay = ExponentialDecay Nothing 0.5 100000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsReward False 20
    , _stabilizationAdditionalRho = 0.025
    , _stabilizationAdditionalRhoDecay = ExponentialDecay Nothing 0.05 100000
    , _updateTargetInterval = 100
    , _updateTargetIntervalDecay = NoDecay
    }

borlSettings :: Settings
borlSettings = def {_workersMinExploration = [], _nStep = 1}


netInp :: St -> V.Vector Float
netInp st = V.singleton (scaleNegPosOne (minVal,maxVal) (fromIntegral $ fromEnum st))

tblInp :: St -> V.Vector Float
tblInp st = V.singleton (fromIntegral $ fromEnum st)


maxVal :: Float
maxVal = fromIntegral $ fromEnum (maxBound :: St)

minVal :: Float
minVal = fromIntegral $ fromEnum (minBound :: St)

numActions :: Int64
numActions = fromIntegral $ length actions

numInputs :: Int64
numInputs = fromIntegral $ V.length (netInp initState)

modelBuilder :: (TF.MonadBuild m) => m TF.TensorflowModel
modelBuilder =
  TF.buildModel $
  TF.inputLayer1D numInputs >>
  TF.fullyConnected [20] TF.relu' >>
  TF.fullyConnected [10] TF.relu' >>
  TF.fullyConnected [numActions] TF.tanh' >>
  TF.trainingByAdamWith TF.AdamConfig {TF.adamLearningRate = 0.001, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}


modelBuilderGrenade :: Integer -> IO SpecConcreteNetwork
modelBuilderGrenade  cols =
  buildModel $
  inputLayer1D (fromIntegral numInputs) >>
  fullyConnected 20 >> relu >> dropout 0.90 >>
  fullyConnected 10 >> relu >>
  fullyConnected 10 >> relu >>
  fullyConnected (cols * fromIntegral numActions) >> reshape (fromIntegral numActions, cols, 1) >> tanhLayer


instance RewardFuture St where
  type StoreType St = ()

alg :: Algorithm St
alg =
        -- AlgDQN 0.99 EpsilonSensitive
        -- AlgDQN 0.5  Exact
        -- AlgDQN 0.8027  Exact
        -- AlgDQN 0.50  EpsilonSensitive
        AlgDQNAvgRewAdjusted 0.8 0.99 (Fixed 2.5)
        -- ByStateValues -- (ByStateValuesAndReward 0.5) -- ByReward -- (Fixed 30)
        -- AlgBORLVOnly ByStateValues mRefStateAct
        -- AlgBORL 0.5 0.8 ByStateValues mRefStateAct

        -- AlgBORL 0.5 0.8 ByStateValues  (ByStateValuesAndReward 1.0 (ExponentialDecay Nothing 0.5 100000))

mRefStateAct :: Maybe (St, ActionIndex)
-- mRefStateAct = Just (initState, fst $ head $ zip [0..] (actionFilter initState))
mRefStateAct = Nothing


main :: IO ()
main = do
  -- createModel >>= mapM_ testRun

  nn <- randomNetworkInitWith HeEtAl :: IO NN
  -- rl <- mkUnichainGrenade alg (liftInitSt initState) actions actionFilter params decay modelBuilderGrenade nnConfig borlSettings
  -- rl <- mkUnichainTensorflow alg (liftInitSt initState) actions actionFilter params decay modelBuilder nnConfig borlSettings Nothing
  rl <- mkUnichainTabular alg (liftInitSt initState) tblInp actions actionFilter params decay borlSettings Nothing
  askUser Nothing True usage cmds [] rl   -- maybe increase learning by setting estimate of rho

  where cmds = zipWith3 (\n (s,a) na -> (s, (n, Action a na))) [0..]
          [("s", moveLeft), ("f", moveRight)] ["moveLeft", "moveRight"]
        usage = [("s", "moveLeft"), ("f", "moveRight")]

-- policy :: Policy St
-- policy s a
--   | s == Left 5  = [((One, right), 1.0)]
--   | s == Right 10 = [((One, left), 1.0)]
--   | s == Left x = [((Left (x+1), moveLeft), 1.0)]
--   | s == Right x = [((Right (x+1), moveRight), 1.0)]
--   | otherwise = []


initState :: St
initState = One

-- | BORL Parameters.
params :: ParameterInitValues
params = Parameters
  { _alpha            = 0.01
  , _alphaRhoMin = 2e-5
  , _beta             = 0.01
  , _delta            = 0.005
  , _gamma            = 0.01
  , _epsilon          = 1.0

  , _exploration      = 1.0
  , _learnRandomAbove = 0.0
  , _zeta             = 0.0
  , _xi               = 0.0075

  }


-- | Decay function of parameters.
decay :: ParameterDecaySetting
decay =
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-6) 0.25 100000
      , _alphaRhoMin      = NoDecay
      , _beta             = ExponentialDecay (Just 5e-3) 0.05 100000
      , _delta            = ExponentialDecay (Just 5e-3) 0.05 100000
      , _gamma            = ExponentialDecay (Just 0.03) 0.05 100000
      , _zeta             = ExponentialDecay (Just 1e-3) 0.5 150000
      , _xi               = ExponentialDecay (Just 1e-3) 0.5 150000
        -- Exploration
      , _epsilon          = [NoDecay]
      , _exploration      = ExponentialDecay (Just 10e-2) 0.01 10000
      , _learnRandomAbove = NoDecay
      }


-- State
data St
  = One
  | Left Int
  | Right Int
  deriving (Ord, Eq, Show, NFData, Generic)

instance Enum St where
  toEnum 1 = One
  toEnum x | x <= 5 = Left x
  toEnum x = Right (x - 4)
  fromEnum One       = 1
  fromEnum (Left x)  = x
  fromEnum (Right x) = x + 4

instance Bounded St where
  minBound = One
  maxBound = Right 10


-- Actions
actions :: [Action St]
actions =
  [ Action moveLeft "left "
  , Action moveRight "right"]

actionFilter :: St -> V.Vector Bool
actionFilter One     = V.fromList [True, True]
actionFilter Left{}  = V.fromList [True, False]
actionFilter Right{} = V.fromList [False, True]


moveLeft :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveLeft tp s =
  case s of
    One    -> return (0, Left 2, False)
    Left 5 -> return (5, One, False)
    Left x -> return (0, Left (x+1), False)
    _      -> moveRight tp s

moveRight :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveRight tp s =
  case s of
    One      -> return (0, Right 2, False)
    Right 10 -> return (20, One, False)
    Right x  -> return (0, Right (x + 1), False)
    _        -> moveLeft tp s


