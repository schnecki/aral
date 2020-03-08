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

import           ML.BORL                hiding (actionFilter)

import           Helper

import           Prelude                hiding (Left, Right)

import           Control.DeepSeq        (NFData)
import           Control.Lens
import           Data.Int               (Int64)
import           Data.List              (genericLength)
import           Data.Text              (Text)
import qualified Data.Vector            as V
import           GHC.Exts               (fromList)
import           GHC.Generics
import           Grenade                hiding (train)


import qualified TensorFlow.Build       as TF (addNewOp, evalBuildT, explicitName, opDef,
                                               opDefWithName, opType, runBuildT, summaries)
import qualified TensorFlow.Core        as TF hiding (value)
import qualified TensorFlow.GenOps.Core as TF (abs, add, approximateEqual,
                                               approximateEqual, assign, cast,
                                               getSessionHandle, getSessionTensor,
                                               identity', lessEqual, matMul, mul,
                                               readerSerializeState, relu', shape, square,
                                               sub, tanh, tanh', truncatedNormal)
import qualified TensorFlow.Minimize    as TF
import qualified TensorFlow.Ops         as TF (initializedVariable, initializedVariable',
                                               placeholder, placeholder', reduceMean,
                                               reduceSum, restore, save, scalar, vector,
                                               zeroInitializedVariable,
                                               zeroInitializedVariable')
import qualified TensorFlow.Tensor      as TF (Ref (..), collectAllSummaries,
                                               tensorNodeName, tensorRefFromName,
                                               tensorValueFromName)


type NN = Network '[ FullyConnected 1 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 2, Tanh] '[ 'D1 1, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 2, 'D1 2]

nnConfig :: NNConfig
nnConfig =
  NNConfig
    { _replayMemoryMaxSize = 10000
    , _trainBatchSize = 32
    , _grenadeLearningParams = LearningParameters 0.01 0.9 0.0001
    , _learningParamsDecay = ExponentialDecay Nothing 0.5 100000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsReward False 20
    , _stabilizationAdditionalRho = 0.025
    , _stabilizationAdditionalRhoDecay = ExponentialDecay Nothing 0.05 100000
    , _updateTargetInterval = 100
    , _updateTargetIntervalDecay = NoDecay
    , _trainMSEMax = Just 0.015
    , _setExpSmoothParamsTo1 = True
    }


netInp :: St -> [Double]
netInp st = [scaleNegPosOne (minVal,maxVal) (fromIntegral $ fromEnum st)]

tblInp :: St -> [Double]
tblInp st = [fromIntegral $ fromEnum st]


maxVal :: Double
maxVal = fromIntegral $ fromEnum (maxBound :: St)

minVal :: Double
minVal = fromIntegral $ fromEnum (minBound :: St)

numActions :: Int64
numActions = genericLength actions

numInputs :: Int64
numInputs = genericLength (netInp initState)

modelBuilder :: (TF.MonadBuild m) => m TensorflowModel
modelBuilder =
  buildModel $
  inputLayer1D numInputs >> fullyConnected [20] TF.relu' >> fullyConnected [10] TF.relu' >> fullyConnected [numActions] TF.tanh' >>
  trainingByAdamWith TF.AdamConfig {TF.adamLearningRate = 0.001, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}


instance RewardFuture St where
  type StoreType St = ()

alg :: Algorithm St
alg =
        -- AlgDQN 0.99 EpsilonSensitive
        -- AlgDQN 0.5  Exact
        -- AlgDQN 0.8027  Exact
        -- AlgDQN 0.50  EpsilonSensitive
        AlgDQNAvgRewAdjusted Nothing 0.8 0.99 (Fixed 2.5)
        -- ByStateValues -- (ByStateValuesAndReward 0.5) -- ByReward -- (Fixed 30)
        -- AlgBORLVOnly ByStateValues mRefStateAct
        -- AlgBORL 0.5 0.8 ByStateValues mRefStateAct

        -- AlgBORL 0.5 0.8 ByStateValues  (ByStateValuesAndReward 1.0 (ExponentialDecay Nothing 0.5 100000))

mRefStateAct :: Maybe (St, ActionIndex)
mRefStateAct = Just (initState, fst $ head $ zip [0..] (actionFilter initState))
-- mRefStateAct = Nothing


main :: IO ()
main = do
  -- createModel >>= mapM_ testRun

  nn <- randomNetworkInitWith HeEtAl :: IO NN
  -- rl <- mkUnichainGrenade alg initState actions actionFilter params decay nn nnConfig
  -- rl <- mkUnichainTensorflow alg initState actions actionFilter params decay modelBuilder nnConfig Nothing
  let rl = mkUnichainTabular alg initState tblInp actions actionFilter params decay Nothing
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
  , _alphaANN = 1
  , _beta             = 0.01
  , _betaANN = 1
  , _delta            = 0.005
  , _deltaANN = 1
  , _gamma            = 0.01
  , _gammaANN = 1
  , _epsilon          = 1.0
  , _explorationStrategy = EpsilonGreedy
  , _exploration      = 1.0
  , _learnRandomAbove = 0.0
  , _zeta             = 0.0
  , _xi               = 0.0075
  , _disableAllLearning = False
  }


-- | Decay function of parameters.
decay :: Decay
decay =
  decaySetupParameters
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-6) 0.25 100000
      , _beta             = ExponentialDecay (Just 5e-3) 0.05 100000
      , _delta            = ExponentialDecay (Just 5e-3) 0.05 100000
      , _gamma            = ExponentialDecay (Just 0.03) 0.05 100000
      , _zeta             = ExponentialDecay (Just 1e-3) 0.5 150000
      , _xi               = ExponentialDecay (Just 1e-3) 0.5 150000
        -- Exploration
      , _epsilon          = NoDecay
      , _exploration      = ExponentialDecay (Just 10e-2) 0.01 10000
      , _learnRandomAbove = NoDecay
      -- ANN
      , _alphaANN         = ExponentialDecay (Just 0.3) 0.75 150000
      , _betaANN          = ExponentialDecay (Just 0.3) 0.75 150000
      , _deltaANN         = ExponentialDecay (Just 0.3) 0.75 150000
      , _gammaANN         = ExponentialDecay (Just 0.3) 0.75 150000
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


type R = Double
type P = Double

-- Actions
actions :: [Action St]
actions =
  [ Action moveLeft "left "
  , Action moveRight "right"]

actionFilter :: St -> [Bool]
actionFilter One     = [True, True]
actionFilter Left{}  = [True, False]
actionFilter Right{} = [False, True]


moveLeft :: St -> IO (Reward St,St, EpisodeEnd)
moveLeft s =
  case s of
    One    -> return (0, Left 2, False)
    Left 5 -> return (5, One, False)
    Left x -> return (0, Left (x+1), False)
    _      -> moveRight s

moveRight :: St -> IO (Reward St,St, EpisodeEnd)
moveRight s =
  case s of
    One      -> return (0, Right 2, False)
    Right 10 -> return (20, One, False)
    Right x  -> return (0, Right (x + 1), False)
    _        -> moveLeft s


encodeImageBatch :: TF.TensorDataType V.Vector a => [[a]] -> TF.TensorData a
encodeImageBatch xs = TF.encodeTensorData [genericLength xs, 2] (V.fromList $ mconcat xs)
-- encodeLabelBatch xs = TF.encodeTensorData [genericLength xs] (V.fromList xs)

setCheckFile :: FilePath -> TensorflowModel' -> TensorflowModel'
setCheckFile tempDir model = model { checkpointBaseFileName = Just tempDir }

prependName :: Text -> TensorflowModel' -> TensorflowModel'
prependName txt model = model { tensorflowModel = (tensorflowModel model)
        { inputLayerName = txt <> "/" <> (inputLayerName $ tensorflowModel model)
        , outputLayerName = txt <> "/" <> (outputLayerName $ tensorflowModel model)
        , labelLayerName = txt <> "/" <> (labelLayerName $ tensorflowModel model)
        }}

