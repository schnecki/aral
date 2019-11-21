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
import           SolveLp

import           Helper

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
nnConfig = NNConfig
  { _replayMemoryMaxSize  = 10000
  , _trainBatchSize       = 32
  , _grenadeLearningParams       = LearningParameters 0.005 0.0 0.0000
  , _prettyPrintElems     = map netInp ([minBound .. maxBound] :: [St])
  , _scaleParameters      = scalingByMaxAbsReward False 2
  , _updateTargetInterval = 1000
  , _trainMSEMax          = Just 0.03
  }


netInp :: St -> [Double]
netInp st = [scaleNegPosOne (minVal,maxVal) (fromIntegral $ fromEnum st)]

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
mRefState = Nothing
-- mRefState = Just (initState, 0)

main :: IO ()
main = do
  let algorithm =
        AlgBORL defaultGamma0 defaultGamma1 ByStateValues False mRefState
        -- algDQNAvgRewardFree
        -- AlgDQNAvgRewardFree 0.8 0.995 ByStateValues
        -- AlgBORLVOnly (Fixed 1) Nothing
        -- AlgDQN 0.99


  runBorlLp policy mRefState >>= print
  putStr "NOTE: Above you can see the solution generated using linear programming."

  nn <- randomNetworkInitWith HeEtAl :: IO NN

  -- rl <- mkUnichainGrenade algorithm initState netInp actions actionFilter params decay nn nnConfig Nothing
  -- rl <- mkUnichainTensorflow algorithm initState netInp actions actionFilter params decay modelBuilder nnConfig Nothing
  let rl = mkUnichainTabular algorithm initState (return . fromIntegral . fromEnum) actions actionFilter params decay Nothing
  askUser Nothing True usage cmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []


initState :: St
initState = A


-- | BORL Parameters.
params :: Parameters
params =
  Parameters
    { _alpha = 0.005
    , _alphaANN = 1
    , _beta = 0.07
    , _betaANN = 1
    , _delta = 0.07
    , _deltaANN = 1
    , _gamma = 0.07
    , _gammaANN = 1
    , _epsilon = 1.5
    , _exploration = 1.0
    , _learnRandomAbove = 0.1
    , _zeta = 0.1
    , _xi = 0.5
    , _disableAllLearning = False
    }

decay :: Decay
decay = exponentialDecayParameters (Just minValues) 0.05 100000
  where
    minValues =
      Parameters
        { _alpha = 0.005
        , _alphaANN = 1
        , _beta = 0.07
        , _betaANN = 1
        , _delta = 0.07
        , _deltaANN = 1
        , _gamma = 0.07
        , _gammaANN = 1
        , _epsilon = 0.05
        , _exploration = 0.01
        , _learnRandomAbove = 0.1
        , _zeta = 0.15
        , _xi = 0.2
        , _disableAllLearning = False
        }


-- State
data St = B | A | C
  deriving (Ord, Eq, Show, Enum, Bounded,NFData,Generic)
type R = Double
type P = Double

-- Actions
actions :: [Action St]
actions = [left, right]

left,right :: Action St
left = Action moveLeft "left "
right = Action moveRight "right"

actionFilter :: St -> [Bool]
actionFilter A = [True, True]
actionFilter B = [False, True]
actionFilter C = [True, False]


moveLeft :: St -> IO (Reward St,St, EpisodeEnd)
moveLeft s =
  return $
  case s of
    A -> (Reward 2, B, False)
    B -> error "not allowed"
    C -> (Reward 2, A, False)

moveRight :: St -> IO (Reward St,St, EpisodeEnd)
moveRight s =
  return $
  case s of
    A -> (Reward 0, C, False)
    B -> (Reward 0, A, False)
    C -> error "not allowed"


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

