{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
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

nnConfig :: NNConfig St
nnConfig = NNConfig
  { _toNetInp              = netInp
  , _replayMemoryMaxSize   = 10000
  , _trainBatchSize        = 32
  , _grenadeLearningParams = LearningParameters 0.005 0.0 0.0000
  , _prettyPrintElems      = [minBound .. maxBound] :: [St]
  , _scaleParameters       = scalingByMaxAbsReward False 20
  , _updateTargetInterval  = 100
  , _trainMSEMax           = Just 0.015
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
  inputLayer1D numInputs >> fullyConnected1D 20 TF.relu' >> fullyConnected1D 10 TF.relu' >> fullyConnected1D numActions TF.tanh' >>
  trainingByAdam1DWith TF.AdamConfig {TF.adamLearningRate = 0.001, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}

main :: IO ()
main = do
  -- createModel >>= mapM_ testRun

  nn <- randomNetworkInitWith HeEtAl :: IO NN
  let algorithm =
        -- AlgDQNAvgRew 0.99 (ByMovAvg 200)
        AlgBORL defaultGamma0 defaultGamma1 (ByMovAvg 200) Normal False

  -- rl <- mkUnichainGrenade algBORL initState actions actionFilter params decay nn nnConfig
  -- rl <- mkUnichainTensorflow (AlgBORL defaultGamma0 defaultGamma0 ByStateValues Normal) initState actions actionFilter params decay modelBuilder nnConfig Nothing
  let rl = mkUnichainTabular algorithm
           initState id actions actionFilter params decay Nothing
  askUser True usage cmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = zipWith3 (\n (s,a) na -> (s, (n, Action a na))) [0..]
          [("s", moveLeft), ("f", moveRight)] ["moveLeft", "moveRight"]
        usage = [("s", "moveLeft"), ("f", "moveRight")]


initState :: St
initState = One


-- | BORL Parameters.
params :: Parameters
params = Parameters
  { _alpha            = 0.5
  , _beta             = 0.15
  , _delta            = 0.04
  , _gamma            = 0.30
  , _epsilon          = 0.075
  , _exploration      = 0.8
  , _learnRandomAbove = 0.0
  , _zeta             = 1.0
  , _xi               = 0.30
  }

-- | Decay function of parameters.
decay :: Decay
decay t  p@(Parameters alp bet del ga eps exp rand zeta xi)
  | t `mod` 200 == 0 =
    Parameters
      (max 0.03 $ slow * alp)
      (max 0.015 $ slow * bet)
      (max 0.015 $ slow * del)
      (max 0.01 $ slow * ga)
      (max 0.05 $ slow * eps) -- (0.5*bet)
      (max 0.01 $ slower * exp)
      rand
      zeta
      (max 0.08 $ slow * xi) -- 0.5*bet)
  | otherwise = p
  where
    slower = 0.995
    slow = 0.98
    faster = 1.0 / 0.99
    f = max 0.01


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


moveLeft :: St -> IO (Reward,St, EpisodeEnd)
moveLeft s =
  case s of
    One    -> return (0, Left 2, False)
    Left 5 -> return (5, One, False)
    Left x -> return (0, Left (x+1), False)
    _      -> moveRight s

moveRight :: St -> IO (Reward,St, EpisodeEnd)
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

