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
-- V(A) = 0.5
-- V(B) = -0.5
-- V(C) = 1.5

-- when selecting action a2 (A->C)
-- V(A) = -0.5
-- V(B) = -1.5
-- V(C) = 0.5

-- Thus the policy selecting a1 (going Left) is preferable.

module Main where

import           ML.BORL                                        hiding (actionFilter)

import           Helper

import           Control.DeepSeq                                (NFData)
import           Control.Lens
import           Data.Int                                       (Int64)
import           Data.List                                      (genericLength)
import qualified Data.Vector                                    as V
import Data.Text (Text)
import           GHC.Exts                                       (fromList)
import           GHC.Generics
import           Grenade                                        hiding (train)


import qualified TensorFlow.Build                               as TF (addNewOp,
                                                                       evalBuildT,
                                                                       explicitName, opDef,
                                                                       opDefWithName,
                                                                       opType, runBuildT,
                                                                       summaries)
import qualified TensorFlow.Core                                as TF hiding (value)
import qualified TensorFlow.GenOps.Core                         as TF (abs, add,
                                                                       approximateEqual,
                                                                       approximateEqual,
                                                                       assign, cast,
                                                                       getSessionHandle,
                                                                       getSessionTensor,
                                                                       identity',
                                                                       lessEqual,
                                                                       lessEqual, matMul,
                                                                       mul,
                                                                       readerSerializeState,
                                                                       relu', shape, square,
                                                                       sub, tanh, tanh',
                                                                       truncatedNormal)
import qualified TensorFlow.Minimize                            as TF
import qualified TensorFlow.Ops                                 as TF (initializedVariable,
                                                                       initializedVariable',
                                                                       placeholder,
                                                                       placeholder',
                                                                       reduceMean,
                                                                       reduceSum, restore,
                                                                       save, scalar,
                                                                       vector,
                                                                       zeroInitializedVariable,
                                                                       zeroInitializedVariable')
import qualified TensorFlow.Tensor                              as TF (Ref (..),
                                                                       collectAllSummaries,
                                                                       tensorNodeName,
                                                                       tensorRefFromName,
                                                                       tensorValueFromName)


type NN = Network '[ FullyConnected 1 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 2, Tanh] '[ 'D1 1, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 2, 'D1 2]

nnConfig :: NNConfig St
nnConfig = NNConfig
  { _toNetInp             = netInp
  , _replayMemory         = mkReplayMemory 10000
  , _trainBatchSize       = 1
  , _learningParams       = LearningParameters 0.005 0.0 0.0000
  , _prettyPrintElems     = [minBound .. maxBound] :: [St]
  , _scaleParameters      = scalingByMaxReward False 2
  , _updateTargetInterval = 5000
  , _trainMSEMax          = 0.015
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
modelBuilder = buildModel $ inputLayer1D numInputs >> fullyConnected1D 20 TF.relu' >> fullyConnected1D 10 TF.relu' >> fullyConnected1D numActions TF.tanh' >> trainingByAdam1D

main :: IO ()
main = do
  -- createModel >>= mapM_ testRun

  nn <- randomNetworkInitWith HeEtAl :: IO NN

  -- rl <- mkBORLUnichainGrenade initState actions actionFilter params decay nn nnConfig
  rl <- mkBORLUnichainTensorflow initState actions actionFilter params decay modelBuilder nnConfig
  -- let rl = mkBORLUnichainTabular initState actions actionFilter params decay
  askUser True usage cmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []


initState :: St
initState = A


-- | BORL Parameters.
params :: Parameters
params = Parameters
  { _alpha            = 0.2
  , _beta             = 0.25
  , _delta            = 0.25
  , _epsilon          = 1.0
  , _exploration      = 1.0
  , _learnRandomAbove = 0.0
  , _zeta             = 1.0
  , _xi               = 0.5
  }


-- | Decay function of parameters.
decay :: Period -> Parameters -> Parameters
decay t p@(Parameters alp bet del eps exp rand zeta xi)
  | t > 0 && t `mod` 200 == 0 =
    Parameters
      (max 0.0001 $ slow * alp)
      (f $ slower * bet)
      (f $ slower * del)
      (max 0.1 $ slow * eps)
      (f $ slow * exp)
      rand
      zeta -- (fromIntegral t / 20000) --  * zeta)
      xi -- (max 0 $ fromIntegral t / 40000) -- * xi)
  | otherwise = p
  where
    slower = 0.995
    slow = 0.95
    faster = 1.0 / 0.995
    f = max 0.001


-- State
data St = B | A | C deriving (Ord, Eq, Show, Enum, Bounded,NFData,Generic)
type R = Double
type P = Double

-- Actions
actions :: [Action St]
actions =
  [ Action moveLeft "left "
  , Action moveRight "right"]

actionFilter :: St -> [Bool]
actionFilter A = [True, True]
actionFilter B = [False, True]
actionFilter C = [True, False]


moveLeft :: St -> IO (Reward,St)
moveLeft s =
  return $
  case s of
    A -> (2, B)
    B -> error "not allowed"
    C -> (2, A)

moveRight :: St -> IO (Reward,St)
moveRight s =
  return $
  case s of
    A -> (0, C)
    B -> (0, A)
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

