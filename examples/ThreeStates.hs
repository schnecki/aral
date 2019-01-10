{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE DeriveAnyClass    #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE OverloadedStrings #-}
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

import           ML.BORL             hiding (actionFilter)

import           Helper

import           Control.DeepSeq     (NFData)
import           GHC.Generics
import           Grenade

import           Data.Int            (Int32, Int64)
import           GHC.Exts            (fromList)
import qualified TensorFlow.Core     as TF
import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.Ops      as TF hiding (initializedVariable,
                                            zeroInitializedVariable)
import qualified TensorFlow.Variable as TF

type NN = Network '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 1, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 1, 'D1 1]

nnConfig :: NNConfig St
nnConfig = NNConfig
  { _toNetInp             = netInp
  , _replayMemory         = mkReplayMemory 10000
  , _trainBatchSize       = 1
  , _learningParams       = LearningParameters 0.005 0.0 0.0000
  , _prettyPrintElems     = [minBound .. maxBound] :: [St]
  , _scaleParameters      = scalingByMaxReward False 2
  , _updateTargetInterval = 5000
  , _trainMSEMax          = 0.05
  }


netInp :: St -> [Double]
netInp st = [scaleNegPosOne (minVal,maxVal) (fromIntegral $ fromEnum st)]

maxVal :: Double
maxVal = fromIntegral $ fromEnum (maxBound :: St)

minVal :: Double
minVal = fromIntegral $ fromEnum (minBound :: St)


tensorflow
     -- Use -1 batch size to support variable sized batches.
 = do
  let batchSize = -1
      numInputs = 2    :: Int64

    -- Inputs.
  images <- TF.placeholder (fromList [2])
    -- Hidden layer.
  let numUnits = 200
  hiddenWeights <- TF.initializedVariable =<< randomParam numPixels [numPixels, numUnits]
  hiddenBiases <- TF.zeroInitializedVariable [numUnits]
  let hiddenZ = (images `TF.matMul` TF.readValue hiddenWeights) `TF.add` TF.readValue hiddenBiases
  let hidden = TF.relu hiddenZ
    -- Logits.
  logitWeights <- TF.initializedVariable =<< randomParam numUnits [numUnits, numInputs]
  logitBiases <- TF.zeroInitializedVariable [numInputs]
  let logits = (hidden `TF.matMul` TF.readValue logitWeights) `TF.add` TF.readValue logitBiases
  TF.render $ TF.cast $ TF.argMax (TF.softmax logits) (TF.scalar (1 :: Double))


main :: IO ()
main = do

  nn <- randomNetworkInitWith HeEtAl :: IO NN


  let rl = mkBORLUnichainGrenade initState actions actionFilter params decay nn nnConfig
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
  , _learnRandomAbove = 0.1
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
      (fromIntegral t / 20000) --  * zeta)
      (max 0 $ fromIntegral t / 40000) -- * xi)
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
    B -> (0, A)
    C -> (2, A)

moveRight :: St -> IO (Reward,St)
moveRight s =
  return $
  case s of
    A -> (0, C)
    B -> (0, A)
    C -> (2, A)
