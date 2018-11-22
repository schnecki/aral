{-# LANGUAGE DataKinds         #-}
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

import           ML.BORL hiding (actionFilter)

import           Helper

import           Grenade

type NN = Network '[ FullyConnected 2 4, Relu, FullyConnected 4 1, Tanh] '[ 'D1 2, 'D1 4, 'D1 4, 'D1 1, 'D1 1]

nnConfig :: NNConfig St
nnConfig = NNConfig netInp [] 32 (LearningParameters 0.001 0.0 0.0001) ([minBound .. maxBound] :: [St]) (scalingByMaxReward 2) 3000

netInp :: St -> [Double]
netInp st = [scaleNegPosOne (minVal,maxVal) (fromIntegral $ fromEnum st)]

maxVal :: Double
maxVal = fromIntegral $ fromEnum (maxBound :: St)

minVal :: Double
minVal = fromIntegral $ fromEnum (minBound :: St)


main :: IO ()
main = do

  nn <- randomNetworkInitWith HeEtAl :: IO NN

  let rl = mkBORLUnichain initState actions actionFilter params decay nn nnConfig
  -- let rl = mkBORLUnichainTabular initState actions actionFilter params decay
  askUser True usage cmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []

initState :: St
initState = A


-- | BORL Parameters.
params :: Parameters
params = Parameters 0.2 0.5 0.5 1.0 1.0 0.1 0.25 0.5


-- | Decay function of parameters.
decay :: Period -> Parameters -> Parameters
decay t p@(Parameters alp bet del eps exp rand zeta xi)
  | t `mod` 200 == 0 = Parameters (max 0.0001 $ slow * alp) (f $ slower * bet) (f $ slower * del) (max 0.1 $ slower * eps) (f $ slower * exp) rand zeta xi -- (1 - slower * (1-frc)) mRho
  | otherwise = p

  where slower = 0.995
        slow = 0.95
        faster = 1.0/0.995
        f = max 0.001


-- State
data St = B | A | C deriving (Ord, Eq, Show, Enum, Bounded)
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
