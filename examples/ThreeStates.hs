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

import           ML.BORL       hiding (actionFilter)

import           Helper

import           Control.Arrow (first, second)
import           Control.Lens  (set, (^.))
import           Control.Monad (foldM, unless, when)
import           System.IO
import           System.Random

main :: IO ()
main = do

  let rl = mkBORLUnichain initState actions actionFilter params decay
  askUser True usage cmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []

params :: Parameters
params = Parameters 0.005 0.07 0.07 1.5 1.0 0.1 1.0 0.2

initState :: St
initState = A

decay :: Period -> Parameters -> Parameters
decay t p@(Parameters alp bet del eps exp rand zeta xi)
  | t `mod` 100 == 0 = Parameters alp bet del (f $ slower * eps) (max 0.01 $ slower * exp) rand zeta xi
  | otherwise = p

  where slower = 0.99
        slow = 0.95
        f = max 0.05


-- State
data St = B | A | C deriving (Ord, Eq, Show)
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
