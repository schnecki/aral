-- This is example is a multichain example from Puttermann 1994 (Example 8.2.2). For multichain MDPs the average reward
-- value may differ between set of recurrent states.

-- Solution as presented in Puttermann's book (pages 345/346):

-- Average Rewards:
-- g(s1) = 2
-- g(s2) = 4.538
-- g(s3) = 4.538
-- g(s4) = 2.882
-- g(s5) = 3.245

-- Bias values with $P^\star * h = 0$ to uniquely determine h:
-- h(s1) = 0
-- h(s2) = 0.354
-- h(s3) = -0.416
-- h(s4) = 2.011
-- h(s5) = 0.666


module Main where

import           ML.BORL

import           Helper

import           Control.Arrow (first, second)
import           Control.Lens  (set, (^.))
import           Control.Monad (foldM, unless, when)
import           System.IO
import           System.Random

main :: IO ()
main = do

  let rl = mkBORLMultichain initState actions (const $ repeat True) params decay
  askUser True usage cmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []

params :: Parameters
params = Parameters 0.15 0.05 0.05 1.0 1.0 0.1 0.75 0.2


initState :: St
initState = St 5

decay :: Period -> Parameters -> Parameters
decay t p@(Parameters alp bet del eps exp rand zeta xi)
  | t `mod` 200 == 0 = Parameters (f $ slower * alp) (f $ slow * bet) (f $ slow * del) (max 0.1 $ slower * eps) (f $ slower * exp) rand zeta xi -- (1 - slower * (1-frc)) mRho
  | otherwise = p

  where slower = 0.995
        slow = 0.95
        faster = 1.0/0.995
        f = max 0.001


-- State
newtype St = St Integer deriving (Ord, Eq, Show)
type R = Double
type P = Double

-- Actions
actions :: [St -> IO [(Probability, (Reward, St))]]
actions = reset : replicate 300 move

reset :: St -> IO [(Probability, (Reward, St))]
reset s = do x <- randomRIO (4,5)
             return [(1, (0, St x))]

move :: St -> IO [(Probability, (Reward, St))]
move s =
  return $
  case s of
    St 1 -> [(1, (2, St 1))]
    St 2 -> [(0.4, (5, St 2)), (0.6, (5, St 3))]
    St 3 -> [(0.7, (4, St 2)), (0.3, (4, St 3))]
    St 4 -> [(0.5, (1, St 1)), (0.2, (1, St 2)), (0.3, (1, St 5))]
    St 5 -> [(0.2, (3, St 1)), (0.3, (3, St 2)), (0.3, (3, St 3)), (0.2, (3, St 4))]
