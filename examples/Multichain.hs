{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE TypeFamilies               #-}
-- This is example is a multichain example from Puttermann 1994 (Example 8.2.2). For multichain MDPs the average reward
-- value may differ between set of recurrent states. There are no decisions, moves are by probability.

-- NOTE: Multichain MDP support is highly experimental!

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

import           Control.Arrow   (first, second)
import           Control.DeepSeq (NFData)
import           Control.Lens    (set, (^.))
import           Control.Monad   (foldM, unless, when)
import           Data.List       (foldl')
import           GHC.Generics
import           System.IO
import           System.Random

main :: IO ()
main = do

  let rl = mkMultichainTabular algBORL initState (\(St x) -> [fromIntegral x]) actions (const $ repeat True) params decay Nothing
  -- let rl = mkUnichainTabular algBORL initState (\(St x) -> [fromIntegral x]) actions (const $ repeat True) params decay Nothing
  askUser Nothing True usage cmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []

initState :: St
initState = St 5

-- | BORL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha              = 0.001
    , _alphaANN           = 0.5
    , _beta               = 0.001
    , _betaANN            = 1
    , _delta              = 0.001
    , _deltaANN           = 1
    , _gamma              = 0.0005
    , _gammaANN           = 1
    , _epsilon            = 5
    , _exploration        = 0.8
    , _learnRandomAbove   = 0.0
    , _zeta               = 0.0
    , _xi                 = 0.01
    , _disableAllLearning = False
    }

-- | Decay function of parameters.
decay :: Decay
decay t p = exponentialDecayParameters (Just minValues) 0.50 300000 t $
            exponentialDecayParametersValue alpha Nothing 0.25 300000 t $
            exponentialDecayParametersValue gamma Nothing 0.25 500000 t $
            exponentialDecayParametersValue xi Nothing 0.25 500000 t p
  where
    minValues =
      Parameters
        { _alpha = 0.0001
        , _alphaANN = 0.5
        , _beta = 0.0001
        , _betaANN = 1.0
        , _delta = 0.0001
        , _deltaANN = 1.0
        , _gamma = 0.0001
        , _gammaANN = 1.0
        , _epsilon = 2
        , _exploration = 0.01
        , _learnRandomAbove = 0.05
        , _zeta = 0.0
        , _xi = 0.01
        , _disableAllLearning = False
        }

-- State
newtype St = St Integer deriving (Ord, Eq, Show, NFData, Generic)
type R = Double
type P = Double

instance RewardFuture St where
  type StoreType St = ()


-- Actions
actions :: [Action St]
actions =  [Action (addReset move) "move"]

addReset :: (St -> IO (Reward St, St, EpisodeEnd)) -> St -> IO (Reward St, St, EpisodeEnd)
addReset f st = do
  r <- randomRIO (0,1)
  if r < (0.003 :: Double)
    then do
    x <- randomRIO (4,5)
    return (Reward 0, St x, True)
    else f st

move :: St -> IO (Reward St,St,EpisodeEnd)
move s = do
  rand <- randomRIO (0, 1 :: Double)
  let possMove = case s of
         St 1 -> [(1.0, (Reward 2, St 1,False))]
         St 2 -> [(0.4, (Reward 5, St 2,False)), (0.6, (Reward 5, St 3,False))]
         St 3 -> [(0.7, (Reward 4, St 2,False)), (0.3, (Reward 4, St 3,False))]
         St 4 -> [(0.5, (Reward 1, St 1,False)), (0.2, (Reward 1, St 2,False)), (0.3, (Reward 1, St 5,False))]
         St 5 -> [(0.2, (Reward 3, St 1,False)), (0.3, (Reward 3, St 2,False)), (0.3, (Reward 3, St 3,False)), (0.2, (Reward 3, St 4,False))]
  return $ snd $ snd $ foldl' (\(ps, c) c'@(p,_) -> if ps <= rand && ps + p > rand then (ps + p, c') else (ps + p, c)) (0, head possMove) possMove

