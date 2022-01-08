{-# LANGUAGE DeriveAnyClass             #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedLists            #-}
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

import           Control.Arrow        (first, second)
import           Control.DeepSeq      (NFData)
import           Control.Lens         (set, (^.))
import           Control.Monad        (foldM, unless, when)
import           Data.Default
import           Data.List            (foldl')
import qualified Data.Vector.Storable as V
import           GHC.Generics
import           System.IO
import           System.Random

import           ML.ARAL

import           Helper

alg :: Algorithm St
alg =
  AlgDQNAvgRewAdjusted 0.8 1.0 ByStateValues
  -- AlgARAL 0.5 0.8 ByStateValues Nothing

main :: IO ()
main = do
  rl <- mkMultichainTabular alg (liftInitSt initState) (\(St x) -> V.singleton (fromIntegral x)) actionFun actionFil params decay borlSettings Nothing

  -- one can still treat it as unichain
  -- rl <- mkUnichainTabular alg (liftInitSt initState) (\(St x) -> V.singleton (fromIntegral x)) actionFun actionFil params decay borlSettings Nothing
  askUser Nothing True usage cmds [] rl -- maybe increase learning by setting estimate of rho
  where
    cmds = []
    usage = []

actionFil :: ActionFilter St
actionFil _ = [V.replicate (length actions) True]

borlSettings :: Settings
borlSettings = def

initState :: St
initState = St 5

-- -- | ARAL Parameters.
-- params :: ParameterInitValues
-- params =
--   Parameters
--     { _alpha              = 0.01
--     , _alphaRhoMin = 2e-5
--     , _beta               = 0.01
--     , _delta              = 0.01
--     , _gamma              = 0.01
--     , _epsilon            = 0.1

--     , _exploration        = 1.0
--     , _learnRandomAbove   = 0.30
--     , _zeta               = 0.03
--     , _xi                 = 0.01

--     }

-- -- | Decay function of parameters.
-- decay :: ParameterDecaySetting
-- decay =
--     Parameters
--       { _alpha            = ExponentialDecay (Just 0) 0.75 50000
--       , _alphaRhoMin      = NoDecay
--       , _beta             = ExponentialDecay (Just 1e-3) 0.75 50000
--       , _delta            = ExponentialDecay (Just 1e-3) 0.75 50000
--       , _gamma            = ExponentialDecay (Just 1e-3) 0.75 50000
--       , _zeta             = NoDecay
--       , _xi               = NoDecay
--         -- Exploration
--       , _epsilon          = [NoDecay]
--       , _exploration      = ExponentialDecay (Just 0.30) 0.75 150000
--       , _learnRandomAbove = NoDecay
--       }

-- | ARAL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha               = 0.01
    , _alphaRhoMin = 2e-5
    , _beta                = 0.01
    , _delta               = 0.005
    , _gamma               = 0.01
    , _zeta                = 0.03
    , _xi                  = 0.005
    -- Exploration
    , _epsilon             = 0.25

    , _exploration         = 1.0
    , _learnRandomAbove    = 0.99

    }

-- | Decay function of parameters.
decay :: ParameterDecaySetting
decay =
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-5) 0.5 50000  -- 5e-4
      , _alphaRhoMin      = NoDecay
      , _beta             = ExponentialDecay (Just 1e-4) 0.5 150000
      , _delta            = ExponentialDecay (Just 5e-4) 0.5 150000
      , _gamma            = ExponentialDecay (Just 1e-3) 0.5 150000
      , _zeta             = ExponentialDecay (Just 0) 0.5 150000
      , _xi               = NoDecay
      -- Exploration
      , _epsilon          = [NoDecay] -- [ExponentialDecay (Just 0.050) 0.05 150000]
      , _exploration      = ExponentialDecay (Just 0.01) 0.50 100000
      , _learnRandomAbove = NoDecay
      }


-- State
newtype St = St Integer deriving (Ord, Eq, Show, NFData, Generic)

instance RewardFuture St where
  type StoreType St = ()


-- Actions
data Act = Move
  deriving (Show, Eq, Ord, Enum, Bounded, Generic, NFData)

actions :: [Action Act]
actions = [Move]

actionFun :: ActionFunction St Act
actionFun tp st [Move] = addReset move tp st

addReset :: (AgentType -> St -> IO (Reward St, St, EpisodeEnd)) -> AgentType -> St -> IO (Reward St, St, EpisodeEnd)
addReset f tp st = do
  r <- randomRIO (0, 1)
  if r < (0.01 :: Double)
    then do
      x <- randomRIO (4, 5)
      return (Reward 0, St x, True)
    else f tp st

move :: AgentType -> St -> IO (Reward St, St, EpisodeEnd)
move _ s = do
  rand <- randomRIO (0, 1 :: Double)
  let possMove = case s of
         St 1 -> [(1.0, (Reward 2, St 1,False))]
         St 2 -> [(0.4, (Reward 5, St 2,False)), (0.6, (Reward 5, St 3,False))]
         St 3 -> [(0.7, (Reward 4, St 2,False)), (0.3, (Reward 4, St 3,False))]
         St 4 -> [(0.5, (Reward 1, St 1,False)), (0.2, (Reward 1, St 2,False)), (0.3, (Reward 1, St 5,False))]
         St 5 -> [(0.2, (Reward 3, St 1,False)), (0.3, (Reward 3, St 2,False)), (0.3, (Reward 3, St 3,False)), (0.2, (Reward 3, St 4,False))]
  return $ snd $ snd $ foldl' (\(ps, c) c'@(p,_) -> if ps <= rand && ps + p > rand then (ps + p, c') else (ps + p, c)) (0, head possMove) possMove
