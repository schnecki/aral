{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
module ML.BORL.Algorithm where


import           ML.BORL.Types

import           Control.Arrow   (first)
import           Control.DeepSeq
import           Data.Serialize
import           GHC.Generics

type RatioStateValue = Double

data AvgReward
  = ByMovAvg Int
  | ByReward
  | ByStateValues
  | ByStateValuesAndReward RatioStateValue
  | Fixed Double
  deriving (NFData, Show, Generic, Eq, Ord, Serialize)

data StateValueHandling
  = Normal
  | DivideValuesAfterGrowth Int Period -- ^ How many periods to track, until how many periods to perform divisions.
  deriving (NFData, Show, Generic, Eq, Ord, Serialize)

type DecideOnVPlusPsi = Bool    -- ^ Decide actions on V + psiV? Otherwise on V solely.

type BalanceRho = Bool    -- ^ Probability of an episode end

data Algorithm s
  = AlgBORL GammaLow
            GammaHigh
            AvgReward
            StateValueHandling
            DecideOnVPlusPsi
            (Maybe (s, ActionIndex))
  | AlgBORLVOnly AvgReward (Maybe (s, ActionIndex)) -- ^ DQN algorithm but subtracts average reward in every state
  | AlgDQN Gamma
  | AlgDQNAvgRewardFree GammaLow GammaHigh AvgReward
  deriving (NFData, Show, Generic, Eq, Ord, Serialize)


mapAlgorithm :: (s -> s') -> Algorithm s -> Algorithm s'
mapAlgorithm f (AlgBORLVOnly avg mSA)     = AlgBORLVOnly avg (first f <$> mSA)
mapAlgorithm f (AlgBORL g0 g1 avg st dec mSA) = AlgBORL g0 g1 avg st dec (first f <$> mSA)
mapAlgorithm _ (AlgDQN ga)                  = AlgDQN ga
mapAlgorithm _ (AlgDQNAvgRewardFree ga0 ga1 avg) = AlgDQNAvgRewardFree ga0 ga1 avg


isAlgBorl :: Algorithm s -> Bool
isAlgBorl AlgBORL{} = True
isAlgBorl _         = False


isAlgDqn :: Algorithm s -> Bool
isAlgDqn AlgDQN{} = True
isAlgDqn _        = False

isAlgDqnAvgRewardFree :: Algorithm s -> Bool
isAlgDqnAvgRewardFree AlgDQNAvgRewardFree{} = True
isAlgDqnAvgRewardFree _                     = False


isAlgBorlVOnly :: Algorithm s -> Bool
isAlgBorlVOnly AlgBORLVOnly{} = True
isAlgBorlVOnly _              = False


defaultGamma0,defaultGamma1,defaultGammaDQN :: Double
defaultGamma0 = 0.50
defaultGamma1 = 0.80
defaultGammaDQN = 0.99


-- ^ Use BORL as algorithm with gamma values `defaultGamma0` and `defaultGamma1` for low and high gamma values.
algBORL :: Algorithm s
algBORL = AlgBORL defaultGamma0 defaultGamma1 ByStateValues Normal False Nothing

  -- (ByMovAvg 100) Normal False -- (DivideValuesAfterGrowth 1000 70000) False


-- ^ Use DQN as algorithm with `defaultGamma1` as gamma value. Algorithm implementation as in Mnih, Volodymyr, et al.
-- "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529.
algDQN :: Algorithm s
algDQN = AlgDQN defaultGammaDQN


algDQNAvgRewardFree :: Algorithm s
algDQNAvgRewardFree = AlgDQNAvgRewardFree defaultGamma0 defaultGammaDQN ByStateValues
