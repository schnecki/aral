{-# LANGUAGE BangPatterns   #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
module ML.BORL.Algorithm where


import           ML.BORL.Types

import           Control.Arrow      (first)
import           Control.DeepSeq
import           Data.Serialize
import           GHC.Generics

import           ML.BORL.Decay.Type

type FractionStateValue = Float

data AvgReward
  = ByMovAvg Int
  | ByReward
  | ByStateValues
  | ByStateValuesAndReward !FractionStateValue !DecaySetup
  | Fixed !Float
  deriving (NFData, Show, Generic, Eq, Ord, Serialize)

type DecideOnVPlusPsi = Bool    -- ^ Decide actions on V + psiV? Otherwise on V solely.

type BalanceRho = Bool    -- ^ Probability of an episode end

-- | What comparison operation to use. Either an epsilon-sensitive lexicographic order or an exact comparison operation
-- (maximum, minimum). This groups the actions, and the policy chooses one of the grouped actions. See Parameters for
-- setting up the values.
data Comparison
  = EpsilonSensitive
  | Exact
  deriving (Ord, Eq, Show, Generic, NFData, Serialize)

type EpsilonMiddle = Float


data Algorithm s
  = AlgBORL !GammaLow
            !GammaHigh
            !AvgReward
            !(Maybe (s, ActionIndex))
  | AlgBORLVOnly !AvgReward !(Maybe (s, ActionIndex)) -- ^ DQN algorithm but subtracts average reward in every state
  | AlgDQN !Gamma !Comparison
  | AlgDQNAvgRewAdjusted !GammaMiddle !GammaHigh !AvgReward
  deriving (NFData, Show, Generic, Eq, Ord, Serialize)

mapAlgorithmState :: (s -> s') -> Algorithm s -> Algorithm s'
mapAlgorithmState f (AlgBORL gl gh avg mSt) = AlgBORL gl gh avg (first f <$> mSt)
mapAlgorithmState f (AlgBORLVOnly avg mSt) = AlgBORLVOnly avg (first f <$> mSt)
mapAlgorithmState _ (AlgDQN g c) = AlgDQN g c
mapAlgorithmState _ (AlgDQNAvgRewAdjusted gm gh avg) = AlgDQNAvgRewAdjusted gm gh avg


isAlgBorl :: Algorithm s -> Bool
isAlgBorl AlgBORL{} = True
isAlgBorl _         = False


isAlgDqn :: Algorithm s -> Bool
isAlgDqn AlgDQN{} = True
isAlgDqn _        = False

isAlgDqnAvgRewardFree :: Algorithm s -> Bool
isAlgDqnAvgRewardFree AlgDQNAvgRewAdjusted{} = True
isAlgDqnAvgRewardFree _                      = False

-- blackwellOptimalVersion :: Algorithm s -> Bool
-- blackwellOptimalVersion (AlgDQNAvgRewAdjusted Just{} _ _ _) = True
-- blackwellOptimalVersion AlgBORL{}                           = True
-- blackwellOptimalVersion _                                   = False

isAlgBorlVOnly :: Algorithm s -> Bool
isAlgBorlVOnly AlgBORLVOnly{} = True
isAlgBorlVOnly _              = False


defaultGamma0,defaultGamma1,defaultGammaDQN :: Float
defaultGamma0 = 0.50
defaultGamma1 = 0.80
defaultGammaDQN = 0.99


-- ^ Use BORL as algorithm with gamma values `defaultGamma0` and `defaultGamma1` for low and high gamma values.
algBORL :: Algorithm s
algBORL = AlgBORL defaultGamma0 defaultGamma1 ByStateValues Nothing

  -- (ByMovAvg 100) Normal False -- (DivideValuesAfterGrowth 1000 70000) False


-- ^ Use DQN as algorithm with `defaultGamma1` as gamma value. Algorithm implementation as in Mnih, Volodymyr, et al.
-- "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529.
algDQN :: Algorithm s
algDQN = AlgDQN defaultGammaDQN Exact


algDQNAvgRewardFree :: Algorithm s
algDQNAvgRewardFree = AlgDQNAvgRewAdjusted defaultGamma1 1.0 ByStateValues
