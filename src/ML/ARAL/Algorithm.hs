{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
module ML.ARAL.Algorithm where


import           ML.ARAL.Types

import           Control.Arrow      (first)
import           Control.DeepSeq
import           Data.Serialize
import           GHC.Generics

import           ML.ARAL.Decay.Type

type FractionStateValue = Double

data AvgReward
  = ByMovAvg Int
  | ByReward
  | ByStateValues
  | ByStateValuesAndReward !FractionStateValue !DecaySetup
  | Fixed !Double
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

type EpsilonMiddle = Double


data Algorithm s
  = AlgNBORL !GammaLow
            !GammaHigh
            !AvgReward
            !(Maybe (s, [ActionIndex]))
  | AlgARALVOnly !AvgReward !(Maybe (s, [ActionIndex])) -- ^ DQN algorithm but subtracts average reward in every state
  | AlgDQN !Gamma !Comparison
  | AlgRLearning
  | AlgARAL !GammaMiddle !GammaHigh !AvgReward
  deriving (NFData, Show, Generic, Eq, Ord, Serialize)

mapAlgorithmState :: (s -> s') -> Algorithm s -> Algorithm s'
mapAlgorithmState f (AlgNBORL gl gh avg mSt) = AlgNBORL gl gh avg (first f <$> mSt)
mapAlgorithmState f (AlgARALVOnly avg mSt)   = AlgARALVOnly avg (first f <$> mSt)
mapAlgorithmState _ (AlgDQN g c)             = AlgDQN g c
mapAlgorithmState _ AlgRLearning             = AlgRLearning
mapAlgorithmState _ (AlgARAL gm gh avg)      = AlgARAL gm gh avg


isAlgBorl :: Algorithm s -> Bool
isAlgBorl AlgNBORL{} = True
isAlgBorl _          = False


isAlgDqn :: Algorithm s -> Bool
isAlgDqn AlgDQN{} = True
isAlgDqn _        = False

isAlgDqnAvgRewardAdjusted :: Algorithm s -> Bool
isAlgDqnAvgRewardAdjusted AlgARAL{} = True
isAlgDqnAvgRewardAdjusted _         = False

-- blackwellOptimalVersion :: Algorithm s -> Bool
-- blackwellOptimalVersion (AlgARAL Just{} _ _ _) = True
-- blackwellOptimalVersion AlgNBORL{}                           = True
-- blackwellOptimalVersion _                                   = False

isAlgBorlVOnly :: Algorithm s -> Bool
isAlgBorlVOnly AlgARALVOnly{} = True
isAlgBorlVOnly _              = False


defaultGamma0,defaultGamma1,defaultGammaDQN :: Double
defaultGamma0 = 0.50
defaultGamma1 = 0.80
defaultGammaDQN = 0.99


-- ^ Use ARAL as algorithm with gamma values `defaultGamma0` and `defaultGamma1` for low and high gamma values.
algNBORL :: Algorithm s
algNBORL = AlgNBORL defaultGamma0 defaultGamma1 ByStateValues Nothing

  -- (ByMovAvg 100) Normal False -- (DivideValuesAfterGrowth 1000 70000) False


-- ^ Use DQN as algorithm with `defaultGamma1` as gamma value. Algorithm implementation as in Mnih, Volodymyr, et al.
-- "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529.
algDQN :: Algorithm s
algDQN = AlgDQN defaultGammaDQN Exact


algDQNAvgRewardFree :: Algorithm s
algDQNAvgRewardFree = AlgARAL defaultGamma1 1.0 ByStateValues
