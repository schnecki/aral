{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
module ML.BORL.Algorithm where


import           ML.BORL.Types

import           Control.DeepSeq
import           Data.Serialize
import           GHC.Generics

data AvgReward
  = ByMovAvg Int
  | ByReward
  | ByStateValues
  | Fixed Double
  deriving (NFData, Show, Generic, Eq, Ord, Serialize)

data StateValueHandling
  = Normal
  | DivideValuesAfterGrowth Int Period -- ^ How many periods to track, until how many periods to perform divisions.
  deriving (NFData, Show, Generic, Eq, Ord, Serialize)

type DecideOnVPlusPsi = Bool    -- ^ Decide actions on V + psiV? Otherwise on V solely.

data Algorithm
  = AlgBORL GammaLow
            GammaHigh
            AvgReward
            StateValueHandling
            DecideOnVPlusPsi
  | AlgBORLVOnly AvgReward -- ^ DQN algorithm but subtracts average reward in every state
  | AlgDQN Gamma
  deriving (NFData, Show, Generic, Eq, Ord, Serialize)


isAlgBorl :: Algorithm -> Bool
isAlgBorl AlgBORL{} = True
isAlgBorl _         = False


isAlgDqn :: Algorithm -> Bool
isAlgDqn AlgDQN{} = True
isAlgDqn _        = False

isAlgBorlVOnly :: Algorithm -> Bool
isAlgBorlVOnly AlgBORLVOnly{} = True
isAlgBorlVOnly _              = False


defaultGamma0,defaultGamma1,defaultGammaDQN :: Double
defaultGamma0 = 0.50
defaultGamma1 = 0.80
defaultGammaDQN = 0.99


-- ^ Use BORL as algorithm with gamma values `defaultGamma0` and `defaultGamma1` for low and high gamma values.
algBORL :: Algorithm
algBORL = AlgBORL defaultGamma0 defaultGamma1 ByStateValues Normal False

  -- (ByMovAvg 100) Normal False -- (DivideValuesAfterGrowth 1000 70000) False


-- ^ Use DQN as algorithm with `defaultGamma1` as gamma value. Algorithm implementation as in Mnih, Volodymyr, et al.
-- "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529.
algDQN :: Algorithm
algDQN = AlgDQN defaultGammaDQN
