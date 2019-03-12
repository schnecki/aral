{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
module ML.BORL.Algorithm where


import           ML.BORL.Types

import           Control.DeepSeq
import           GHC.Generics

data Algorithm
  = AlgBORL GammaLow
            GammaHigh
  | AlgDQN Gamma
  deriving (NFData, Generic, Eq, Ord)


isAlgBorl :: Algorithm -> Bool
isAlgBorl AlgBORL{} = True
isAlgBorl _         = False


isAlgDqn :: Algorithm -> Bool
isAlgDqn AlgDQN{} = True
isAlgDqn _        = False


defaultGamma0,defaultGamma1 :: Double
defaultGamma0 = 0.25
defaultGamma1 = 0.99


-- ^ Use BORL as algorithm with gamma values `defaultGamma0` and `defaultGamma1` for low and high gamma values.
algBORL :: Algorithm
algBORL = AlgBORL defaultGamma0 defaultGamma1


-- ^ Use DQN as algorithm with `defaultGamma1` as gamma value. Algorithm implementation as in Mnih, Volodymyr, et al.
-- "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529.
algDQN :: Algorithm
algDQN = AlgDQN defaultGamma1
