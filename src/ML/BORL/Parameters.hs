{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.Parameters where

import           Control.DeepSeq
import           Control.Lens
import           Data.Serialize
import           GHC.Generics

import           ML.BORL.Decay.Type
import           ML.BORL.Exploration

type ParameterInitValues = Parameters Double
type ParameterDecayedValues = Parameters Double
type ParameterDecaySetting = Parameters DecaySetup

-- Parameters
data Parameters a = Parameters
  { _alpha               :: !a             -- ^ for rho value
  , _alphaANN            :: !a             -- ^ for rho value when training the ANN (e.g. after filling the replay memory)
  , _beta                :: !a             -- ^ for V values
  , _betaANN             :: !a             -- ^ for V value when training the ANN (e.g. after filling the replay memory)
  , _delta               :: !a             -- ^ for W values
  , _deltaANN            :: !a             -- ^ for W value when training the ANN
  , _gamma               :: !a             -- ^ Gamma values for R0/R1.
  , _gammaANN            :: !a             -- ^ Gamma values for R0/R1 when using the ANN.
  , _epsilon             :: !a             -- ^ for comparison between state values
  , _explorationStrategy :: ExplorationStrategy -- ^ Strategy for exploration.
  , _exploration         :: !a             -- ^ exploration rate
  , _learnRandomAbove    :: !a             -- ^ Value which specifies until when randomized actions are learned.
                                          -- Learning from randomized actions disturbs the state values and thus
                                          -- hinders convergence. A too low high will result in sub-optimal
                                          -- policies as the agent does not learn while exploring the solution
                                          -- space.
  , _zeta                :: !a             -- ^ Force bias optimality once the absolute error of psiV is less than
                                          -- or equal to this value.
  , _xi                  :: !a             -- ^ Ratio in the interval (0,1) on how much of the difference of W value
                                               -- to enforce on V values.
  , _disableAllLearning  :: Bool               -- ^ Completely disable learning (e.g. for evaluation). Enabling
                                               -- increases performance.
  } deriving (Show, Eq, Ord, NFData, Generic, Serialize)
makeLenses ''Parameters


