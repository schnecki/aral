{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.Parameters where

import           Control.DeepSeq
import           Control.Lens
import           Data.Serialize
import           GHC.Generics

import           ML.BORL.Decay.Type


-- Parameters
data Parameters = Parameters
  { _alpha              :: !Double             -- ^ for rho value
  , _alphaANN           :: !Double             -- ^ for rho value when training the ANN (e.g. after filling the replay memory)
  , _beta               :: !Double             -- ^ for V values
  , _betaANN            :: !Double             -- ^ for V value when training the ANN (e.g. after filling the replay memory)
  , _delta              :: !Double             -- ^ for W values
  , _deltaANN           :: !Double             -- ^ for W value when training the ANN
  , _gamma              :: !Double             -- ^ Gamma values for R0/R1.
  , _gammaANN           :: !Double             -- ^ Gamma values for R0/R1 when using the ANN.
  , _epsilon            :: !Double             -- ^ for comparison between state values
  , _exploration        :: !Double             -- ^ exploration rate
  , _learnRandomAbove   :: !Double             -- ^ Value which specifies until when randomized actions are learned.
                                               -- Learning from randomized actions disturbs the state values and thus
                                               -- hinders convergence. A too low high will result in sub-optimal
                                               -- policies as the agent does not learn while exploring the solution
                                               -- space.
  , _zeta               :: !Double             -- ^ Force bias optimality once the absolute error of psiV is less than
                                               -- or equal to this value.
  , _xi                 :: !Double             -- ^ Ratio in the interval (0,1) on how much of the difference of W value
                                               -- to enforce on V values.
  , _disableAllLearning :: !Bool               -- ^ Completely disable learning (e.g. for evaluation). Enabling
                                               -- increases performance.
  } deriving (Show, Eq, Ord, NFData, Generic, Serialize)
makeLenses ''Parameters


