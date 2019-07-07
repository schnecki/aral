{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.Parameters where

import           Control.DeepSeq
import           Control.Lens
import           Data.Serialize
import           GHC.Generics


-- Parameters
data Parameters = Parameters
  { _alpha            :: !Double             -- ^ for rho value
  , _beta             :: !Double             -- ^ for V values
  , _delta            :: !Double             -- ^ for W values
  , _gamma            :: !Double             -- ^ Gamma values for R0/R1.
  , _epsilon          :: !Double             -- ^ for comparison between state values
  , _exploration      :: !Double             -- ^ exploration rate
  , _learnRandomAbove :: !Double             -- ^ Value which specifies until when randomized actions are learned. Learning from
                                             -- randomized actions disturbs the state values and thus hinders convergence. A too low
                                             -- high will result in sub-optimal policies as the agent does not learn while exploring
                                             -- the solution space.
  , _zeta             :: !Double             -- ^ Force bias optimality once the absolute error of psiV is less than or
                                             -- equal to this value.
  , _xi               :: !Double             -- ^ Ratio in the interval (0,1) on how much of the difference of W value to enforce on V values.
  } deriving (Show, Eq, Ord, NFData, Generic, Serialize)
makeLenses ''Parameters


