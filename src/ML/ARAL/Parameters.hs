{-# LANGUAGE DeriveAnyClass    #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedLists   #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE TypeFamilies      #-}
module ML.ARAL.Parameters where

import           Control.DeepSeq
import           Control.Lens
import           Data.Default
import           Data.Serialize
import           GHC.Generics

import           ML.ARAL.Decay.Type
import           ML.ARAL.InftyVector

type ParameterInitValues = Parameters Double
type ParameterDecayedValues = Parameters Double
type ParameterDecaySetting = Parameters DecaySetup

-- Parameters
data Parameters a = Parameters
  { _alpha            :: a               -- ^ for rho value
  , _alphaRhoMin      :: a               -- ^ Alpha for minimum rho value.
  , _beta             :: a               -- ^ for V values
  , _delta            :: a               -- ^ for W values
  , _gamma            :: a               -- ^ Gamma values for R0/R1.
  , _epsilon          :: InftyVector a   -- ^ for comparison between state values. E.g.: [R0, R1], or [Rho, V]
  , _exploration      :: a               -- ^ exploration rate
  , _learnRandomAbove :: a               -- ^ Value which specifies until when randomized actions are learned. Learning from randomized actions disturbs the state values and thus hinders convergence.
                                         -- A too low high will result in sub-optimal policies as the agent does not learn while exploring the solution space.
  , _zeta             :: a               -- ^ Force bias optimality once the absolute error of psiV is less than or equal to this value.
  , _xi               :: a               -- ^ Ratio in the interval (0,1) on how much of the difference of W value to enforce on V values.
  } deriving (Show, Eq, Ord, NFData, Generic, Serialize)
makeLenses ''Parameters

mkStaticDecayedParams :: ParameterDecayedValues -> ParameterDecayedValues
mkStaticDecayedParams params = Parameters
  { _alpha            = 0
  , _alphaRhoMin      = 0
  , _beta             = 0
  , _delta            = 0
  , _gamma            = 0
  , _epsilon          = params ^. epsilon
  , _exploration      = params ^. exploration
  , _learnRandomAbove = params ^. learnRandomAbove
  , _zeta             = 0
  , _xi               = 0
  }


instance Default (Parameters Double) where
  def =
    Parameters
    { _alpha               = 0.01
    , _alphaRhoMin = 2e-5
    , _beta                = 0.01
    , _delta               = 0.005
    , _gamma               = 0.01
    , _epsilon             = 0.25
    , _exploration         = 1.0
    , _learnRandomAbove    = 0.995
    , _zeta                = 0.03
    , _xi                  = 0.005
    }

instance Default (Parameters DecaySetup) where
  def =
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-7) 0.5 50000 -- 5e-4
      , _alphaRhoMin      = NoDecay
      , _beta             = ExponentialDecay (Just 1e-4) 0.5 150000
      , _delta            = ExponentialDecay (Just 5e-4) 0.5 150000
      , _gamma            = ExponentialDecay (Just 1e-3) 0.5 150000 -- 1e-3
      , _zeta             = ExponentialDecay (Just 0) 0.5 150000
      , _xi               = NoDecay
      -- Exploration
      , _epsilon          = [NoDecay]
      , _exploration      = ExponentialDecay (Just 0.01) 0.50 100000
      , _learnRandomAbove = NoDecay
      }
