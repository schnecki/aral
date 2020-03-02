{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.Decay.Type where

import           Control.DeepSeq
import           Control.Lens
import           Data.Maybe      (fromMaybe)
import           Data.Serialize
import           GHC.Generics

import           ML.BORL.Types


type DecayRate = Double
type DecaySteps = Integer
type Value = Double
type MinimumValue = Double
type DecayedValue = Double

data DecaySetup
  = NoDecay
  | ExponentialDecay                  -- ^ Exponentially decrease a given value to 0, maybe cut by the given minimum value. v * rate^(t/steps
      { _decayMinimum :: Maybe Double -- ^ Minimum value.
      , _decayExpRate :: Double       -- ^ Decay rate.
      , _decyaSteps   :: Integer      -- ^ Decay steps.
      }
  | ExponentialIncrease -- ^ Exponentially increase from 0 (or the specified minimum) to the provided value. Formula
                        -- used: v * (1-rate^(t/steps))
      { _increaseMinimum :: Maybe Double -- ^ Minimum value.
      , _increaseExpRate :: Double       -- ^ Exponential rate.
      , _increaseSteps   :: Integer      -- ^ Steps.
      }
  | LinearIncrease              -- ^ Linearly increase to a value from 0, maybe starting with a minimum. y = k * x + d
    { _increaseD :: Maybe Double -- ^ d [Default: 0]
    , _increaseK :: Double       -- ^ k
    }
  | StepWiseIncrease                -- ^ Linearly increase to a value from 0, maybe starting with a minimum. y = k * x + d but only updated to y every X step.
    { _increaseD    :: Maybe Double -- ^ d [Default: 0]
    , _increaseK    :: Double       -- ^ k
    , _increaseStep :: Int          -- ^ Step
    }

  deriving (Generic, Show, Eq, Ord, NFData, Serialize)
makeLenses ''DecaySetup

