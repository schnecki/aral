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


type DecayRate = Float
type DecaySteps = Integer
type Value = Float
type MinimumValue = Float
type DecayedValue = Float

data DecaySetup
  = NoDecay
  | ExponentialDecay                  -- ^ Exponentially decrease a given value to 0, maybe cut by the given minimum value. v * rate^(t/steps
      { _decayMinimum :: Maybe Float -- ^ Minimum value.
      , _decayExpRate :: Float       -- ^ Decay rate.
      , _decyaSteps   :: Integer      -- ^ Decay steps.
      }
  | ExponentialIncrease -- ^ Exponentially increase from 0 (or the specified minimum) to the provided value. Formula
                        -- used: v * (1-rate^(t/steps))
      { _increaseMinimum :: Maybe Float -- ^ Minimum value.
      , _increaseExpRate :: Float       -- ^ Exponential rate.
      , _increaseSteps   :: Integer      -- ^ Steps.
      }
  | LinearIncrease              -- ^ Linearly increase to a value from 0, maybe starting with a minimum. y = k * x + d
    { _increaseD :: Maybe Float -- ^ d [Default: 0]
    , _increaseK :: Float       -- ^ k
    }
  | StepWiseIncrease                -- ^ Linearly increase to a value from 0, maybe starting with a minimum. y = k * x + d but only updated to y every X step.
    { _increaseD    :: Maybe Float -- ^ d [Default: 0]
    , _increaseK    :: Float       -- ^ k
    , _increaseStep :: Int          -- ^ Step
    }

  deriving (Generic, Show, Eq, Ord, NFData, Serialize)
makeLenses ''DecaySetup

