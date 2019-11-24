{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
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
type DecayedValue = Double

data DecaySetup  = ExponentialDecay
  { _decayMinimum :: Maybe Double
  , _decayExpRate :: Double
  , _decyaSteps   :: Integer
  } deriving (Generic, NFData, Serialize)


-- data MinimumValues = MinimumValues
--   { minAlpha :: Double
--   , minAlphaANN :: Double
--   , minBeta :: Double
--   , minBetaANN :: Double
--   , minDelta :: Double
--   , minDeltaANN :: Double
--   , minGamma :: Double
--   , minGammaANN :: Double
--   , minEpsilon :: Double
--   , min
--   }


