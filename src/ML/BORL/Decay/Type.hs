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
  | ExponentialDecay { _decayMinimum :: Maybe Double
                     , _decayExpRate :: Double
                     , _decyaSteps   :: Integer }
  deriving (Generic, NFData, Serialize)
makeLenses ''DecaySetup


-- data MinimumValues = MinimumValues
--   { _minAlpha       :: Double
--   , _minAlphaANN    :: Double
--   , _minBeta        :: Double
--   , _minBetaANN     :: Double
--   , _minDelta       :: Double
--   , _minDeltaANN    :: Double
--   , _minGamma       :: Double
--   , _minGammaANN    :: Double
--   , _minEpsilon     :: Double
--   , _minExploration :: Double
--   , _minXi          :: Double
--   }   deriving (Generic, NFData, Serialize)
-- makeLenses ''MinimumValues
