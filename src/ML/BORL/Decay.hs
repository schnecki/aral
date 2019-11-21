module ML.BORL.Decay where

import           Control.Lens
import           Data.Maybe         (fromMaybe)

import           ML.BORL.Parameters
import           ML.BORL.Types


type Decay = Period -> Parameters -> Parameters -- ^ Function specifying the decay of the parameters at time t.

type DecayRate = Double
type DecaySteps = Integer


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


-- | Exponential Decay with possible minimum values. All ANN parameters, the minimum learning rate for random actions,
-- and zeta are not decayed!
exponentialDecayParameters :: Maybe Parameters -> DecayRate -> DecaySteps -> Decay
exponentialDecayParameters Nothing rate steps t p = exponentialDecayParameters (Just (Parameters 0 0 0 0 0 0 0 0 0 0 0 0 0 False)) rate steps t p
exponentialDecayParameters (Just (Parameters mAlp mAlpANN mBet mBetANN mDel mDelANN mGa mGaANN mEps mExp mRand mZeta mXi _)) rate steps t (Parameters alp alpANN bet betANN del delANN ga gaANN eps exp rand zeta xi disable) =
  Parameters
    (max mAlp $ decay * alp)
    (max mAlpANN alpANN)        -- no decay
    (max mBet $ decay * bet)
    (max mBetANN  betANN)       -- no decay
    (max mDel $ decay * del)
    (max mDelANN delANN)        -- no decay
    (max mGa $ decay * ga)
    (max mGaANN gaANN)          -- no decay
    (max mEps $ decay * eps)
    (max mExp $ decay * exp)
    (max mRand rand)            -- no decay
    (max mZeta zeta)  -- no decay
    (max mXi $ decay * xi)
    disable
  where
    decay = rate ** (fromIntegral t / fromIntegral steps)


exponentialDecayParametersValue :: ASetter Parameters Parameters Double Double -> Maybe Double -> DecayRate -> DecaySteps -> Decay
exponentialDecayParametersValue accessor mMin rate steps t = over accessor (\v -> max (fromMaybe 0 mMin) $ decay * v)
  where decay = rate ** (fromIntegral t / fromIntegral steps)


exponentialDecayValue :: Maybe Double -> DecayRate -> DecaySteps -> Period -> Double -> Double
exponentialDecayValue mMin rate steps t v = max (fromMaybe 0 mMin) (decay * v)
  where decay = rate ** (fromIntegral t / fromIntegral steps)


