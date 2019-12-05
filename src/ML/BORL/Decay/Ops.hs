{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes     #-}
module ML.BORL.Decay.Ops where

import           Control.DeepSeq
import           Control.Lens
import           Data.Maybe         (fromMaybe)
import           Data.Serialize
import           GHC.Generics

import           ML.BORL.Decay.Type
import           ML.BORL.Parameters
import           ML.BORL.Types

type Decay = Period -> Parameters Double -> Parameters Double -- ^ Function specifying the decay of the parameters at time t.

decaySetup :: DecaySetup -> Period -> Value -> DecayedValue
decaySetup NoDecay                            = const id
decaySetup (ExponentialDecay mMin rate steps) = exponentialDecayValue mMin rate steps

-- | Override the decay of specified parameters.
overrideDecayParameters ::
     Period
  -> [(ASetter ParameterInitValues ParameterDecayedValues Double Double, Getting Double ParameterInitValues Double, DecayRate, DecaySteps, MinimumValue)]
  -> ParameterInitValues
  -> Parameters Double
  -> ParameterDecayedValues
overrideDecayParameters t xs params0 params = foldl (\p (setter, getter, rate, steps, minVal) -> set setter (decayedVal minVal rate steps (view getter params0)) p) params xs
  where
    decay rate steps = rate ** (fromIntegral t / fromIntegral steps)
    decayedVal minVal rate steps v = max minVal (v * decay rate steps)

-- overrideDecayParameters :: forall (f :: * -> *) . Functor f => Period -> [((Double -> f Double) -> Parameters -> f Parameters, DecayRate, DecaySteps, MinimumValue)] -> ParametersAtT0 -> Parameters -> Parameters
-- overrideDecayParameters t xs params0 params = foldl (\p (f, rate, steps, minVal) -> set f (decayedVal minVal rate steps (view f params0)) p) params xs
--   where
--     decay rate steps = rate ** (fromIntegral t / fromIntegral steps)
--     decayedVal minVal rate steps v = max minVal (v * decay rate steps)


-- | Exponential Decay with possible minimum values. All ANN parameters, the minimum learning rate for random actions,
-- and zeta are not decayed!
exponentialDecayParameters :: Maybe (Parameters Double) -> DecayRate -> DecaySteps -> Decay
exponentialDecayParameters Nothing rate steps t p = exponentialDecayParameters (Just (Parameters 0 0 0 0 0 0 0 0 0 0 0 0 0 False)) rate steps t p
exponentialDecayParameters (Just (Parameters mAlp mAlpANN mBet mBetANN mDel mDelANN mGa mGaANN mEps mExp mRand mZeta mXi _)) rate steps t (Parameters alp alpANN bet betANN del delANN ga gaANN eps exp rand zeta xi disable) =
  Parameters
    (max mAlp $ decay * alp)
    (max mAlpANN $ decay * alpANN)
    (max mBet $ decay * bet)
    (max mBetANN $ decay * betANN)
    (max mDel $ decay * del)
    (max mDelANN $ decay * delANN)
    (max mGa $ decay * ga)
    (max mGaANN $ decay * gaANN)
    (max mEps $ decay * eps)
    (max mExp $ decay * exp)
    (max mRand rand) -- no decay
    (max mZeta $ decay * zeta) -- no decay
    (max mXi $ decay * xi)
    disable
  where
    decay = rate ** (fromIntegral t / fromIntegral steps)


exponentialDecayParametersValue :: ASetter (Parameters Double) (Parameters Double) Double Double -> Maybe Double -> DecayRate -> DecaySteps -> Decay
exponentialDecayParametersValue accessor mMin rate steps t = over accessor (\v -> max (fromMaybe 0 mMin) $ decay * v)
  where decay = rate ** (fromIntegral t / fromIntegral steps)


exponentialDecayValue :: Maybe Double -> DecayRate -> DecaySteps -> Period -> Double -> Double
exponentialDecayValue mMin rate steps t v = max (fromMaybe 0 mMin) (decay * v)
  where decay = rate ** (fromIntegral t / fromIntegral steps)


