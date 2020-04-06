{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes     #-}
module ML.BORL.Decay.Ops where

import           Control.Lens
import           Data.Maybe          (fromMaybe)

import           ML.BORL.Decay.Type
import           ML.BORL.Exploration
import           ML.BORL.Parameters
import           ML.BORL.Types

type Decay = Period -> Parameters Float -> Parameters Float -- ^ Function specifying the decay of the parameters at time t.

decaySetup :: DecaySetup -> Period -> Value -> DecayedValue
decaySetup NoDecay                               = const id
decaySetup (ExponentialDecay mMin rate steps)    = exponentialDecayValue mMin rate steps
decaySetup (ExponentialIncrease mMin rate steps) = exponentialIncreaseValue mMin rate steps
decaySetup (LinearIncrease mD k)                 = linearIncreaseValue mD k
decaySetup (StepWiseIncrease mD k step)          = stepWiseIncreaseValue mD k step

-- | Override the decay of specified parameters.
overrideDecayParameters ::
     Period
  -> [(ASetter ParameterInitValues ParameterDecayedValues Float Float, Getting Float ParameterInitValues Float, DecayRate, DecaySteps, MinimumValue)]
  -> ParameterInitValues
  -> Parameters Float
  -> ParameterDecayedValues
overrideDecayParameters t xs params0 params = foldl (\p (setter, getter, rate, steps, minVal) -> set setter (decayedVal minVal rate steps (view getter params0)) p) params xs
  where
    decay rate steps = rate ** (fromIntegral t / fromIntegral steps)
    decayedVal minVal rate steps v = max minVal (v * decay rate steps)

-- overrideDecayParameters :: forall (f :: * -> *) . Functor f => Period -> [((Float -> f Float) -> Parameters -> f Parameters, DecayRate, DecaySteps, MinimumValue)] -> ParametersAtT0 -> Parameters -> Parameters
-- overrideDecayParameters t xs params0 params = foldl (\p (f, rate, steps, minVal) -> set f (decayedVal minVal rate steps (view f params0)) p) params xs
--   where
--     decay rate steps = rate ** (fromIntegral t / fromIntegral steps)
--     decayedVal minVal rate steps v = max minVal (v * decay rate steps)

decaySetupParameters :: Parameters DecaySetup -> Decay
decaySetupParameters (Parameters decAlp decAlpRhoMin decBet decDel decGa decEps decExp decRand decZeta decXi) period (Parameters alp alpRhoMin bet del ga eps exp rand zeta xi) =
  Parameters
    { _alpha = decaySetup decAlp period alp
    , _alphaRhoMin = decaySetup decAlpRhoMin period alpRhoMin
    , _beta = decaySetup decBet period bet
    , _delta = decaySetup decDel period del
    , _gamma = decaySetup decGa period ga
    , _epsilon = (\de e -> decaySetup de period e) <$> decEps <*> eps
    , _exploration = decaySetup decExp period exp
    , _learnRandomAbove = decaySetup decRand period rand
    , _zeta = decaySetup decZeta period zeta
    , _xi = decaySetup decXi period xi
    }


-- | Exponential Decay with possible minimum values. All ANN parameters, the minimum learning rate for random actions,
-- and zeta are not decayed!
exponentialDecayParameters :: Maybe (Parameters Float) -> DecayRate -> DecaySteps -> Decay
exponentialDecayParameters Nothing rate steps t p = exponentialDecayParameters (Just (Parameters 0 0 0 0 0 0 0 0 0 0)) rate steps t p
exponentialDecayParameters (Just (Parameters mAlp mAlpRhoMin mBet mDel mGa mEps mExp mRand mZeta mXi)) rate steps t (Parameters alp alpRhoMin bet del ga eps exp rand zeta xi) =
  Parameters
    (max mAlp $ decay * alp)
    (max mAlpRhoMin $ decay * alpRhoMin)
    (max mBet $ decay * bet)
    (max mDel $ decay * del)
    (max mGa $ decay * ga)
    (max <$> mEps <*> ((decay *) <$> eps))
    (max mExp $ decay * exp)
    (max mRand rand) -- no decay
    (max mZeta $ decay * zeta) -- no decay
    (max mXi $ decay * xi)
  where
    decay = rate ** (fromIntegral t / fromIntegral steps)


exponentialDecayParametersValue :: ASetter (Parameters Float) (Parameters Float) Float Float -> Maybe Float -> DecayRate -> DecaySteps -> Decay
exponentialDecayParametersValue accessor mMin rate steps t = over accessor (\v -> max (fromMaybe 0 mMin) $ decay * v)
  where decay = rate ** (fromIntegral t / fromIntegral steps)


exponentialDecayValue :: Maybe Float -> DecayRate -> DecaySteps -> Period -> Float -> Float
exponentialDecayValue mMin rate steps t v = max (fromMaybe 0 mMin) (decay * v)
  where decay = rate ** (fromIntegral t / fromIntegral steps)

exponentialIncreaseValue :: Maybe Float -> DecayRate -> DecaySteps -> Period -> Float -> Float
exponentialIncreaseValue mMin rate steps t v = v - exponentialDecayValue ((v *) <$> mMin) rate steps t v


linearIncreaseValue :: Maybe Float -> Float -> Period -> Float -> Float
linearIncreaseValue mD k t v = min v $ fromIntegral t * k + fromMaybe 0 mD

stepWiseIncreaseValue :: Maybe Float -> Float -> Period -> Period -> Float -> Float
stepWiseIncreaseValue mD k step t = linearIncreaseValue mD k t'
  where
    t' = t - t `mod` step
