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
decaySetupParameters (Parameters decAlp decAlpANN decBet decBetANN decDel decDelANN decGa decGaANN decEps _ decExp decRand decZeta decXi _) period (Parameters alp alpANN bet betANN del delANN ga gaANN eps expStrat exp rand zeta xi disable) =
  Parameters
    { _alpha = decaySetup decAlp period alp
    , _alphaANN = decaySetup decAlpANN period alpANN
    , _beta = decaySetup decBet period bet
    , _betaANN = decaySetup decBetANN period betANN
    , _delta = decaySetup decDel period del
    , _deltaANN = decaySetup decDelANN period delANN
    , _gamma = decaySetup decGa period ga
    , _gammaANN = decaySetup decGaANN period gaANN
    , _epsilon = (\de e -> decaySetup de period e) <$> decEps <*> eps
    , _explorationStrategy = expStrat
    , _exploration = decaySetup decExp period exp
    , _learnRandomAbove = decaySetup decRand period rand
    , _zeta = decaySetup decZeta period zeta
    , _xi = decaySetup decXi period xi
    , _disableAllLearning = disable
    }


-- | Exponential Decay with possible minimum values. All ANN parameters, the minimum learning rate for random actions,
-- and zeta are not decayed!
exponentialDecayParameters :: Maybe (Parameters Float) -> DecayRate -> DecaySteps -> Decay
exponentialDecayParameters Nothing rate steps t p = exponentialDecayParameters (Just (Parameters 0 0 0 0 0 0 0 0 0 EpsilonGreedy  0 0 0 0 False)) rate steps t p
exponentialDecayParameters (Just (Parameters mAlp mAlpANN mBet mBetANN mDel mDelANN mGa mGaANN mEps _ mExp mRand mZeta mXi _)) rate steps t (Parameters alp alpANN bet betANN del delANN ga gaANN eps expStrat exp rand zeta xi disable) =
  Parameters
    (max mAlp $ decay * alp)
    (max mAlpANN $ decay * alpANN)
    (max mBet $ decay * bet)
    (max mBetANN $ decay * betANN)
    (max mDel $ decay * del)
    (max mDelANN $ decay * delANN)
    (max mGa $ decay * ga)
    (max mGaANN $ decay * gaANN)
    (max <$> mEps <*> ((decay *) <$> eps))
    expStrat
    (max mExp $ decay * exp)
    (max mRand rand) -- no decay
    (max mZeta $ decay * zeta) -- no decay
    (max mXi $ decay * xi)
    disable
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
