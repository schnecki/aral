{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes     #-}
module ML.ARAL.Decay.Ops where

import           Control.Lens
import           Data.Maybe          (fromMaybe)

import           ML.ARAL.Decay.Type
import           ML.ARAL.Exploration
import           ML.ARAL.Parameters
import           ML.ARAL.Types

type Decay = Period -> Parameters Double -> Parameters Double -- ^ Function specifying the decay of the parameters at time t.


decaySetup :: DecaySetup -> Period -> InitialValue -> DecayedValue
decaySetup NoDecay                               = const id
decaySetup (ExponentialDecay mMin rate steps)    = exponentialDecayValue mMin rate steps
decaySetup (ExponentialIncrease mMin rate steps) = exponentialIncreaseValue mMin rate steps
decaySetup (LinearIncrease mD k)                 = linearIncreaseValue mD k
decaySetup (StepWiseIncrease mD k step)          = stepWiseIncreaseValue mD k step

-- | Exponentiall decay the paramters.
decaySettingParameters :: Parameters DecaySetup -> Decay
decaySettingParameters (Parameters decAlp decAlpRhoMin decBet decDel decGa decEps decExp decRand decZeta decXi) period (Parameters alp alpRhoMin bet del ga eps exp rand zeta xi) =
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


exponentialDecayValue :: Maybe Double -> DecayRate -> DecaySteps -> Period -> Double -> Double
exponentialDecayValue mMin rate steps t v = max (fromMaybe 0 mMin) (decay * v)
  where decay = rate ** (fromIntegral t / fromIntegral steps)

exponentialIncreaseValue :: Maybe Double -> DecayRate -> DecaySteps -> Period -> Double -> Double
exponentialIncreaseValue mMin rate steps t v = v - exponentialDecayValue ((v *) <$> mMin) rate steps t v


linearIncreaseValue :: Maybe Double -> Double -> Period -> Double -> Double
linearIncreaseValue mD k t v = min v $ fromIntegral t * k + fromMaybe 0 mD

stepWiseIncreaseValue :: Maybe Double -> Double -> Period -> Period -> Double -> Double
stepWiseIncreaseValue mD k step t = linearIncreaseValue mD k t'
  where
    t' = t - t `mod` step
