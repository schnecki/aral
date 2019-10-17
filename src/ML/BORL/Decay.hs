module ML.BORL.Decay where

import           Control.Lens
import           Data.Maybe         (fromMaybe)

import           ML.BORL.Parameters
import           ML.BORL.Types


type Decay = Period -> Parameters -> Parameters -- ^ Function specifying the decay of the parameters at time t.

type DecayRate = Double
type DecaySteps = Integer

-- | Exponential Decay with possible minimum values. All ANN parameters and the minimum learning rate for random actions
-- are not decayed!
exponentialDecay :: Maybe Parameters -> DecayRate -> DecaySteps -> Decay
exponentialDecay Nothing rate steps t p = exponentialDecay (Just (Parameters 0 0 0 0 0 0 0 0 0 0 0 0 0 False)) rate steps t p
exponentialDecay (Just (Parameters mAlp mAlpANN mBet mBetANN mDel mDelANN mGa mGaANN mEps mExp mRand mZeta mXi _)) rate steps t (Parameters alp alpANN bet betANN del delANN ga gaANN eps exp rand zeta xi disable) =
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
    (max mZeta $ decay * zeta)
    (max mXi $ decay * xi)
    disable
  where
    decay = rate ** (fromIntegral t / fromIntegral steps)


exponentialDecayValue :: ASetter Parameters Parameters Double Double -> Maybe Double -> DecayRate -> DecaySteps -> Decay
exponentialDecayValue alpha mMin rate steps t = over alpha (\v -> max (fromMaybe 0 mMin) $ decay * v)
  where decay = rate ** (fromIntegral t / fromIntegral steps)


