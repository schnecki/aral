module ML.BORL.Decay where

import           ML.BORL.Parameters
import           ML.BORL.Types

type Decay = Period -> Parameters -> Parameters -- ^ Function specifying the decay of the parameters at time t.

type DecayRate = Double
type DecaySteps = Integer

-- | Exponential Decay with possible minimum values.
exponentialDecay :: Maybe Parameters -> DecayRate -> DecaySteps -> Decay
exponentialDecay Nothing rate steps t p = exponentialDecay (Just (Parameters 0 0 0 0 0 0 0 0 0 False)) rate steps t p
exponentialDecay (Just (Parameters mAlp mBet mDel mGa mEps mExp mRand mZeta mXi _)) rate steps t (Parameters alp bet del ga eps exp rand zeta xi disable) =
  Parameters
    (max mAlp $ decay * alp)
    (max mBet $ decay * bet)
    (max mDel $ decay * del)
    (max mGa $ decay * ga)
    (max mEps $ decay * eps)
    (max mExp $ decay * exp)
    (max mRand $ decay * rand)
    (max mZeta $ decay * zeta)
    (max mXi $ decay * xi)
    disable
  where
    decay = rate ** (fromIntegral t / fromIntegral steps)
