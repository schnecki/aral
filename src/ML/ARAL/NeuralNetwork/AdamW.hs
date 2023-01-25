{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE RecordWildCards #-}
module ML.ARAL.NeuralNetwork.AdamW
    ( AdamW (..)
    , mkAdamW
    , adamW
    ) where

import           Control.DeepSeq
import           Data.List                 (zipWith4)
import           Data.Serialize
import           GHC.Generics
import qualified Torch
import qualified Torch.Functional.Internal as Torch (powScalar)
import qualified Torch.Optim               as Torch

import           ML.ARAL.Decay.Type

-- | State representation for Adam Optimizer
data AdamW = AdamW
  { nu          :: !Double         -- ^ Learning rate
  , beta1       :: !Double         -- ^ 1st moment forgetting factor
  , beta2       :: !Double         -- ^ 2nd moment forgetting factor
  , m1          :: ![Torch.Tensor] -- ^ 1st moment
  , m2          :: ![Torch.Tensor] -- ^ 2nd moment
  , iter        :: !Int            -- ^ iteration
  , weightDecay :: !Double         -- ^ weight decay
  }
  deriving (Show, Generic)

instance NFData AdamW where
  rnf (AdamW _ _ _ mm1 mm2 _ _) = map (\(!_) -> ()) mm1 `seq` map (\(!_) -> ()) mm2 `seq` ()


mkAdamW ::
  Double ->
  Double ->
  [Torch.Parameter] ->
  Double ->
  AdamW
mkAdamW beta1 beta2 parameters weightDecay =
  AdamW
    1
    beta1
    beta2
    (initZeros <$> parameters)
    (initZeros <$> parameters)
    0
    weightDecay
  where
    initZeros = Torch.zerosLike . Torch.toDependent

-- | AdamW step
adamW ::
  -- | learning rate
  Torch.LearningRate ->
  -- | model parameter gradients
  Torch.Gradients ->
  -- | model parameters
  [Torch.Tensor] ->
  -- | adam parameters - beta1, beta2, moments, iteration
  AdamW ->
  -- | returns new parameters + updated adam parameters
  ([Torch.Tensor], AdamW)
adamW lr (Torch.Gradients gradients) parameters AdamW {..} = (parameters', AdamW nu beta1 beta2 m1' m2' (iter + 1) weightDecay)
  where
    -- decaying averages of 1st & 2nd moments
    f1 m1 dp = Torch.mulScalar beta1 m1 + Torch.mulScalar (1 - beta1) dp
    f2 m2 dp = Torch.mulScalar beta2 m2 + Torch.mulScalar (1 - beta2) (Torch.powScalar dp 2)
    -- calculate step size
    biasCorrection1 = 1 - beta1 ^ (iter + 1)
    biasCorrection2 = 1 - beta2 ^ (iter + 1)
    stepSizeCorrection = sqrt biasCorrection2 / biasCorrection1
    -- stepSize = Torch.mulScalar stepSizeCorrection lr
    -- add l2
    -- l2 :: Double
    -- l2 = 1e-3
    -- gradients' -- Note that normal l2 is not effective, see `Decoupled Weight Decay Regularization`: https://arxiv.org/abs/1711.05101
    --   | l2 == 0 = gradients
    --   | otherwise = zipWith Torch.add gradients (map (Torch.mulScalar (l2 * realToFrac (1 - stepSizeCorrection))) parameters)
    m1' = zipWith f1 m1 gradients -- gradients'
    m2' = zipWith f2 m2 gradients -- gradients'
    -- bias adjustment
    eps = 1e-8 -- 1e-37
    --
    --
    -- 1. Original implementation
    -- a beta = Torch.divScalar (1 - beta ^ (iter + 1))
    -- a1 = fmap (a beta1) m1'
    -- a2 = fmap (a beta2) m2'
    -- update prevParam a1' a2'    -- parameter update
    --   | weightDecay == 0 = prevParam - Torch.mulScalar nu (lr * a1' / (Torch.sqrt a2' + eps))
    --   | otherwise = prevParam - Torch.mulScalar nu (lr * a1' / (Torch.sqrt a2' + eps) + Torch.mulScalar weightDecay prevParam)
    --
    --
    -- 2. Implementation with stepsize for weigth decay (1)
    update isLastLayer prevParam mm1 mm2
      --  | weightDecay == 0 = prevParam - Torch.mulScalar nu ((lr `Torch.mul` a1') / (Torch.sqrt a2' + eps))
      --  | otherwise = prevParam - Torch.mulScalar nu ((lr `Torch.mul` a1') / (Torch.sqrt a2' + eps) + Torch.mulScalar weightDecay (stepSize `Torch.mul` prevParam))
      --  | otherwise = prevParam - Torch.mulScalar nu ((lr `Torch.mul` a1') / (Torch.sqrt a2' + eps)) - Torch.mulScalar (nu * weightDecay) (stepSize `Torch.mul` prevParam)
      | isLastLayer || weightDecay == 0 = prevParam - Torch.mulScalar (nu * stepSizeCorrection) (lr `Torch.mul` mm1 `Torch.div` (Torch.sqrt mm2 + eps))
      | otherwise = prevParam - Torch.mulScalar (nu * stepSizeCorrection) (lr `Torch.mul` mm1 `Torch.div` (Torch.sqrt mm2 + eps))
                              -- - Torch.mulScalar (nu * stepSizeCorrection * weightDecay) prevParam
                              -- - Torch.mulScalar (nu * stepSizeCorrection * weightDecay) (Torch.sign prevParam `Torch.mul` Torch.powScalar prevParam 2)
                              - Torch.mulScalar (nu * stepSizeCorrection * weightDecay) (Torch.powScalar prevParam 5) -- values > 1 are penalized much more than values < 1.
    -- parameters' = zipWith3 update parameters a1 a2
    lastLayerFlag = replicate (length parameters - 1) False ++ [True]
    parameters' = zipWith4 update lastLayerFlag parameters m1' m2'

    --
    --
    -- 3. Implementation with step size for weightDecay (2) (at least `epsilon` is different by being scaled by `sqrt(beta2^t)` in this version 2.)
    -- biasCorrection1 = 1 - beta1 ^ (iter + 1)
    -- biasCorrection2 = 1 - beta2 ^ (iter + 1)
    -- stepSize = Torch.mulScalar (sqrt biasCorrection2 / biasCorrection1) lr
    -- step = zipWith (\mm1 mm2 -> mm1 / (Torch.sqrt mm2 + eps) ) m1' m2'
    -- update :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor
    -- update prevParam step'
    --   | weightDecay == 0 = prevParam - Torch.mulScalar nu (stepSize `Torch.mul` step')
    --   | otherwise = prevParam - Torch.mulScalar nu (stepSize `Torch.mul` (step' + Torch.mulScalar weightDecay prevParam))
    -- parameters' = zipWith update parameters step


instance Torch.Optimizer AdamW where
  step = adamW


