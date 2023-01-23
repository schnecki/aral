{-# LANGUAGE RecordWildCards #-}
module ML.ARAL.NeuralNetwork.AdamW
    ( AdamW (..)
    , mkAdamW
    , adamW
    ) where

import qualified Torch
import qualified Torch.Optim        as Torch

import           ML.ARAL.Decay.Type

-- | State representation for Adam Optimizer
data AdamW = AdamW
  { nu          :: Float
  , beta1       :: Float          -- 1st moment forgetting factor
  , beta2       :: Float          -- 2nd moment forgetting factor
  , m1          :: [Torch.Tensor] -- 1st moment
  , m2          :: [Torch.Tensor] -- 2nd moment
  , iter        :: Int            -- iteration
  , l2          :: Double         -- l2
  , weightDecay :: Double
  }
  deriving (Show)

mkAdamW ::
  Float ->
  Float ->
  [Torch.Parameter] ->
  Double ->
  Double ->
  AdamW
mkAdamW beta1 beta2 parameters l2 weightDecay =
  AdamW
    1
    beta1
    beta2
    (initZeros <$> parameters)
    (initZeros <$> parameters)
    0
    l2
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
adamW lr (Torch.Gradients gradients) parameters AdamW {..} = (parameters', AdamW nu beta1 beta2 m1' m2' (iter + 1) l2 weightDecay)
    -- decaying averages of 1st & 2nd moments
  where
    f1 m1 dp = Torch.mulScalar beta1 m1 + Torch.mulScalar (1 - beta1) dp
    f2 m2 dp = Torch.mulScalar beta2 m2 + Torch.mulScalar (1 - beta2) (dp * dp)
    gradients'
      | l2 == 0 = gradients
      | otherwise = zipWith Torch.add gradients (map (Torch.mulScalar l2) parameters)
    m1' = zipWith f1 m1 gradients'
    m2' = zipWith f2 m2 gradients'
    -- bias adjustment
    a beta = Torch.divScalar (1 - beta ^ (iter + 1))
    a1 = fmap (a beta1) m1'
    a2 = fmap (a beta2) m2'
    -- parameter update
    eps = 1e-8 -- 1e-37
    update prevParam a1' a2'
      | weightDecay == 0 = prevParam - Torch.mulScalar nu (lr * a1' / (Torch.sqrt a2' + eps))
      | otherwise = prevParam - Torch.mulScalar nu (lr * a1' / (Torch.sqrt a2' + eps) + Torch.mulScalar weightDecay prevParam)
    parameters' = zipWith3 update parameters a1 a2

instance Torch.Optimizer AdamW where
  step = adamW


