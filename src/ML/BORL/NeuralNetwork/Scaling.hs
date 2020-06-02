{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.NeuralNetwork.Scaling where

import           ML.BORL.Types

import           Control.DeepSeq
import           Control.Lens
import           Data.Serialize
import           GHC.Generics

data ScalingAlgorithm
  = ScaleMinMax    -- ^ Scale using min-max normalisation.
  | ScaleLog Float -- ^ First apply a logarithm and then scale using min-max. Parameters specifies shift. Must be >= 1! A value of around 100 or 1000 usually works well. The smaller the number the
                   -- higher the spread for small values and thus the smaller the spread for large values.
  deriving (Eq, Ord, Show, NFData, Generic, Serialize)

isScaleLog :: ScalingAlgorithm -> Bool
isScaleLog ScaleLog{} = True
isScaleLog _          = True


data ScalingNetOutParameters = ScalingNetOutParameters
  { _scaleMinVValue  :: !(MinValue Float)
  , _scaleMaxVValue  :: !(MaxValue Float)
  , _scaleMinWValue  :: !(MinValue Float)
  , _scaleMaxWValue  :: !(MaxValue Float)
  , _scaleMinR0Value :: !(MinValue Float)
  , _scaleMaxR0Value :: !(MaxValue Float)
  , _scaleMinR1Value :: !(MinValue Float)
  , _scaleMaxR1Value :: !(MaxValue Float)
  } deriving (Eq, Ord, Show, NFData, Generic, Serialize)
makeLenses ''ScalingNetOutParameters


multiplyScale :: Float -> ScalingNetOutParameters -> ScalingNetOutParameters
multiplyScale v (ScalingNetOutParameters minV maxV minW maxW minR0 maxR0 minR1 maxR1) =
  ScalingNetOutParameters (v * minV) (v * maxV) (v * minW) (v * maxW) (v * minR0) (v * maxR0) (v * minR1) (v * maxR1)

-- | Scale a value using the given algorithm.
scaleValue :: (Floating n, Ord n) => ScalingAlgorithm -> Maybe (MinValue n, MaxValue n) -> n -> n
scaleValue ScaleMinMax    = maybe id scaleMinMax
scaleValue (ScaleLog shift) = maybe (error "scaling with ScaleLog requries minimum and maximum values!") (scaleLog shift)

-- | Unscale a value using the given algorithm.
unscaleValue :: (Floating n) => ScalingAlgorithm -> Maybe (MinValue n, MaxValue n) -> n -> n
unscaleValue ScaleMinMax = maybe id unscaleMinMax
unscaleValue (ScaleLog shift) = maybe (error "scaling with ScaleLog requries minimum and maximum values!") (unscaleLog shift)

scaleZeroOneValue :: (Fractional n) => (MinValue n, MaxValue n) -> n -> n
scaleZeroOneValue (mn,mx) val = (val - mn) / (mx-mn)

unscaleZeroOneValue :: (Fractional n) => (MinValue n, MaxValue n) -> n -> n
unscaleZeroOneValue (mn,mx) val = val * (mx-mn) + mn

-- | Scale using min-max normalization.
scaleMinMax :: (Fractional n) => (MinValue n, MaxValue n) -> n -> n
scaleMinMax (mn,mx) val = 2 * (val - mn) / (mx-mn) - 1

unscaleMinMax :: (Fractional n) => (MinValue n, MaxValue n) -> n -> n
unscaleMinMax (mn,mx) val = (val + 1) / 2 * (mx-mn) + mn


-- | Scale the value applying @log@ first. Note that the actual value is clipped by the minimum value, the maximum is open.
scaleLog :: (Floating n, Ord n) => Float -> (MinValue n, MaxValue n) -> n -> n
scaleLog shift (mn, mx) val = scaleMinMax (log shift', appLog mx) $ appLog $ max mn val
  where appLog x = log (x - mn + shift')
        shift' = realToFrac shift

unscaleLog :: (Floating n) => Float -> (MinValue n, MaxValue n) -> n -> n
unscaleLog shift (mn, mx) val = appExp $ unscaleMinMax (log shift', appLog mx) val
  where appExp x = exp x + mn - shift'
        appLog x = log (x - mn + shift')
        shift' = realToFrac shift
