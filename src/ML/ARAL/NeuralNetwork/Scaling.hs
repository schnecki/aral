{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}
module ML.ARAL.NeuralNetwork.Scaling where


import           Control.DeepSeq
import           Control.Lens
import           Data.Serialize
import qualified Data.Vector.Storable as V
import           GHC.Generics
import           Numeric
import           Prelude              hiding (scaleDouble)

import           Grenade.Utils.Vector

import           ML.ARAL.Types


data ScalingAlgorithm
  = ScaleMinMax    -- ^ Scale using min-max normalisation.
  | ScaleLog Double -- ^ First apply a logarithm and then scale using min-max. Parameters specifies shift. Must be >= 1! A value of around 100 or 1000 usually works well. The smaller the number the
                   -- higher the spread for small values and thus the smaller the spread for large values.
  | ScaleSymlog    -- ^ See https://arxiv.org/pdf/2301.04104v1.pdf
  | ScaleClip (MinValue Double, MaxValue Double) ScalingAlgorithm
  deriving (Eq, Ord, Show, NFData, Generic, Serialize)

isScaleLog :: ScalingAlgorithm -> Bool
isScaleLog ScaleLog{} = True
isScaleLog _          = False


data ScalingNetOutParameters = ScalingNetOutParameters
  { _scaleMinVValue  :: !(MinValue Double)
  , _scaleMaxVValue  :: !(MaxValue Double)
  , _scaleMinWValue  :: !(MinValue Double)
  , _scaleMaxWValue  :: !(MaxValue Double)
  , _scaleMinR0Value :: !(MinValue Double)
  , _scaleMaxR0Value :: !(MaxValue Double)
  , _scaleMinR1Value :: !(MinValue Double)
  , _scaleMaxR1Value :: !(MaxValue Double)
  } deriving (Eq, Ord, Show, NFData, Generic, Serialize)
makeLenses ''ScalingNetOutParameters


multiplyScale :: Double -> ScalingNetOutParameters -> ScalingNetOutParameters
multiplyScale v (ScalingNetOutParameters minV maxV minW maxW minR0 maxR0 minR1 maxR1) =
  ScalingNetOutParameters (v * minV) (v * maxV) (v * minW) (v * maxW) (v * minR0) (v * maxR0) (v * minR1) (v * maxR1)

scaleValue :: ScalingAlgorithm -> Maybe (MinValue Double, MaxValue Double) -> Value -> Value
scaleValue alg minMax = mapValue (scaleDouble alg minMax)

scaleValues :: ScalingAlgorithm -> Maybe (MinValue Double, MaxValue Double) -> Values -> Values
scaleValues alg minMax = mapValues (mapVector $ scaleDouble alg minMax)

unscaleValue :: ScalingAlgorithm -> Maybe (MinValue Double, MaxValue Double) -> Value -> Value
unscaleValue alg minMax = mapValue (unscaleDouble alg minMax)

unscaleValues :: ScalingAlgorithm -> Maybe (MinValue Double, MaxValue Double) -> Values -> Values
unscaleValues alg minMax = mapValues (mapVector $ unscaleDouble alg minMax)


-- | Scale a value using the given algorithm.
scaleDouble :: ScalingAlgorithm -> Maybe (MinValue Double, MaxValue Double) -> Double -> Double
scaleDouble (ScaleClip (minVal, maxVal) scale) ms = max minVal . min maxVal . scaleDouble scale ms
scaleDouble ScaleMinMax ms                        = maybe id scaleMinMax ms
scaleDouble ScaleSymlog  _                        = symlog
scaleDouble (ScaleLog shift) ms                   = maybe (error "scaling with ScaleLog requries minimum and maximum values!") (scaleLog (realToFrac shift)) ms

-- | Unscale a value using the given algorithm.
unscaleDouble :: ScalingAlgorithm -> Maybe (MinValue Double, MaxValue Double) -> Double -> Double
unscaleDouble (ScaleClip (minVal, maxVal) scale) ms = max minVal . min maxVal . unscaleDouble scale ms
unscaleDouble ScaleMinMax ms                        = maybe id unscaleMinMax ms
unscaleDouble ScaleSymlog _                         = symexp
unscaleDouble (ScaleLog shift) ms                   = maybe (error "scaling with ScaleLog requries minimum and maximum values!") (unscaleLog (realToFrac shift)) ms

scaleZeroOneDouble :: (Fractional n) => (MinValue n, MaxValue n) -> n -> n
scaleZeroOneDouble (mn,mx) val = (val - mn) / (mx-mn)

unscaleZeroOneDouble :: (Fractional n) => (MinValue n, MaxValue n) -> n -> n
unscaleZeroOneDouble (mn,mx) val = val * (mx-mn) + mn

-- | Scale using min-max normalization.
scaleMinMax :: (Fractional n) => (MinValue n, MaxValue n) -> n -> n
scaleMinMax (mn,mx) val = 2 * (val - mn) / (mx-mn) - 1

unscaleMinMax :: (Fractional n) => (MinValue n, MaxValue n) -> n -> n
unscaleMinMax (mn,mx) val = (val + 1) / 2 * (mx-mn) + mn


symlog :: (Floating n) => n -> n
symlog x = signum x * log (abs x + 1) -- log1p (abs x) --  @'log1p' x@ computes @'log' (1 + x)@, but provides more precise results for small (absolute) values of @x@ if possible.

symexp :: (Floating n) => n -> n
symexp x = signum x * exp(abs x) - 1  -- @'expm1' x@ computes @'exp' x - 1@, but provides more precise results for small (absolute) values of @x@ if possible.


-- | Scale the value applying @log@ first. Note that the actual value is clipped by the minimum value, the maximum is open.
scaleLog :: (Floating n, Ord n) => n -> (MinValue n, MaxValue n) -> n -> n
scaleLog shift (mn, mx) val = scaleMinMax (log shift, appLog mx) $ appLog $ max mn val
  where appLog x = log (x - mn + shift)

-- | Unscale from a logarthmic scale.
unscaleLog :: (Floating n) => n -> (MinValue n, MaxValue n) -> n -> n
unscaleLog shift (mn, mx) val = appExp $ unscaleMinMax (log shift, appLog mx) val
  where appExp x = exp x + mn - shift
        appLog x = log (x - mn + shift)
