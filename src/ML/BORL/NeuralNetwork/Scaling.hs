{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.NeuralNetwork.Scaling where


import           Control.DeepSeq
import           Control.Lens
import           Data.Serialize
import qualified Data.Vector.Storable as V
import           GHC.Generics
import           Prelude              hiding (scaleFloat)

import           ML.BORL.Types


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

scaleValue :: ScalingAlgorithm -> Maybe (MinValue Float, MaxValue Float) -> Value -> Value
scaleValue alg minMax = mapValue (scaleFloat alg minMax)

scaleValues :: ScalingAlgorithm -> Maybe (MinValue Float, MaxValue Float) -> Values -> Values
scaleValues alg minMax = mapValues (V.map $ scaleFloat alg minMax)

unscaleValue :: ScalingAlgorithm -> Maybe (MinValue Float, MaxValue Float) -> Value -> Value
unscaleValue alg minMax = mapValue (unscaleFloat alg minMax)

unscaleValues :: ScalingAlgorithm -> Maybe (MinValue Float, MaxValue Float) -> Values -> Values
unscaleValues alg minMax = mapValues (V.map $ unscaleFloat alg minMax)


-- | Scale a value using the given algorithm.
scaleFloat :: (Floating n, Ord n) => ScalingAlgorithm -> Maybe (MinValue n, MaxValue n) -> n -> n
scaleFloat ScaleMinMax    = maybe id scaleMinMax
scaleFloat (ScaleLog shift) = maybe (error "scaling with ScaleLog requries minimum and maximum values!") (scaleLog shift)

-- | Unscale a value using the given algorithm.
unscaleFloat :: (Floating n) => ScalingAlgorithm -> Maybe (MinValue n, MaxValue n) -> n -> n
unscaleFloat ScaleMinMax = maybe id unscaleMinMax
unscaleFloat (ScaleLog shift) = maybe (error "scaling with ScaleLog requries minimum and maximum values!") (unscaleLog shift)

scaleZeroOneFloat :: (Fractional n) => (MinValue n, MaxValue n) -> n -> n
scaleZeroOneFloat (mn,mx) val = (val - mn) / (mx-mn)

unscaleZeroOneFloat :: (Fractional n) => (MinValue n, MaxValue n) -> n -> n
unscaleZeroOneFloat (mn,mx) val = val * (mx-mn) + mn

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
