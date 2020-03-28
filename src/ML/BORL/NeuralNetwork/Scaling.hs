{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.NeuralNetwork.Scaling where

import           ML.BORL.Types

import           Control.DeepSeq
import           Control.Lens
import           Data.Serialize
import           GHC.Generics

data ScalingNetOutParameters = ScalingNetOutParameters
  { _scaleMinVValue  :: MinValue Float
  , _scaleMaxVValue  :: MaxValue Float
  , _scaleMinWValue  :: MinValue Float
  , _scaleMaxWValue  :: MaxValue Float
  , _scaleMinR0Value :: MinValue Float
  , _scaleMaxR0Value :: MaxValue Float
  , _scaleMinR1Value :: MinValue Float
  , _scaleMaxR1Value :: MaxValue Float
  } deriving (Eq, Ord, Show, NFData, Generic, Serialize)
makeLenses ''ScalingNetOutParameters


multiplyScale :: Float -> ScalingNetOutParameters -> ScalingNetOutParameters
multiplyScale v (ScalingNetOutParameters minV maxV minW maxW minR0 maxR0 minR1 maxR1) =
  ScalingNetOutParameters (v * minV) (v * maxV) (v * minW) (v * maxW) (v * minR0) (v * maxR0) (v * minR1) (v * maxR1)


scaleValue :: (Fractional n) => Maybe (MinValue n, MaxValue n) -> n -> n
scaleValue = maybe id scaleNegPosOne

unscaleValue :: (Fractional n) => Maybe (MinValue n, MaxValue n) -> n -> n
unscaleValue = maybe id unscaleNegPosOne

scaleZeroOneValue :: (Fractional n) => (MinValue n, MaxValue n) -> n -> n
scaleZeroOneValue (mn,mx) val = (val - mn) / (mx-mn)

unscaleZeroOneValue :: (Fractional n) => (MinValue n, MaxValue n) -> n -> n
unscaleZeroOneValue (mn,mx) val = val * (mx-mn) + mn

scaleNegPosOne :: (Fractional n) => (MinValue n, MaxValue n) -> n -> n
scaleNegPosOne (mn,mx) val = 2 * (val - mn) / (mx-mn) - 1

unscaleNegPosOne :: (Fractional n) => (MinValue n, MaxValue n) -> n -> n
unscaleNegPosOne (mn,mx) val = (val + 1) / 2 * (mx-mn) + mn


