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
  { _scaleMinVValue  :: MinValue
  , _scaleMaxVValue  :: MaxValue
  , _scaleMinWValue  :: MinValue
  , _scaleMaxWValue  :: MaxValue
  , _scaleMinR0Value :: MinValue
  , _scaleMaxR0Value :: MaxValue
  , _scaleMinR1Value :: MinValue
  , _scaleMaxR1Value :: MaxValue
  } deriving (Eq, Ord, Show, NFData, Generic, Serialize)
makeLenses ''ScalingNetOutParameters


multiplyScale :: Double -> ScalingNetOutParameters -> ScalingNetOutParameters
multiplyScale v (ScalingNetOutParameters minV maxV minW maxW minR0 maxR0 minR1 maxR1) =
  ScalingNetOutParameters (v * minV) (v * maxV) (v * minW) (v * maxW) (v * minR0) (v * maxR0) (v * minR1) (v * maxR1)


scaleValue :: Maybe (MinValue,MaxValue) -> Double -> Double
scaleValue = maybe id scaleNegPosOne

unscaleValue :: Maybe (MinValue,MaxValue) -> Double -> Double
unscaleValue = maybe id unscaleNegPosOne

scaleZeroOneValue :: (MinValue,MaxValue) -> Double -> Double
scaleZeroOneValue (mn,mx) val = (val - mn) / (mx-mn)

unscaleZeroOneValue :: (MinValue,MaxValue) -> Double -> Double
unscaleZeroOneValue (mn,mx) val = val * (mx-mn) + mn


scaleNegPosOne :: (MinValue,MaxValue) -> Double -> Double
scaleNegPosOne (mn,mx) val = 2 * (val - mn) / (mx-mn) - 1

unscaleNegPosOne :: (MinValue,MaxValue) -> Double -> Double
unscaleNegPosOne (mn,mx) val = (val + 1) / 2 * (mx-mn) + mn


