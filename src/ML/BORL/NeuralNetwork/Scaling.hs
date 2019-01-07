{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.NeuralNetwork.Scaling where

import           ML.BORL.Types

import           Control.DeepSeq
import           Control.Lens
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
  } deriving (Show,NFData,Generic)
makeLenses ''ScalingNetOutParameters


scaleValue :: (MinValue,MaxValue) -> Double -> Double
scaleValue = const id -- scaleNegPosOne

unscaleValue :: (MinValue,MaxValue) -> Double -> Double
unscaleValue = const id -- unscaleNegPosOne

scaleZeroOneValue :: (MinValue,MaxValue) -> Double -> Double
scaleZeroOneValue (mn,mx) val = (val - mn) / (mx-mn)

unscaleZeroOneValue :: (MinValue,MaxValue) -> Double -> Double
unscaleZeroOneValue (mn,mx) val = val * (mx-mn) + mn


scaleNegPosOne :: (MinValue,MaxValue) -> Double -> Double
scaleNegPosOne (mn,mx) val = 2 * (val - mn) / (mx-mn) - 1

unscaleNegPosOne :: (MinValue,MaxValue) -> Double -> Double
unscaleNegPosOne (mn,mx) val = (val + 1) / 2 * (mx-mn) + mn


