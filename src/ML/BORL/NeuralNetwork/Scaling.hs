{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.NeuralNetwork.Scaling where

import           Control.DeepSeq
import           Control.Lens
import           GHC.Generics

data ScalingNetOutParameters = ScalingNetOutParameters
  { _scaleMaxVValue  :: Double
  , _scaleMinVValue  :: Double
  , _scaleMaxWValue  :: Double
  , _scaleMinWValue  :: Double
  , _scaleMaxR0Value :: Double
  , _scaleMinR0Value :: Double
  , _scaleMaxR1Value :: Double
  , _scaleMinR1Value :: Double
  } deriving (Show,NFData,Generic)
makeLenses ''ScalingNetOutParameters


type MaxValue = Double
type MinValue = Double


scaleValue :: (MinValue,MaxValue) -> Double -> Double
scaleValue = scaleNegPosOne

unscaleValue :: (MinValue,MaxValue) -> Double -> Double
unscaleValue = unscaleNegPosOne

scaleZeroOneValue :: (MinValue,MaxValue) -> Double -> Double
scaleZeroOneValue (mn,mx) val = (val - mn) / (mx-mn)

unscaleZeroOneValue :: (MinValue,MaxValue) -> Double -> Double
unscaleZeroOneValue (mn,mx) val = val * (mx-mn) + mn


scaleNegPosOne :: (MinValue,MaxValue) -> Double -> Double
scaleNegPosOne (mn,mx) val = 2 * (val - mn) / (mx-mn) - 1

unscaleNegPosOne :: (MinValue,MaxValue) -> Double -> Double
unscaleNegPosOne (mn,mx) val = (val + 1) / 2 * (mx-mn) + mn


