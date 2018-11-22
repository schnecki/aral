{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.NeuralNetwork.Scaling where

import           Control.Lens


data ScalingNetOutParameters = ScalingNetOutParameters
  { _scaleMaxVValue  :: Double
  , _scaleMaxWValue  :: Double
  , _scaleMaxR0Value :: Double
  , _scaleMaxR1Value :: Double
  } deriving (Show)
makeLenses ''ScalingNetOutParameters


type MaxValue = Double


scaleValue :: MaxValue -> Double -> Double
scaleValue mx val = val / mx

unscaleValue :: MaxValue -> Double -> Double
unscaleValue mx val = val * mx


scaleNegPosOne :: MaxValue -> Double -> Double
scaleNegPosOne mx val = 2 * val / mx - 1

unscaleNegPosOne :: MaxValue -> Double -> Double
unscaleNegPosOne mx val = (val + 1) * mx / 2


