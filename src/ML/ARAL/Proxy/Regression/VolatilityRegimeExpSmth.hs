{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
module ML.ARAL.Proxy.Regression.VolatilityRegimeExpSmth
    ( RegimeDetection (..)
    , addValueToRegime
    , currentRegimeExp
    , Regime (..)
    ) where

import           Control.DeepSeq
import           Data.Default
import           Data.Serialize
import           GHC.Generics
import           Statistics.Sample.WelfordOnlineMeanVariance

data Regime = Low | High
-- data Regime = Rising | High | Falling | Low
   deriving (Eq, Ord, Enum, Bounded, NFData, Generic, Serialize, Show)

data RegimeDetection =
  RegimeDetection
    { regimeWelfordAll  :: !(WelfordExistingAggregate Double)               -- ^ Welford for values.
    , regimeExpSmthFast :: !Double
    , regimeExpSmthSlow :: !Double
    , regimeExpSmthSt   :: !Regime
    }
  deriving (Show, NFData, Generic, Serialize)

-- instance Show RegimeDetection where
--   show (RegimeDetection wel _ _ st)          = show st ++ " Welford: " ++ show wl

instance Default RegimeDetection where
  def = RegimeDetection WelfordExistingAggregateEmpty 0 0 Low


updateExp :: WelfordExistingAggregate Double -> (Double, Double, Regime) -> Double -> (Double, Double, Regime)
updateExp wel (fastExp, slowExp, curSt) x = (fastExp', slowExp', curSt')
  where
    (mean, _, variance) =
      case wel of
        WelfordExistingAggregate {}   -> finalize wel
        WelfordExistingAggregateEmpty -> (0, 0, 0)
    fastExp' = (1 - alphaFast) * fastExp + alphaFast * abs x
    slowExp' = (1 - alphaSlow) * slowExp + alphaSlow * abs x
    alphaFast = 0.01
    alphaSlow = 0.001
    border = mean + sqrt variance
    borderUp = 0.612 * border
    borderDown = 0.5 * border
    curSt' =
      case curSt of
        Low
          | fastExp' > borderUp && slowExp' > borderDown -> High
        High
          | fastExp' <= borderDown && slowExp' <= borderUp -> Low
        st -> st


addValueToRegime :: RegimeDetection -> Double -> RegimeDetection
addValueToRegime (RegimeDetection welAll expFast expSlow st) x = RegimeDetection welAll' expFast' expSlow' st'
  where
    (expFast', expSlow', st') = updateExp welAll' (expFast, expSlow, st) x
    welAll' = addValue welAll x


currentRegimeExp :: RegimeDetection -> Regime
currentRegimeExp (RegimeDetection _ _ _ st) = st
