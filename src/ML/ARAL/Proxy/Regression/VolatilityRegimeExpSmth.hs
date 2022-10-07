{-# LANGUAGE DeriveAnyClass       #-}
{-# LANGUAGE DeriveGeneric        #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TupleSections        #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE TypeSynonymInstances #-}
module ML.ARAL.Proxy.Regression.VolatilityRegimeExpSmth
    ( RegimeDetection (..)
    , addValueToRegime
    , currentRegimeExp
    , State (..)
    ) where

import           Control.DeepSeq
import           Data.Default
import           Data.List                                   (foldl')
import qualified Data.Map.Strict                             as M
import           Data.Maybe                                  (fromMaybe)
import           Data.Monoid
import           Data.Serialize
import           Debug.Trace
import           GHC.Generics
import           Statistics.Sample.WelfordOnlineMeanVariance

data State = Low | High
-- data State = Rising | High | Falling | Low
   deriving (Eq, Ord, Enum, Bounded, NFData, Generic, Serialize, Show)

data RegimeDetection =
  RegimeDetection
    { regimeWelfordAll  :: WelfordExistingAggregate Double               -- ^ Welford for values.
    , regimeExpSmthFast :: Double
    , regimeExpSmthSlow :: Double
    , regimeExpSmthSt   :: State
    }
  deriving (Show, NFData, Generic, Serialize)

-- instance Show RegimeDetection where
--   show (RegimeDetection wel _ _ st)          = show st ++ " Welford: " ++ show wl

instance Default RegimeDetection where
  def = RegimeDetection WelfordExistingAggregateEmpty 0 0 Low


updateExp :: WelfordExistingAggregate Double -> (Double, Double, State) -> Double -> (Double, Double, State)
updateExp wel (fastExp, slowExp, curSt) x = (fastExp', slowExp', curSt')
  where
    (mean, _, variance) =
      case wel of
        WelfordExistingAggregate {}   -> finalize wel
        WelfordExistingAggregateEmpty -> (0, 0, 0)
    fastExp' = (1 - alphaFast) * fastExp + alphaFast * abs x
    slowExp' = (1 - alphaSlow) * slowExp + alphaSlow * abs x
    alphaFast = 0.05
    alphaSlow = 0.01
    border = mean + sqrt variance
    curSt' =
      case curSt of
        Low
          | fastExp' > border && slowExp' > 0.612 * border -> High
        High
          | fastExp' <= 0.612 * border && slowExp' <= 0.612 * border -> Low
        st -> st


addValueToRegime :: RegimeDetection -> Double -> RegimeDetection
addValueToRegime (RegimeDetection welAll expFast expSlow st) x = RegimeDetection welAll' expFast' expSlow' st'
  where
    (expFast', expSlow', st') = updateExp welAll' (expFast, expSlow, st) x
    welAll' = addValue welAll x


currentRegimeExp :: RegimeDetection -> State
currentRegimeExp (RegimeDetection _ _ _ st) = st
