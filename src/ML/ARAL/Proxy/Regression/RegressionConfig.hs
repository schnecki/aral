{-# LANGUAGE DeriveAnyClass    #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE OverloadedStrings #-}
module ML.ARAL.Proxy.Regression.RegressionConfig
    ( RegressionConfig (..)
    ) where

import           Control.Applicative
import           Control.DeepSeq
import           Data.Default
import           Data.Serialize
import qualified Data.Vector                              as VB
import           GHC.Generics
import           Prelude                                  hiding ((<>))

import           ML.ARAL.Decay.Type
import           ML.ARAL.Proxy.Regression.RegressionModel


-- | Regression Configuration for each node.
data RegressionConfig = RegressionConfig
  { regConfigDataOutStepSize            :: !Double           -- ^ Step size in terms of output value to group observation data. Default: 0.1
  , regConfigDataMaxObservationsPerStep :: !Int              -- ^ Maximum number of data points per group. Default: 5
  , regConfigMinCorrelation             :: !Double           -- ^ Minimum correlation, or feature is turned off completely. Default: 0.01
  , regConfigModel                      :: !RegressionModels -- ^ Models to use for Regression: Default: @RegressionModels True $ VB.fromList [RegModelAll RegTermLinear]@
  , regConfigUseVolatilityRegimes       :: !Bool             -- ^ Use differnt regression functions for different variance regimes. Default: False
  , regConfigVerbose                    :: !Bool             -- ^ Verbose output. Default: False
  } deriving (Eq, Show, Generic, NFData, Serialize)


instance Default RegressionConfig where
  def = RegressionConfig 0.1 5 0.01 (RegressionModels True $ VB.fromList [RegModelAll RegTermLinear]) False False
