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

import           ML.ARAL.Proxy.Regression.RegressionModel


-- | Regression Configuration for each node.
data RegressionConfig = RegressionConfig
  { regConfigDataOutStepSize            :: !Double -- ^ Step size in terms of output value to group observation data.
  , regConfigDataMaxObservationsPerStep :: !Int    -- ^ Maximum number of data points per group.
  , regConfigVerbose                    :: !Bool   -- ^ Verbose output
  , regConfigModel                      :: !RegressionFunction -- (VB.Vector RegressionModel)
  } deriving (Eq, Show, Generic, NFData, Serialize)


-- instance Serialize RegressionConfig where
--   get = RegressionConfig <$> get <*> get <*> get <*> get


instance Default RegressionConfig where
  def = RegressionConfig 0.1 30 False RegQuadratic -- RegLinear
