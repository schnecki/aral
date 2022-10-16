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
import           GHC.Generics
import           Prelude             hiding ((<>))


-- | Regression Configuration for each node.
data RegressionConfig = RegressionConfig
  { regConfigDataOutStepSize            :: !Double -- ^ Step size in terms of output value to group observation data.
  , regConfigDataMaxObservationsPerStep :: !Int    -- ^ Maximum number of data points per group.
  --  , regConfigFunction                   :: !RegFunction              -- ^ Regression function.
  , regConfigVerbose                    :: !Bool   -- ^ Verbose output
  } deriving (Eq, Show, Generic, NFData)


instance Serialize RegressionConfig where
  get = (RegressionConfig <$> get <*> get <*> get) <|> (RegressionConfig <$> get <*> get <*> pure False)


instance Default RegressionConfig where
  def = RegressionConfig 0.1 30 False
