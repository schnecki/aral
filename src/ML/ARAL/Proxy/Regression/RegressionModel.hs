{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
module ML.ARAL.Proxy.Regression.RegressionModel
    ( RegressionModel (..)
    , fromRegressionModel
    ) where

import           Control.DeepSeq
import           Data.Serialize
import qualified Data.Vector.Storable as VS
import           GHC.Generics


data RegressionModel =
  RegLinear
  deriving (Show, Eq, Ord, NFData, Generic, Serialize, Enum, Bounded)


fromRegressionModel :: RegressionModel -> VS.Vector Double  -> VS.Vector Double
fromRegressionModel RegLinear xs = xs
