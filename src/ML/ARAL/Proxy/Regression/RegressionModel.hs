{-# LANGUAGE DeriveAnyClass   #-}
{-# LANGUAGE DeriveGeneric    #-}
{-# LANGUAGE FlexibleContexts #-}
module ML.ARAL.Proxy.Regression.RegressionModel
    ( RegressionModel (..)
    , RegressionFunction (..)
    -- , fromRegressionModel
    -- , fromRegressionFunction
    ) where

import           Control.DeepSeq
import           Data.Dynamic
import           Data.Monoid
import           Data.Reflection
import           Data.Serialize
import qualified Data.Vector                as VB
import qualified Data.Vector.Serialize      ()
import qualified Data.Vector.Storable       as VS
import           GHC.Generics
import           Numeric.Regression.Generic


-- | Definition of a regression model.
data RegressionModel
  = RegModelAll RegressionFunction                     -- ^ Regression over all inputs
  | RegModelIndices (VB.Vector Int) RegressionFunction -- ^ Certain function to apply to specific inputs
  deriving (Show, Eq, Ord, NFData, Generic, Serialize)


-- fromRegressionModel :: (Floating a) => RegressionModel -> VB.Vector a -> VB.Vector a -> a
-- fromRegressionModel (RegModelAll fun)          = fromRegressionFunction fun
-- fromRegressionModel (RegModelIndices idxs fun) = \theta inp -> fromRegressionFunction fun (selectIndices theta) (selectIndices inp)
--   where
--     selectIndices vec = VB.foldl' (\acc idx -> acc VB.++ VB.singleton (vec VB.! idx)) VB.empty idxs

-- fromRegressionFunctionDbl :: (Reifies s Tape) => RegressionFunction -> VB.Vector (ReverseDouble s) -> (VB.Vector (ReverseDouble s) -> ReverseDouble s)
-- fromRegressionFunctionDbl RegLinear theta inp = theta `dot` inp

-- -- | Function to apply to the input.
-- data RegressionFunction =
--   RegLinear
--   deriving (Show, Eq, Ord, NFData, Generic, Serialize, Enum, Bounded)


-- fromRegressionFunction :: (Num a, Reifies s Tape) => RegressionFunction -> VB.Vector (Reverse s a) -> (VB.Vector (Reverse s a) -> Reverse s a)
-- fromRegressionFunction RegLinear theta inp = theta `dot` inp


dot :: (Num a)
    => VB.Vector a
    -> VB.Vector a
    -> a
dot x y = (+y0) . getSum . foldMap Sum $ VB.zipWith (*) x y
  where y0 | VB.length x == VB.length y + 1 = VB.last x
           | VB.length y == VB.length x + 1 = VB.last y
           | otherwise = 0
