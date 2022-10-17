{-# LANGUAGE DeriveAnyClass   #-}
{-# LANGUAGE DeriveGeneric    #-}
{-# LANGUAGE FlexibleContexts #-}
module ML.ARAL.Proxy.Regression.RegressionModel
    ( RegressionModel (..)
    , RegressionFunction (..)
    , regressionNrCoefficients
    , compute
    -- , fromRegressionModel
    -- , fromRegressionFunction
    ) where

import           Control.DeepSeq
import           Data.Dynamic
import           Data.Monoid
import           Data.Reflection
import           Data.Serialize
import qualified Data.Vector                 as VB
import qualified Data.Vector.Serialize       ()
import qualified Data.Vector.Storable        as VS
import           GHC.Generics
import           Numeric.Regression.Generic
import           Numeric.Regression.Internal


type RegressionModels = [RegressionModel]


-- | Definition of a regression model.
data RegressionModel
  = RegModelAll RegressionFunction                     -- ^ Regression over all inputs
  | RegModelIndices (VB.Vector Int) RegressionFunction -- ^ Certain function to apply to specific inputs
  deriving (Show, Eq, Ord, NFData, Generic, Serialize)


-- | Number of coefficients for specified model.
regressionNrCoefficients :: RegressionFunction -> Int -> Int
regressionNrCoefficients RegLinear n    = n + 1
regressionNrCoefficients RegQuadratic n = 2 * n + 1


-- fromRegressionModel :: (Floating a) => RegressionModel -> VB.Vector a -> VB.Vector a -> a
-- fromRegressionModel (RegModelAll fun)          = fromRegressionFunction fun
-- fromRegressionModel (RegModelIndices idxs fun) = \theta inp -> fromRegressionFunction fun (selectIndices theta) (selectIndices inp)
--   where
--     selectIndices vec = VB.foldl' (\acc idx -> acc VB.++ VB.singleton (vec VB.! idx)) VB.empty idxs

-- fromRegressionFunctionDbl :: (Reifies s Tape) => RegressionFunction -> VB.Vector (ReverseDouble s) -> (VB.Vector (ReverseDouble s) -> ReverseDouble s)
-- fromRegressionFunctionDbl RegLinear theta inp = theta `dot` inp

-- | Function to apply to the input.
data RegressionFunction
  = RegLinear    -- ^ Linear model
  | RegQuadratic -- ^ Quadratic input: f((x1,x2,...)) = a1 * x1^2 + a2 * x2 * 2.
  deriving (Show, Eq, Ord, NFData, Generic, Serialize, Enum, Bounded)


-- fromRegressionFunction :: (Num a, Reifies s Tape) => RegressionFunction -> VB.Vector (Reverse s a) -> (VB.Vector (Reverse s a) -> Reverse s a)
-- fromRegressionFunction RegLinear theta inp = theta `dot` inp

computeModel ::
  (ModelVector v, Foldable v, Num a)
  => RegressionModel -- ^ Regression function
  -> Model v a          -- ^ theta vector, the model's parameters
  -> v a                -- ^ @x@ vector, with the observed numbers
  -> a                  -- ^ predicted @y@ for this observation
computeModel (RegModelAll fun) theta inp =  computeModel fun theta inp


-- | Compute the predicted value for the given model on the given observation.
compute ::
     (ModelVector v, Foldable v, Num a)
  => RegressionFunction -- ^ Regression function
  -> Model v a          -- ^ theta vector, the model's parameters
  -> v a                -- ^ @x@ vector, with the observed numbers
  -> a                  -- ^ predicted @y@ for this observation
compute RegLinear theta x    = theta `dot` x
compute RegQuadratic theta x = theta `dot` (x `fAppend` fMap (^2) x) -- TODO: all x1*x2 combinations + theta * x
{-# INLINE compute #-}
