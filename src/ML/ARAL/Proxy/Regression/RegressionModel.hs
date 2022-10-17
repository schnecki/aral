{-# LANGUAGE DeriveAnyClass   #-}
{-# LANGUAGE DeriveGeneric    #-}
{-# LANGUAGE FlexibleContexts #-}
module ML.ARAL.Proxy.Regression.RegressionModel
    ( RegressionModels
    , RegressionModel (..)
    , RegressionFunction (..)
    , nrCoefsRegressionModels
    , nrCoefsRegressionModel
    , nrCoefsRegressionFunction
    , computeModels
    , computeModel
    , compute
    -- , fromRegressionModel
    -- , fromRegressionFunction
    ) where

import           Control.DeepSeq
import           Data.Dynamic
import           Data.List                   (scanl')
import           Data.Monoid
import           Data.Reflection
import           Data.Serialize
import qualified Data.Vector                 as VB
import qualified Data.Vector.Serialize       ()
import qualified Data.Vector.Storable        as VS
import           GHC.Generics
import           Numeric.Regression.Generic
import           Numeric.Regression.Internal


type RegressionModels = VB.Vector RegressionModel


-- | Definition of a regression model.
data RegressionModel
  = RegModelAll RegressionFunction                     -- ^ Regression over all inputs
  | RegModelIndices (VB.Vector Int) RegressionFunction -- ^ Certain function to apply to specific inputs
  deriving (Show, Eq, Ord, NFData, Generic, Serialize)

-- | Number of required coefficients for vector of specific @RegressionModels@ and specified input vector length..
nrCoefsRegressionModels :: RegressionModels -> Int -> Int
nrCoefsRegressionModels models n = VB.sum . VB.map (`nrCoefsRegressionModel` n) $ models


-- | Number of required coefficients for a specific @RegressionModel@ and specified input vector length..
nrCoefsRegressionModel :: RegressionModel -> Int -> Int
nrCoefsRegressionModel (RegModelAll fun) n          = nrCoefsRegressionFunction fun n
nrCoefsRegressionModel (RegModelIndices idxs fun) n = nrCoefsRegressionFunction fun (VB.length idxs)

-- | Number of required coefficients for a specific @RegressionFunction@ and specified input vector length..
nrCoefsRegressionFunction :: RegressionFunction -> Int -> Int
nrCoefsRegressionFunction RegLinear n    = n + 1
nrCoefsRegressionFunction RegQuadratic n = 2 * n + 1


-- | Function to apply to the input.
data RegressionFunction
  = RegLinear    -- ^ Linear model
  | RegQuadratic -- ^ Quadratic input: f((x1,x2,...)) = a1 * x1^2 + a2 * x2 * 2.
  deriving (Show, Eq, Ord, NFData, Generic, Serialize, Enum, Bounded)


-- | Compute the error of the models.
computeModels :: (ModelVector v, Foldable v, Floating a)
  => RegressionModels -- ^ Regression function
  -> Model v a          -- ^ theta vector, the model's parameters
  -> v a                -- ^ @x@ vector, with the observed numbers
  -> a                  -- ^ predicted @y@ for this observation
computeModels mdls _ _ | VB.null mdls = error "computeModels: Empty models. Cannot compute regression!"
computeModels mdls theta inp = (/ fromIntegral (VB.length mdls)) . VB.sum $ VB.zipWith3 (\nrBefs nrCofs  mdl -> computeModel mdl (fTake nrCofs . fDrop nrBefs $ theta) inp) nrBefCoefs nrCoefs mdls
  where nrCoefs = VB.map (\mdl -> nrCoefsRegressionModel mdl (fLength inp)) mdls
        nrBefCoefs = VB.scanl' (+) 0 nrCoefs

computeModel ::
  (ModelVector v, Foldable v, Floating a)
  => RegressionModel -- ^ Regression function
  -> Model v a          -- ^ theta vector, the model's parameters
  -> v a                -- ^ @x@ vector, with the observed numbers
  -> a                  -- ^ predicted @y@ for this observation
computeModel (RegModelAll fun) theta inp = compute fun theta inp


-- | Compute the predicted value for the given model on the given observation.
compute ::
     (ModelVector v, Foldable v, Floating a)
  => RegressionFunction -- ^ Regression function
  -> Model v a          -- ^ theta vector, the model's parameters
  -> v a                -- ^ @x@ vector, with the observed numbers
  -> a                  -- ^ predicted @y@ for this observation
compute RegLinear theta x    = theta `dot` x
compute RegQuadratic theta x = theta `dot` (x `fAppend` fMap (^2) x) -- TODO: all x1*x2 combinations + theta * x
{-# INLINE compute #-}
