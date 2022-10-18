{-# LANGUAGE DeriveAnyClass   #-}
{-# LANGUAGE DeriveGeneric    #-}
{-# LANGUAGE FlexibleContexts #-}
module ML.ARAL.Proxy.Regression.RegressionModel
    ( RegressionModels
    , RegressionModel (..)
    , RegressionTerm (..)
    , nrCoefsRegressionModels
    , nrCoefsRegressionModel
    , nrCoefsRegressionTerm
    , computeModels
    , computeModel
    , compute
    -- , fromRegressionModel
    -- , fromRegressionTerm
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
  = RegModelAll RegressionTerm                     -- ^ Regression over all inputs
  | RegModelIndices (VB.Vector Int) RegressionTerm -- ^ Certain function to apply to specific inputs
  deriving (Show, Eq, Ord, NFData, Generic, Serialize)

-- | Number of required coefficients for vector of specific @RegressionModels@ and specified input vector length..
nrCoefsRegressionModels :: RegressionModels -> Int -> Int
nrCoefsRegressionModels models n = (+1) . VB.sum . VB.map (`nrCoefsRegressionModel` n) $ models


-- | Number of required coefficients for a specific @RegressionModel@ and specified input vector length..
nrCoefsRegressionModel :: RegressionModel -> Int -> Int
nrCoefsRegressionModel (RegModelAll fun) n          = nrCoefsRegressionTerm fun n
nrCoefsRegressionModel (RegModelIndices idxs fun) n = nrCoefsRegressionTerm fun (VB.length idxs)

-- | Number of required coefficients for a specific @RegressionTerm@ and specified input vector length..
nrCoefsRegressionTerm :: RegressionTerm -> Int -> Int
nrCoefsRegressionTerm RegTermLinear n    = n
nrCoefsRegressionTerm RegTermQuadratic n = n
nrCoefsRegressionTerm RegTermNonLinear n = (n * (n-1)) `div` 2

-- over :: Int -> Int -> Int
-- over n k = fac n / (fac k * fac (n - k))

--

-- | Function to apply to the input.
data RegressionTerm
  = RegTermLinear    -- ^ Linear model
  | RegTermQuadratic -- ^ Quadratic input: f((x1,x2,x3)) = a1 * x1^2 + a2 * x2^2 + a3 * x2^3.
  | RegTermNonLinear -- ^ Non-linear terms wo self-multiplication: f((x1,x2,x3)) = a1 * x1 * x2 + a2 * x1 * x3 + a3 * x2 * x3
  deriving (Show, Eq, Ord, NFData, Generic, Serialize, Enum, Bounded)


-- | Compute the error of the models.
computeModels :: (ModelVector v, Foldable v, Floating a)
  => RegressionModels -- ^ Regression function
  -> Model v a          -- ^ theta vector, the model's parameters
  -> v a                -- ^ @x@ vector, with the observed numbers
  -> a                  -- ^ predicted @y@ for this observation
computeModels mdls _ _ | VB.null mdls = error "computeModels: Empty models. Cannot compute regression!"
computeModels mdls theta inp = (+ fLast theta) . VB.sum $ VB.zipWith3 (\nrBefs nrCofs  mdl -> computeModel mdl (fTake nrCofs . fDrop nrBefs $ theta) inp) nrBefCoefs nrCoefs mdls
  where nrCoefs = VB.map (\mdl -> nrCoefsRegressionModel mdl (fLength inp)) mdls
        nrBefCoefs = VB.scanl' (+) 0 nrCoefs

computeModel ::
  (ModelVector v, Foldable v, Floating a)
  => RegressionModel -- ^ Regression function
  -> Model v a          -- ^ theta vector, the model's parameters
  -> v a                -- ^ @x@ vector, with the observed numbers
  -> a                  -- ^ predicted @y@ for this observation
computeModel (RegModelAll fun) theta inp             = compute fun theta inp
computeModel (RegModelIndices indices fun) theta inp = compute fun theta (fFromList $ map (inp `fIdx`) (VB.toList indices))


-- | Compute the predicted value for the given model on the given observation.
compute ::
     (ModelVector v, Foldable v, Floating a)
  => RegressionTerm -- ^ Regression function
  -> Model v a          -- ^ theta vector, the model's parameters
  -> v a                -- ^ @x@ vector, with the observed numbers
  -> a                  -- ^ predicted @y@ for this observation
compute RegTermLinear theta x    = theta `dot'` x
compute RegTermQuadratic theta x = theta `dot'` fMap (^2) x
compute RegTermNonLinear theta x = theta `dot'` fFromList (map (\(xIdx, yIdx) -> x `fIdx` xIdx * x `fIdx` yIdx) xs)
  where xs = [(xIdx, yIdx) | xIdx <- [0..n-1], yIdx <- [0..n-1], xIdx < yIdx ]
        n = fLength x
