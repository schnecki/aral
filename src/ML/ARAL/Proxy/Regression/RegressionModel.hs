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
import           Data.List                   (intersperse, scanl', sort)
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

-- | Number of required coefficients for vector of specific @RegressionModels@ and specified input vector length..
nrCoefsRegressionModels :: RegressionModels -> Int -> Int
nrCoefsRegressionModels models n = (+1) . VB.sum . VB.map (`nrCoefsRegressionModel` n) $ models

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

-- | Definition of a regression model.
data RegressionModel
  = RegModelAll RegressionTerm                     -- ^ Regression over all inputs
  | RegModelIndices (VB.Vector Int) RegressionTerm -- ^ Certain function to apply to specific inputs
  deriving (Eq, Ord, NFData, Generic, Serialize)

instance Show RegressionModel where
  show (RegModelAll term) = "RegModelAll " ++ show term
  show (RegModelIndices ind term) = "RegModelIndices [" ++ lsTxt ++ "] " ++ show term
    where ind' =  sort $ VB.toList ind
          ls = foldl (\acc@((start, end):rs) x -> if end + 1 == x then (start, x):rs else (x, x):acc) [(head ind', head ind')] ind'
          lsTxt = concat $ intersperse "," $ map (\(start, end) -> if start == end then show start else (show start ++ "-" ++ show end)) ls


-- | Number of required coefficients for a specific @RegressionModel@ and specified input vector length..
nrCoefsRegressionModel :: RegressionModel -> Int -> Int
nrCoefsRegressionModel (RegModelAll fun) n          = nrCoefsRegressionTerm fun n
nrCoefsRegressionModel (RegModelIndices idxs fun) n = nrCoefsRegressionTerm fun (VB.length idxs)


-- | Compute function for one specified regression model.
computeModel ::
  (ModelVector v, Foldable v, Floating a)
  => RegressionModel -- ^ Regression function
  -> Model v a          -- ^ theta vector, the model's parameters
  -> v a                -- ^ @x@ vector, with the observed numbers
  -> a                  -- ^ predicted @y@ for this observation
computeModel (RegModelAll fun) theta inp             = compute fun theta inp
computeModel (RegModelIndices indices fun) theta inp = compute fun theta (fFromList $ map (inp `fIdx`) (VB.toList indices))

-- | Function to apply to the input.
data RegressionTerm
  = RegTermLinear    -- ^ Linear model
  | RegTermQuadratic -- ^ Quadratic input: f((x1,x2,x3)) = a1 * x1^2 + a2 * x2^2 + a3 * x2^3.
  | RegTermNonLinear -- ^ Non-linear terms wo self-multiplication: f((x1,x2,x3)) = a1 * x1 * x2 + a2 * x1 * x3 + a3 * x2 * x3
  | RegTermPriceHLC  -- ^ Term for price spread: f([h1,l1,c1,h2,l2,c2,..]) = a1 * (h1 - l1) + a2 * (c1 - l1) + a1 * (h2 - l2) + a2 * (c2 - l2)
  deriving (Show, Eq, Ord, NFData, Generic, Serialize, Enum, Bounded)

-- | Number of required coefficients for a specific @RegressionTerm@ and specified input vector length..
nrCoefsRegressionTerm :: RegressionTerm -> Int -> Int
nrCoefsRegressionTerm RegTermLinear n    = n
nrCoefsRegressionTerm RegTermQuadratic n = n
nrCoefsRegressionTerm RegTermNonLinear n = (n * (n-1)) `div` 2
nrCoefsRegressionTerm RegTermPriceHLC n
  | n `mod` 3 == 0 = 2 * (n `div` 3)
  | otherwise = error $ "nrCoefsRegressionTerm: RegTermPriceHLC expect the input length being a multiple of 3. Received: " ++ show n

-- over :: Int -> Int -> Int
-- over n k = fac n / (fac k * fac (n - k))

--


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
compute RegTermPriceHLC theta x
  | fLength x `mod` 3 /= 0 = error $ "compute: Unexpected input length for RegTermPriceHLC: " ++ show (fLength x)
  | otherwise = theta `dot'` fConcatList (fmap (\idx -> mkCalc (x `fIdx` idx, x `fIdx` (idx + 1), x `fIdx` (idx + 2))) [0, 3..fLength x - 3])
  where mkCalc (h, l, c) = fFromList [(h - l), (c - l)]
