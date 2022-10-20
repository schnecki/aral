{-# LANGUAGE DeriveAnyClass   #-}
{-# LANGUAGE DeriveGeneric    #-}
{-# LANGUAGE FlexibleContexts #-}
module ML.ARAL.Proxy.Regression.RegressionModel
    ( RegressionModels (..)
    , RegressionModel (..)
    , RegressionTerm (..)
    , nrCoefsRegressionModels
    , nrCoefsRegressionModel
    , nrCoefsRegressionTerm
    , computeModels
    , computeModel
    , computeTerm
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

import           Debug.Trace

data RegressionModels =
  RegressionModels
    { mdlsUseIntercept :: Bool                      -- ^ Use an Intercept a0 in f(xs) = as * xs + a0
    , mdls             :: VB.Vector RegressionModel -- ^ Regression models
    } deriving (Show, Eq, Ord, NFData, Generic, Serialize)


-- | Number of required coefficients for vector of specific @RegressionModels@ and specified input vector length..
nrCoefsRegressionModels :: RegressionModels -> Int -> Int
nrCoefsRegressionModels (RegressionModels useIntercept models) n = (+ nrIntercept) . VB.sum . VB.map (`nrCoefsRegressionModel` n) $ models
  where
    nrIntercept
      | useIntercept = 1
      | otherwise = 0

-- | Compute the error of the models.
computeModels :: (ModelVector v, Foldable v, Floating a)
  => RegressionModels -- ^ Regression function
  -> Model v a          -- ^ theta vector, the model's parameters
  -> v a                -- ^ @x@ vector, with the observed numbers
  -> a                  -- ^ predicted @y@ for this observation
computeModels (RegressionModels _ mdls) _ _ | VB.null mdls = error "computeModels: Empty models. Cannot compute regression!"
computeModels (RegressionModels useIntercept mdls) theta inp =
  -- trace ("Length theta: " ++ show (fLength theta))
  -- trace ("nrCoefs: " ++ show nrCoefs)
  -- trace ("nrBefCoefs: " ++ show nrBefCoefs)
  -- trace ("res: " ++ show (VB.zipWith (\nrBefs nrCofs -> take nrCofs $ drop nrBefs  [0..fLength theta - 1]) nrBefCoefs nrCoefs))
  (+ fLast theta) . VB.sum $ VB.zipWith3 (\nrBefs nrCofs mdl -> computeModel mdl (fTake nrCofs . fDrop nrBefs $ theta) inp) nrBefCoefs nrCoefs mdls
  where
    nrCoefs = VB.map (\mdl -> nrCoefsRegressionModel mdl (fLength inp)) mdls
    nrBefCoefs = VB.scanl' (+) 0 nrCoefs
    intercept
      | useIntercept = fLength theta
      | otherwise = 0

-- | Definition of a regression model.
data RegressionModel
  = RegModelAll !RegressionTerm                                      -- ^ Regression over all inputs
  | RegModelIndices !(VB.Vector Int) !RegressionTerm                 -- ^ Certain function to apply to specific inputs
  | RegModelLayer !Bool !RegressionTerm !(VB.Vector RegressionModel) -- ^ Layered regressions (compose a regression of regressions) with or without intercepts for each model
  deriving (Eq, Ord, NFData, Generic, Serialize)

instance Show RegressionModel where
  show (RegModelAll term) = "RegModelAll " ++ show term
  show (RegModelIndices ind term) = "RegModelIndices [" ++ lsTxt ++ "] " ++ show term
    where ind' =  sort $ VB.toList ind
          ls = foldl (\acc@((start, end):rs) x -> if end + 1 == x then (start, x):rs else (x, x):acc) [(head ind', head ind')] (tail ind')
          lsTxt = concat $ intersperse "," $ map (\(start, end) -> if start == end then show start else (show start ++ "-" ++ show end)) ls
  show (RegModelLayer useIntercepts term model) = show term ++ "(" ++ show model ++ ")" ++ intercept
    where intercept
            | useIntercepts = " with model intercepts"
            | otherwise = ""


-- | Number of required coefficients for a specific @RegressionModel@ and specified input vector length..
nrCoefsRegressionModel :: RegressionModel -> Int -> Int
nrCoefsRegressionModel (RegModelAll term) n          = nrCoefsRegressionTerm term n
nrCoefsRegressionModel (RegModelIndices idxs term) n = nrCoefsRegressionTerm term (VB.length idxs)
nrCoefsRegressionModel (RegModelLayer useIntercepts term layers) n = nrCoefsRegressionTerm term (VB.length layers) + VB.sum (VB.map (`nrCoefsRegressionModel` n) layers) + intercept
  where
    intercept
      | useIntercepts = VB.length layers
      | otherwise = 0

-- | Compute function for one specified regression model.
computeModel ::
  (ModelVector v, Foldable v, Floating a)
  => RegressionModel -- ^ Regression function
  -> Model v a          -- ^ theta vector, the model's parameters
  -> v a                -- ^ @x@ vector, with the observed numbers
  -> a                  -- ^ predicted @y@ for this observation
computeModel (RegModelAll term) theta inp             = computeTerm term theta inp
computeModel (RegModelIndices indices term) theta inp = computeTerm term theta (fFromList $ map (inp `fIdx`) (VB.toList indices))
computeModel (RegModelLayer useIntercepts term layers) theta inp =
  computeTerm term (fTake lenTerm theta) (fFromList . VB.toList $ VB.zipWith3 computeSubModel nrBefCoefs nrCoefs layers)
  where
    len = VB.length layers
    lenTerm = nrCoefsRegressionTerm term len
    lenInterceptMdl
      | useIntercepts = 1
      | otherwise = 0
    nrCoefs = VB.map (\mdl -> lenInterceptMdl + nrCoefsRegressionModel mdl (fLength inp)) layers
    nrBefCoefs = VB.scanl' (+) 0 nrCoefs
    computeSubModel nrBefs nrCoefs mdl =
      let theta' = fTake nrCoefs . fDrop (nrBefs + lenTerm) $ theta
          (thetaMdl, intercept)
            | useIntercepts = (fInit theta', fLast theta')
            | otherwise = (theta', 0)
       in computeModel mdl thetaMdl inp + intercept


-- TODO:
-- class RegressionTerm x where
--   nrCoefsRegressionTerm :: x -> Int -> Int
--   computeTerm :: (ModelVector v, Foldable v, Floating a) => x -> Model v a -> v a -> a

-- | Function to apply to the input.
data RegressionTerm
  = RegTermLinear                               -- ^ Linear model:                             f((x1,x2,x3)) = a1 * x1 + a2 * x2 + a3 * x3
  | RegTermQuadratic                            -- ^ Quadratic input:                          f((x1,x2,x3)) = a1 * x1^2 + a2 * x2^2 + a3 * x2^3.
  | RegTermNonLinear                            -- ^ Non-linear terms wo self-multiplication:  f((x1,x2,x3)) = a1 * x1 * x2 + a2 * x1 * x3 + a3 * x2 * x3
  | RegTermAddition !(VB.Vector RegressionTerm) -- ^ Addition of multiple RegressionTerms:     f((t1,t2), (x1,x2,x3)) = t1((x1,x2,x3)) + t2((x1,x2,x3))
  | RegTermPriceHLC                             -- ^ Term for price spread:                    f((h1,l1,c1,h2,l2,c2,..)) = a1 * (h1 - l1) + a2 * (c1 - l1) + a3 * (h2 - l2) + a4 * (c2 - l2)
  deriving (Show, Eq, Ord, NFData, Generic, Serialize)

-- | Number of required coefficients for a specific @RegressionTerm@ and specified input vector length..
nrCoefsRegressionTerm :: RegressionTerm -> Int -> Int
nrCoefsRegressionTerm RegTermLinear n    = n
nrCoefsRegressionTerm RegTermQuadratic n = n
nrCoefsRegressionTerm RegTermNonLinear n = (n * (n-1)) `div` 2
nrCoefsRegressionTerm (RegTermAddition terms) n = VB.sum (VB.map (`nrCoefsRegressionTerm` n) terms)
nrCoefsRegressionTerm RegTermPriceHLC n
  | n `mod` 3 == 0 = 2 * (n `div` 3)
  | otherwise = error $ "nrCoefsRegressionTerm: RegTermPriceHLC expect the input length being a multiple of 3. Received: " ++ show n

-- over :: Int -> Int -> Int
-- over n k = fac n / (fac k * fac (n - k))

-- | Compute the predicted value for the given model on the given observation.
computeTerm ::
     (ModelVector v, Foldable v, Floating a)
  => RegressionTerm -- ^ Regression function
  -> Model v a          -- ^ theta vector, the model's parameters
  -> v a                -- ^ @x@ vector, with the observed numbers
  -> a                  -- ^ predicted @y@ for this observation
computeTerm RegTermLinear theta x    = theta `dot'` x
computeTerm RegTermQuadratic theta x = theta `dot'` fMap (^2) x
computeTerm RegTermNonLinear theta x = theta `dot'` fFromList (map (\(xIdx, yIdx) -> x `fIdx` xIdx * x `fIdx` yIdx) xs)
  where xs = [(xIdx, yIdx) | xIdx <- [0..n-1], yIdx <- [0..n-1], xIdx < yIdx ]
        n = fLength x
computeTerm (RegTermAddition terms) theta x = VB.sum $ VB.zipWith3 computeSubTerm nrBefCoefs nrCoefs terms
  where
    nrCoefs = VB.map (`nrCoefsRegressionTerm` fLength x) terms
    nrBefCoefs = VB.scanl' (+) 0 nrCoefs
    computeSubTerm nrBefs nrCoefs term =
      let theta' = fTake nrCoefs . fDrop nrBefs $ theta
       in computeTerm term theta' x
computeTerm RegTermPriceHLC theta x
  | fLength x `mod` 3 /= 0 = error $ "compute: Unexpected input length for RegTermPriceHLC: " ++ show (fLength x)
  | otherwise = theta `dot'` fConcatList (fmap (\idx -> mkCalc (x `fIdx` idx, x `fIdx` (idx + 1), x `fIdx` (idx + 2))) [0, 3..fLength x - 3])
  where mkCalc (h, l, c) = fFromList [(h - l), (c - l)]
