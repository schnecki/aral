module ML.ARAL.NeuralNetwork.Normalisation
    ( normaliseStateFeature
    , normaliseStateFeatureUnbounded
    , normaliseUnbounded
    , denormaliseUnbounded
    ) where

import qualified Data.Vector.Storable                        as VS
import           Statistics.Sample.WelfordOnlineMeanVariance

import           ML.ARAL.Types

sqrt' = max 1e-3 . sqrt

normaliseStateFeature :: WelfordExistingAggregate StateFeatures -> StateFeatures -> StateFeatures
normaliseStateFeature WelfordExistingAggregateEmpty x = VS.map (min 2 . max (-2)) x
normaliseStateFeature wel feats = VS.zipWith3 (\mu var f -> min 5 . max (-5) $ (f - mu) / sqrt' var) mean variance feats
  where (mean, _, variance) = finalize wel


normaliseStateFeatureUnbounded :: WelfordExistingAggregate StateFeatures -> StateFeatures -> StateFeatures
normaliseStateFeatureUnbounded WelfordExistingAggregateEmpty x = VS.map (min 2 . max (-2)) x
normaliseStateFeatureUnbounded wel feats
  | count < 100 = VS.zipWith3 (\mu var f -> min 5 . max (-5) $ (f - mu) / sqrt' var) mean variance feats
  | otherwise = VS.zipWith3 (\mu var f -> (f - mu) / sqrt' var) mean variance feats
  where (mean, _, variance) = finalize wel
        count = welfordCount wel

normaliseUnbounded :: WelfordExistingAggregate Double -> Double -> Double
normaliseUnbounded WelfordExistingAggregateEmpty x = min 2 . max (-2) $ x
normaliseUnbounded wel x
  | count < 100 = min 5 . max (-5) $ x - mean / sqrt' variance
  | otherwise = (x - mean) / sqrt' variance
  where (mean, _, variance) = finalize wel
        count = welfordCount wel

denormaliseUnbounded :: WelfordExistingAggregate Double -> Double -> Double
denormaliseUnbounded WelfordExistingAggregateEmpty x = min 2 . max (-2) $ x
denormaliseUnbounded wel x
  | count < 100 = min 5 . max (-5) $ x * sqrt' variance + mean
  | otherwise = x * sqrt' variance + mean
  where (mean, _, variance) = finalize wel
        count = welfordCount wel
