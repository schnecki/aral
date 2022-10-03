module ML.ARAL.NeuralNetwork.Normalisation
    ( normaliseStateFeature
    , normaliseStateFeatureUnbounded
    , normaliseUnbounded
    , denormaliseUnbounded
    ) where

import qualified Data.Vector.Storable                        as V
import           Statistics.Sample.WelfordOnlineMeanVariance

import           ML.ARAL.Types

sqrt' = max 1e-6 . sqrt

normaliseStateFeature :: WelfordExistingAggregate StateFeatures -> StateFeatures -> StateFeatures
normaliseStateFeature WelfordExistingAggregateEmpty x = x
normaliseStateFeature wel feats = V.zipWith3 (\mu var f -> min 5 . max (-5) $ (f - mu) / sqrt' var) mean variance feats
  where (mean, _, variance) = finalize wel


normaliseStateFeatureUnbounded :: WelfordExistingAggregate StateFeatures -> StateFeatures -> StateFeatures
normaliseStateFeatureUnbounded WelfordExistingAggregateEmpty x = x
normaliseStateFeatureUnbounded wel feats = V.zipWith3 (\mu var f -> (f - mu) / sqrt' var) mean variance feats
  where (mean, _, variance) = finalize wel

normaliseUnbounded :: WelfordExistingAggregate Double -> Double -> Double
normaliseUnbounded WelfordExistingAggregateEmpty x = x
normaliseUnbounded wel x = (x - mean) / sqrt' variance
  where (mean, _, variance) = finalize wel

denormaliseUnbounded :: WelfordExistingAggregate Double -> Double -> Double
denormaliseUnbounded WelfordExistingAggregateEmpty x = x
denormaliseUnbounded wel x = x * sqrt' variance + mean
  where (mean, _, variance) = finalize wel
