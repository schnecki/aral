module ML.ARAL.NeuralNetwork.Normalisation
    ( normaliseStateFeature

    ) where

import qualified Data.Vector.Storable                        as V
import           Statistics.Sample.WelfordOnlineMeanVariance

import           ML.ARAL.Types

normaliseStateFeature :: WelfordExistingAggregate StateFeatures -> StateFeatures -> StateFeatures
normaliseStateFeature WelfordExistingAggregateEmpty x = x
normaliseStateFeature wel feats = V.zipWith3 (\mu var f -> min 5 . max (-5) $ (f - mu) / sqrt var) mean variance feats
  where (mean, _, variance) = finalize wel


-- instance (WelfordOnline StateFeatures)
