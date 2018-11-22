{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.NeuralNetwork.NNConfig where

import           ML.BORL.NeuralNetwork.Scaling

import           Control.Lens
import           Grenade


type Cache k = [(k,Double)]


data NNConfig k = NNConfig
  { _toNetInp         :: k -> [Double]
  , _cache            :: Cache k
  , _trainBatchSize   :: Int
  , _learningParams   :: LearningParameters
  , _prettyPrintElems :: [k]
  , _scaleParameters  :: ScalingNetOutParameters
  }
makeLenses ''NNConfig


