{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.NeuralNetwork.NNConfig where

import           ML.BORL.NeuralNetwork.ReplayMemory
import           ML.BORL.NeuralNetwork.Scaling

import           Control.Lens
import           Grenade

data NNConfig k = NNConfig
  { _toNetInp             :: k -> [Double]
  , _replayMemory         :: ReplayMemory k
  , _trainBatchSize       :: Int
  , _learningParams       :: LearningParameters
  , _prettyPrintElems     :: [k]
  , _scaleParameters      :: ScalingNetOutParameters
  , _updateTargetInterval :: Integer
  }
makeLenses ''NNConfig


