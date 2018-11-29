{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.NeuralNetwork.NNConfig where

import           ML.BORL.NeuralNetwork.ReplayMemory
import           ML.BORL.NeuralNetwork.Scaling

import           Control.DeepSeq
import           Control.Lens
import           Grenade

data NNConfig k = NNConfig
  { _toNetInp             :: !(k -> [Double])
  , _replayMemory         :: !(ReplayMemory k)
  , _trainBatchSize       :: !Int
  , _learningParams       :: !LearningParameters
  , _prettyPrintElems     :: ![k]
  , _scaleParameters      :: !ScalingNetOutParameters
  , _updateTargetInterval :: !Integer
  }
makeLenses ''NNConfig

instance (NFData k) => NFData (NNConfig k) where
  rnf (NNConfig inp rep tr lp pp sc up) = rnf inp `seq` rnf rep `seq` rnf tr `seq` rnf lp `seq` rnf pp `seq` rnf sc `seq` rnf up


