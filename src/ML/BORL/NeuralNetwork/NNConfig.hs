{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.NeuralNetwork.NNConfig where

import           ML.BORL.NeuralNetwork.ReplayMemory
import           ML.BORL.NeuralNetwork.Scaling
import           ML.BORL.Types

import           Control.DeepSeq
import           Control.Lens
import           Grenade

data NNConfig s = NNConfig
  { _toNetInp              :: !(s -> [Double])
  , _replayMemoryMaxSize   :: !Int
  , _trainBatchSize        :: !Int
  , _grenadeLearningParams :: !LearningParameters
  , _prettyPrintElems      :: ![s]
  , _scaleParameters       :: !ScalingNetOutParameters
  , _updateTargetInterval  :: !Integer
  , _trainMSEMax           :: !(Maybe MSE) -- ^ Mean squared error to train for.
  }
makeLenses ''NNConfig

instance (NFData k) => NFData (NNConfig k) where
  rnf (NNConfig inp rep tr lp pp sc up mse) = rnf inp `seq` rnf rep `seq` rnf tr `seq` rnf lp `seq` rnf pp `seq` rnf sc `seq` rnf up `seq` rnf mse


