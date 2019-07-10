{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.NeuralNetwork.NNConfig where

import           ML.BORL.NeuralNetwork.ReplayMemory
import           ML.BORL.NeuralNetwork.Scaling
import           ML.BORL.Types

import           Control.DeepSeq
import           Control.Lens
import           Grenade

type NetInputWoAction = [Double]
type NetInput = [Double]

data NNConfig = NNConfig
  { _replayMemoryMaxSize   :: !Int
  , _trainBatchSize        :: !Int
  , _grenadeLearningParams :: !LearningParameters
  , _prettyPrintElems      :: ![NetInput]
  , _scaleParameters       :: !ScalingNetOutParameters
  , _updateTargetInterval  :: !Integer
  , _trainMSEMax           :: !(Maybe MSE) -- ^ Mean squared error to train for.
  }
makeLenses ''NNConfig


instance NFData NNConfig where
  rnf (NNConfig rep tr lp pp sc up mse) = rnf rep `seq` rnf tr `seq` rnf lp `seq` rnf pp `seq` rnf sc `seq` rnf up `seq` rnf mse


