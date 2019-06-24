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

data NNConfig s = NNConfig
  { _toNetInp              :: !(s -> NetInput)
  , _replayMemoryMaxSize   :: !Int
  , _trainBatchSize        :: !Int
  , _grenadeLearningParams :: !LearningParameters
  , _prettyPrintElems      :: ![NetInput]
  , _scaleParameters       :: !ScalingNetOutParameters
  , _updateTargetInterval  :: !Integer
  , _trainMSEMax           :: !(Maybe MSE) -- ^ Mean squared error to train for.
  }
makeLenses ''NNConfig

mapNNConfigForSerialise :: NNConfig s -> NNConfig s'
mapNNConfigForSerialise (NNConfig inp rep bs lp pr sc up train) = NNConfig (const []) rep bs lp pr sc up train

instance (NFData k) => NFData (NNConfig k) where
  rnf (NNConfig inp rep tr lp pp sc up mse) = rnf inp `seq` rnf rep `seq` rnf tr `seq` rnf lp `seq` rnf pp `seq` rnf sc `seq` rnf up `seq` rnf mse


