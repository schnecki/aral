{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.NeuralNetwork.NNConfig where

import           ML.BORL.Decay
import           ML.BORL.NeuralNetwork.ReplayMemory
import           ML.BORL.NeuralNetwork.Scaling
import           ML.BORL.Types

import           Control.DeepSeq
import           Control.Lens
import           Grenade

type NetInputWoAction = [Double]
type NetInput = [Double]
type NetInputWithAction = [Double]

data NNConfig = NNConfig
  { _replayMemoryMaxSize             :: !Int                     -- ^ Maximum size of the replay memory.
  , _trainBatchSize                  :: !Int                     -- ^ Batch size for training. Values are fed from the replay memory.
  , _grenadeLearningParams           :: !LearningParameters      -- ^ Grenade (not used for Tensorflow!) learning parameters.
  , _grenadeLearningParamsDecay      :: !DecaySetup               -- ^ Decay setup for grenade learning parameters
  , _prettyPrintElems                :: ![NetInputWoAction]      -- ^ Sample input features for printing.
  , _scaleParameters                 :: !ScalingNetOutParameters -- ^ How to scale the output to the original range.
  , _stabilizationAdditionalRho      :: Double                   -- ^ Additional rho as a percantage of [minV, maxV] which is
                                                                 -- expected in the beginning.
  , _stabilizationAdditionalRhoDecay :: !DecaySetup               -- ^ Decay for stabilization
  , _updateTargetInterval            :: !Int                     -- ^ After how many steps should the target network be replaced by the worker?
  , _trainMSEMax                     :: !(Maybe MSE)             -- ^ Mean squared error when output is scaled to (-1,1). Used to initialise
                                                                 -- the neural network once the replay memory is filled.
  , _setExpSmoothParamsTo1           :: Bool                     -- ^ Set all exponential smoothing parameters to 1 and
                                                                 -- use ANN learning rate to decay learning solely.
  }
makeLenses ''NNConfig


instance NFData NNConfig where
  rnf (NNConfig rep tr lp dec pp sc stab stabDec up mse param) =
    rnf rep `seq` rnf tr `seq` rnf lp `seq` rnf dec `seq` rnf pp `seq` rnf sc `seq` rnf stab `seq` rnf stabDec `seq` rnf up `seq` rnf mse `seq` rnf param
