{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE DataKinds       #-}
{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE GADTs           #-}
{-# LANGUAGE KindSignatures  #-}
{-# LANGUAGE RankNTypes      #-}
{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.NeuralNetwork.NNConfig where

import           Control.DeepSeq
import           Control.Lens
import           Data.Serialize
import           Data.Singletons
import           GHC.Generics
import           Grenade

import           ML.BORL.Decay
import           ML.BORL.NeuralNetwork.Scaling
import           ML.BORL.Types


------------------------------ Replay Memory Setup ------------------------------

data ReplayMemoryStrategy
  = ReplayMemorySingle          -- ^ Use a single replay memory and store all experiences there.
  | ReplayMemoryPerAction       -- ^ Split experiences according to the chosen action and select `ceiling(batchsize/#actions)` experiences of each action.
  deriving (Show, Eq, Ord, Generic, NFData, Serialize)

------------------------------ NN Config ------------------------------

data NNConfig =
  NNConfig
    { _replayMemoryMaxSize             :: !Int                     -- ^ Maximum size of the replay memory.
    , _replayMemoryStrategy            :: !ReplayMemoryStrategy    -- ^ How to store experiences. @ReplayMemoryPerAction@ only works with n-step=1.
    , _trainBatchSize                  :: !Int                     -- ^ Batch size for training. Values are fed from the replay memory.
    , _grenadeLearningParams           :: !(Optimizer 'Adam)       -- ^ Grenade (not used for Tensorflow!) learning parameters.
    , _grenadeSmoothTargetUpdate       :: Rational                 -- ^ Rate of smooth updates of the target network.
    , _learningParamsDecay             :: !DecaySetup              -- ^ Decay setup for grenade learning parameters
    , _prettyPrintElems                :: ![NetInputWoAction]      -- ^ Sample input features for printing.
    , _scaleParameters                 :: !ScalingNetOutParameters -- ^ How to scale the output to the original range.
    , _stabilizationAdditionalRho      :: Float                    -- ^ Additional rho as a percantage of [minV, maxV] which is
                                                                   --   expected in the beginning.
    , _stabilizationAdditionalRhoDecay :: !DecaySetup              -- ^ Decay for stabilization
    , _updateTargetInterval            :: !Int                     -- ^ After how many steps should the target network be replaced by the worker?
    , _updateTargetIntervalDecay       :: !DecaySetup              -- ^ After how many steps should the target network be replaced by the worker?
    }
makeLenses ''NNConfig


instance NFData NNConfig where
  rnf (NNConfig rep repStrat tr !lp smooth dec pp sc stab stabDec up upDec) =
    rnf rep `seq` rnf repStrat `seq` rnf tr `seq` rnf lp `seq` rnf smooth `seq` rnf dec `seq` rnf pp `seq` rnf sc `seq` rnf stab `seq` rnf stabDec `seq` rnf up `seq` rnf upDec


setLearningRate :: Double -> Optimizer opt -> Optimizer opt
setLearningRate rate (OptSGD _ momentum l2) = OptSGD rate momentum l2
setLearningRate rate (OptAdam _ b1 b2 e)    = OptAdam rate b1 b2 e

getLearningRate :: Optimizer opt -> Double
getLearningRate (OptSGD rate _ _)   = rate
getLearningRate (OptAdam a b1 b2 e) = a
