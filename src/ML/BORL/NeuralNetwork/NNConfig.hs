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
  | ReplayMemoryPerAction       -- ^ Split experiences according to the chosen action and select `ceiling(batchsize/#actions)` experiences of each action. For multiple agents the agent that determines
                                -- the action is selected randomly.
  deriving (Show, Eq, Ord, Generic, NFData, Serialize)

------------------------------ NN Config ------------------------------

data NNConfig =
  NNConfig
    { _replayMemoryMaxSize             :: !Int                     -- ^ Maximum size of the replay memory. If you set this to `trainBatchSize * nStep` then there is no random selection, but all
                                                                   -- memories are used! This size if for one agent, if you use more than one, then the size will be scaled up.
    , _replayMemoryStrategy            :: !ReplayMemoryStrategy    -- ^ How to store experiences. @ReplayMemoryPerAction@ only works with n-step=1.
    , _trainBatchSize                  :: !Int                     -- ^ Batch size for training. Values are fed from the replay memory.
    , _trainingIterations              :: !Int                     -- ^ How often to repeat the training with the same gradients in each step.
    , _grenadeLearningParams           :: !(Optimizer 'Adam)       -- ^ Grenade (not used for Tensorflow!) learning parameters.
    , _grenadeSmoothTargetUpdate       :: !Double                -- ^ Rate of smooth updates of the target network. Set 0 to use hard updates using @_updateTargetInterval@.
    , _grenadeSmoothTargetUpdatePeriod :: !Int                     -- ^ Every x periods the smooth update will take place.
    , _learningParamsDecay             :: !DecaySetup              -- ^ Decay setup for grenade learning parameters
    , _prettyPrintElems                :: ![NetInputWoAction]      -- ^ Sample input features for printing.
    , _scaleParameters                 :: !ScalingNetOutParameters -- ^ How to scale the output to the original range.
    , _scaleOutputAlgorithm            :: !ScalingAlgorithm        -- ^ What algorithm to use for scaling. Usually @ScaleMinMax@ is a good value and @ScaleLog@ might be interesting for minimization problem.
    , _cropTrainMaxValScaled           :: !(Maybe Double)           -- ^ Crop the min and max of the learned scaled values, e.g. Just 0.98 -> Crops all values to (-0.98, 0.98) prior to learning. Useful
                                                                   -- when using Tanh as output activation. Currently for Grenade only (as this part is in the sublibrary higher-level-tensorflow)!
    , _grenadeDropoutFlipActivePeriod  :: !Int                     -- ^ Flip dropout active/inactive state every X periods.
    , _grenadeDropoutOnlyInactiveAfter :: !Int                     -- ^ Keep dropout inactive when reaching the given number of periods. Set to 0 to inactive dropout active state flipping!
    , _clipGradients                   :: Bool                     -- ^ Clip the gradients (takes time, but is a safer update). The amount is deduced by the min-max and the global norm.
    } deriving (Show)
makeLenses ''NNConfig


instance NFData NNConfig where
  rnf (NNConfig rep repStrat batchsize tr !lp smooth smoothPer dec pp sc scalg crop dropFlip dropInactive clip) =
    rnf rep `seq` rnf repStrat `seq` rnf batchsize `seq`
    rnf tr `seq` rnf lp `seq` rnf smooth `seq` rnf smoothPer `seq` rnf dec `seq` rnf pp `seq` rnf sc `seq` rnf scalg `seq` rnf crop `seq` rnf dropFlip `seq` rnf dropInactive `seq` rnf clip


setLearningRate :: Double -> Optimizer opt -> Optimizer opt
setLearningRate rate (OptSGD _ momentum l2) = OptSGD rate momentum l2
setLearningRate rate (OptAdam _ b1 b2 e w)  = OptAdam rate b1 b2 e w

getLearningRate :: Optimizer opt -> Double
getLearningRate (OptSGD rate _ _)    = rate
getLearningRate (OptAdam a _ _ _ _ ) = a
