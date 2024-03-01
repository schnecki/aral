{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE DataKinds       #-}
{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE GADTs           #-}
{-# LANGUAGE KindSignatures  #-}
{-# LANGUAGE RankNTypes      #-}
{-# LANGUAGE TemplateHaskell #-}
module ML.ARAL.NeuralNetwork.NNConfig where

import           Control.DeepSeq
import           Control.Lens
import           Data.Serialize
import           Data.Singletons
import           GHC.Generics

import           ML.ARAL.Decay
import           ML.ARAL.NeuralNetwork.Scaling
import           ML.ARAL.Types


------------------------------ Replay Memory Setup ------------------------------

data ReplayMemoryStrategy
  = ReplayMemorySingle          -- ^ Use a single replay memory and store all experiences there.
  | ReplayMemoryPerAction       -- ^ Split experiences according to the chosen action and select `ceiling(batchsize/#actions)` experiences of each action. For multiple agents the agent that determines
                                -- the action is selected randomly.
  deriving (Show, Eq, Ord, Generic, NFData, Serialize)

data Clipping = NoClipping              -- ^ No Clipping
              | ClipByGlobalNorm Double -- ^ Specifies the maximum norm of the gradient on the normed values (-1,1).
              | ClipByValue Double      -- ^ Clip the gradients by this exact value for the normed values in range (-1,1).
  deriving (Show, Eq, Ord, Generic, NFData, Serialize)

------------------------------ NN Config ------------------------------

-- | The available optimizers to choose from. If an optimiser is not implemented for a layer SGD
--   with the default settings (see @Default 'SGD@ instance) will be used instead. This is default
--   and thus fallback optimizer.
--
--  Concreate instance for the optimizers.
data Optimizer =
  OptAdam
    { adamAlpha             :: !Double -- ^ Alpha [Default: 0.001]
    , adamBeta1             :: !Double -- ^ Beta 1 [Default: 0.9]
    , adamBeta2             :: !Double -- ^ Beta 2 [Default: 0.999]
    , adamEpsilon           :: !Double -- ^ Epsilon [Default: 1e-7]
    , adamWeightDecayLambda :: !Double -- ^ Weight decay to use [Default: 0.001]
    } deriving (Generic, Serialize)


instance Show Optimizer where
  show (OptAdam alpha beta1 beta2 epsilon wD) = "Adam" ++ show (alpha, beta1, beta2, epsilon, wD)

instance NFData Optimizer where
  rnf (OptAdam alpha beta1 beta2 epsilon wD) = rnf alpha `seq` rnf beta1 `seq` rnf beta2 `seq` rnf epsilon `seq` rnf wD

instance Eq Optimizer where
  (OptAdam al1 b11 b21 eps1 lmd1) == (OptAdam al2 b12 b22 eps2 lmd2) = (al1, b11, b21, eps1, lmd1) == (al2, b12, b22, eps2, lmd2)

instance Ord Optimizer where
  (OptAdam al1 b11 b21 eps1 lmd1) `compare` (OptAdam al2 b12 b22 eps2 lmd2) = (al1, b11, b21, eps1, lmd1) `compare` (al2, b12, b22, eps2, lmd2)


data NNConfig =
  NNConfig
    { _replayMemoryMaxSize             :: !Int                     -- ^ Maximum size of the replay memory. If you set this to `trainBatchSize * nStep` then there is no random selection, but all
                                                                   -- memories are used! This size if for one agent, if you use more than one, then the size will be scaled up.
    , _replayMemoryStrategy            :: !ReplayMemoryStrategy    -- ^ How to store experiences. @ReplayMemoryPerAction@ only works with n-step=1.
    , _trainBatchSize                  :: !Int                     -- ^ Batch size of each worker/of the main agent for training. Values are fed from the replay memory. Thus, resulting number of
                                                                   -- batchsize = #workers * trainBatchSize.
    , _trainingIterations              :: !Int                     -- ^ How often to repeat the training with the same gradients in each step.
    , _grenadeLearningParams           :: !Optimizer               -- ^ Learning parameters.
    , _grenadeSmoothTargetUpdate       :: !Double                  -- ^ Rate of smooth updates of the target network. Set 0 to use hard updates using @_updateTargetInterval@.
    , _grenadeSmoothTargetUpdatePeriod :: !Int                     -- ^ Every x periods the smooth update will take place.
    , _learningParamsDecay             :: !DecaySetup              -- ^ Decay setup for grenade learning parameters
    , _prettyPrintElems                :: ![NetInputWoAction]      -- ^ Sample input features for printing.
    , _scaleParameters                 :: !ScalingNetOutParameters -- ^ How to scale the output to the original range.
    , _scaleOutputAlgorithm            :: !ScalingAlgorithm        -- ^ What algorithm to use for scaling. Usually @ScaleMinMax@ is a good value and @ScaleLog@ might be interesting for minimization problem.
    , _cropTrainMaxValScaled           :: !(Maybe Double)           -- ^ Crop the min and max of the learned scaled values, e.g. Just 0.98 -> Crops all values to (-0.98, 0.98) prior to learning. Useful
                                                                   -- when using Tanh as output activation. Currently for Grenade only (as this part is in the sublibrary higher-level-tensorflow)!
    , _grenadeDropoutFlipActivePeriod  :: !Int                     -- ^ Flip dropout active/inactive state every X periods.
    , _grenadeDropoutOnlyInactiveAfter :: !Int                     -- ^ Keep dropout inactive when reaching the given number of periods. Set to 0 to inactive dropout active state flipping!
    , _clipGradients                   :: !Clipping                -- ^ Clip the gradients (takes time, but is a safer update).
    , _autoNormaliseInput              :: !Bool                    -- ^ Automatically normalize the input
    } deriving (Show)
makeLenses ''NNConfig


instance NFData NNConfig where
  rnf (NNConfig rep repStrat batchsize tr !lp smooth smoothPer dec pp sc scalg crop dropFlip dropInactive clip autoNorm) =
    rnf rep `seq` rnf repStrat `seq` rnf batchsize `seq` rnf tr `seq` rnf lp `seq` rnf smooth `seq`
    rnf smoothPer `seq` rnf dec `seq` rnf pp `seq` rnf sc `seq` rnf scalg `seq` rnf crop `seq` rnf dropFlip
    `seq` rnf dropInactive `seq` rnf clip `seq` rnf autoNorm


setLearningRate :: Double -> Optimizer -> Optimizer
setLearningRate rate (OptAdam _ b1 b2 e w)  = OptAdam rate b1 b2 e w

getLearningRate :: Optimizer -> Double
getLearningRate (OptAdam a _ _ _ _ ) = a
