module ML.BORL.NeuralNetwork.Tensorflow where

import qualified Data.Vector            as V
import qualified TensorFlow.Core        as TF
import qualified TensorFlow.GenOps.Core as TF (approximateEqual, lessEqual, square)
import qualified TensorFlow.Minimize    as TF
import qualified TensorFlow.Ops         as TF hiding (initializedVariable,
                                               zeroInitializedVariable)
import qualified TensorFlow.Variable    as TF

type Output = Float
type Input = Float


data Model = Model
  { weights :: [TF.Variable Float] -- ^ Weights of the network
  , train :: TF.TensorData Input   -- ^ input
          -> TF.TensorData Output  -- ^ train values (desired output)
          -> TF.Session ()
  , infer :: TF.TensorData Input          -- ^ input
          -> TF.Session (V.Vector Output) -- ^ prediction
  , errorRate :: TF.TensorData Input      -- ^ input
              -> TF.TensorData Output     -- ^ train values (desired output)
              -> TF.Session Float
  }
