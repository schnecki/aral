{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE OverloadedStrings #-}
module ML.BORL.NeuralNetwork.Tensorflow where

import           Control.DeepSeq
import           Control.Monad.IO.Class (liftIO)
import qualified Data.ByteString.Char8  as B8
import           Data.Maybe             (isJust)
import           Data.Text              (Text)
import qualified Data.Vector            as V
import           System.IO.Temp
import qualified TensorFlow.Core        as TF
import qualified TensorFlow.GenOps.Core as TF (approximateEqual, lessEqual, square)
import qualified TensorFlow.Minimize    as TF
import qualified TensorFlow.Ops         as TF hiding (initializedVariable,
                                               zeroInitializedVariable)
import qualified TensorFlow.Variable    as TF

type Output = TF.TensorData Float
type Input = TF.TensorData Float


data TensorflowModel = TensorflowModel
  { inputLayerName         :: Text -- ^ Input layer name for feeding input.
  , outputLayerName        :: Text -- ^ Output layer name for predictions.
  , labelLayerName         :: Text -- ^ Labels input layer name for training.
  , errorRateName          :: Text -- ^ Error rate tensor name.
  , trainingNode           :: TF.ControlNode -- ^ Training node.
  , neuralNetworkVariables :: [TF.Tensor TF.Ref Float] -- ^ Neural network variables for saving and restoring.
  , trainingVariables      :: [TF.Tensor TF.Ref Float] -- ^ Training data/settings for saving and restoring.
  , checkpointBaseFileName :: Maybe FilePath
  }

instance NFData TensorflowModel where
  rnf (TensorflowModel i o l e !_ !_ !_ c) = rnf i `seq` rnf o `seq` rnf l `seq` rnf e `seq` rnf c


getRef :: Text -> TF.Tensor TF.Ref Float
getRef = TF.tensorFromName

modelName :: String
modelName = "model"

trainName :: String
trainName = "train"


saveModel :: TensorflowModel -> Input -> Output -> TF.Session TensorflowModel
saveModel model inp lab = do
  let tempDir = getCanonicalTemporaryDirectory >>= flip createTempDirectory ""
  basePath <- maybe (liftIO tempDir) return (checkpointBaseFileName model)
  let pathModel = B8.pack $ basePath ++ "/" ++ modelName
      pathTrain = B8.pack $ basePath ++ "/" ++ trainName
  let inRef = getRef (inputLayerName model)
      labRef = getRef (labelLayerName model)

  TF.save pathModel (neuralNetworkVariables model) >>= TF.run_
  TF.save pathTrain (trainingVariables model) >>= TF.runWithFeeds_ [TF.feed inRef inp, TF.feed labRef lab]
  return $ if isJust (checkpointBaseFileName model)
    then model
    else model { checkpointBaseFileName = Just basePath }

restoreModel :: TensorflowModel -> Input -> Output -> TF.Session ()
restoreModel model inp lab = do
  basePath <- maybe (error "cannot restore from unknown location: checkpointBaseFileName is Nothing") return (checkpointBaseFileName model)
  let pathModel = B8.pack $ basePath ++ "/" ++ modelName
      pathTrain = B8.pack $ basePath ++ "/" ++ trainName
  let inRef = getRef (inputLayerName model)
      labRef = getRef (labelLayerName model)
  mapM (TF.restore pathTrain) (trainingVariables model) >>= TF.runWithFeeds_ [TF.feed inRef inp, TF.feed labRef lab]
  mapM (TF.restore pathModel) (neuralNetworkVariables model) >>= TF.run_


