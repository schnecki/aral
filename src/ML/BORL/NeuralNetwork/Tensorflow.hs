{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE OverloadedLists   #-}
{-# LANGUAGE OverloadedStrings #-}
module ML.BORL.NeuralNetwork.Tensorflow where

import           Control.DeepSeq
import           Control.Monad.IO.Class (liftIO)
import qualified Data.ByteString.Char8  as B8
import           Data.List              (genericLength)
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

type Output = [Float]
type Input = [[Float]]
type Labels = Output


data TensorflowModel = TensorflowModel
  { inputLayerName         :: Text -- ^ Input layer name for feeding input.
  , outputLayerName        :: Text -- ^ Output layer name for predictions.
  , labelLayerName         :: Text -- ^ Labels input layer name for training.
  , errorRateName          :: Text -- ^ Error rate tensor name.
  , trainingNode           :: TF.ControlNode -- ^ Training node.
  , neuralNetworkVariables :: [TF.Tensor TF.Ref Float] -- ^ Neural network variables for saving and restoring.
  , trainingVariables      :: [TF.Tensor TF.Ref Float] -- ^ Training data/settings for saving and restoring.
  , checkpointBaseFileName :: Maybe FilePath
  , lastInputOutputTuple   :: Maybe ([Float], Float)
  }

instance NFData TensorflowModel where
  rnf (TensorflowModel i o l e !_ !_ !_ c m) = rnf i `seq` rnf o `seq` rnf l `seq` rnf e `seq` rnf c `seq` rnf m


getRef :: Text -> TF.Tensor TF.Ref Float
getRef = TF.tensorFromName

modelName :: String
modelName = "model"

trainName :: String
trainName = "train"


encodeInputBatch :: Input -> TF.TensorData Float
encodeInputBatch xs = TF.encodeTensorData [genericLength xs, 2] (V.fromList $ mconcat xs)

encodeLabelBatch :: Output -> TF.TensorData Float
encodeLabelBatch xs = TF.encodeTensorData [genericLength xs] (V.fromList xs)

forwardRun :: TensorflowModel -> Input -> IO Output
forwardRun model inp = TF.runSession $ do
  maybe (error "empty input output in lastInputOutputTuple, cannot restore model") (\(i,o) -> restoreModel model [i] [o]) (lastInputOutputTuple model)
  let inRef = getRef (inputLayerName model)
      outRef = getRef (outputLayerName model)
      inpT = encodeInputBatch inp
  V.toList <$> TF.runWithFeeds [TF.feed inRef inpT] outRef


backwardRun :: TensorflowModel -> Input -> Labels -> IO TensorflowModel
backwardRun model inp lab = TF.runSession $ do
  let inRef = getRef (inputLayerName model)
      labRef = getRef (labelLayerName model)
      inpT = encodeInputBatch inp
      labT = encodeLabelBatch lab
      resetLastIO mdl = mdl { lastInputOutputTuple = Just (last inp, last lab)}
  restoreModel model [head inp] [head lab]
  TF.runWithFeeds_ [TF.feed inRef inpT, TF.feed labRef labT] (trainingNode model)
  resetLastIO <$> saveModel model [head inp] [head lab]


saveModel :: TensorflowModel -> Input -> Output -> TF.Session TensorflowModel
saveModel model inp lab = do
  let tempDir = getCanonicalTemporaryDirectory >>= flip createTempDirectory ""
  basePath <- maybe (liftIO tempDir) return (checkpointBaseFileName model)
  let pathModel = B8.pack $ basePath ++ "/" ++ modelName
      pathTrain = B8.pack $ basePath ++ "/" ++ trainName
  let inRef = getRef (inputLayerName model)
      labRef = getRef (labelLayerName model)
  let inpT = encodeInputBatch inp
      labT = encodeLabelBatch lab

  TF.save pathModel (neuralNetworkVariables model) >>= TF.run_
  TF.save pathTrain (trainingVariables model) >>= TF.runWithFeeds_ [TF.feed inRef inpT, TF.feed labRef labT]
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
  let inpT = encodeInputBatch inp
      labT = encodeLabelBatch lab
  mapM (TF.restore pathTrain) (trainingVariables model) >>= TF.runWithFeeds_ [TF.feed inRef inpT, TF.feed labRef labT]
  mapM (TF.restore pathModel) (neuralNetworkVariables model) >>= TF.run_


