{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Strict              #-}
module ML.BORL.NeuralNetwork.Tensorflow where

import           Control.DeepSeq
import           Control.Monad                                  (unless, void, zipWithM)
import           Control.Monad.IO.Class                         (liftIO)
import qualified Data.ByteString.Char8                          as B8
import           Data.Int                                       (Int32, Int64)
import           Data.List                                      (genericLength)
import           Data.Maybe                                     (isJust)
import           Data.Text                                      (Text)
import qualified Data.Text                                      as T
import qualified Data.Vector                                    as V
import qualified Proto.Tensorflow.Core.Framework.Graph_Fields   as TF (node)
import qualified Proto.Tensorflow.Core.Framework.NodeDef_Fields as TF (name, op, value)
import           System.IO.Temp
import qualified TensorFlow.Core                                as TF
import qualified TensorFlow.GenOps.Core                         as TF (approximateEqual,
                                                                       lessEqual, square)
import qualified TensorFlow.Minimize                            as TF
import qualified TensorFlow.Ops                                 as TF hiding
                                                                       (initializedVariable,
                                                                       zeroInitializedVariable)
import qualified TensorFlow.Variable                            as TF hiding (assign)

import           ML.BORL.Types

type Output = [Float]
type Input = [[Float]]
type Labels = Output


data TensorflowModel = TensorflowModel
  { inputLayerName         :: Text -- ^ Input layer name for feeding input.
  , outputLayerName        :: Text -- ^ Output layer name for predictions.
  , labelLayerName         :: Text -- ^ Labels input layer name for training.
  , trainingNode           :: TF.ControlNode -- ^ Training node.
  , neuralNetworkVariables :: [TF.Tensor TF.Ref Float] -- ^ Neural network variables for saving and restoring.
  , trainingVariables      :: [TF.Tensor TF.Ref Float] -- ^ Training data/settings for saving and restoring.
  }

instance NFData TensorflowModel where
  rnf (TensorflowModel i o l !_ !_ !_) = rnf i `seq` rnf o `seq` rnf l

data TensorflowModel' = TensorflowModel'
  { tensorflowModel        :: TensorflowModel
  , checkpointBaseFileName :: Maybe FilePath
  , lastInputOutputTuple   :: Maybe ([Float], Float)
  , tensorflowModelBuilder :: TF.Session TensorflowModel
  }

instance NFData TensorflowModel' where
  rnf (TensorflowModel' m f l !_) = rnf m `seq` rnf f `seq` rnf l


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

forwardRun :: TensorflowModel' -> Input -> IO Output
forwardRun model inp =
  TF.runSession $ do
    restoreModelWithLastIO model
    forwardRunSession model inp


forwardRunSession :: TensorflowModel' -> Input -> TF.Session Output
forwardRunSession model inp = do
  let inRef = getRef (inputLayerName $ tensorflowModel model)
      outRef = getRef (outputLayerName $ tensorflowModel model)
      inpT = encodeInputBatch inp
  V.toList <$> TF.runWithFeeds [TF.feed inRef inpT] outRef


backwardRun :: TensorflowModel' -> Input -> Labels -> IO TensorflowModel'
backwardRun model inp lab
  | null inp || any null inp || null lab = error $ "Empty input in backwardRun not allowed! inp: " ++ show inp ++ ", lab: " ++ show lab
  | otherwise =
    TF.runSession $ do
      restoreModel model [head inp] [head lab]
      backwardRunSession model inp lab
      saveModel model [head inp] [head lab]

backwardRunSession :: TensorflowModel' -> Input -> Labels -> TF.Session ()
backwardRunSession model inp lab = do
  let inRef = getRef (inputLayerName $ tensorflowModel model)
      labRef = getRef (labelLayerName $ tensorflowModel model)
      outRef = getRef (outputLayerName $ tensorflowModel model)
      inpT = encodeInputBatch inp
      labT = encodeLabelBatch lab

  -- bef <- forwardRunSession model inp
  TF.runWithFeeds_ [TF.feed inRef inpT, TF.feed labRef labT] (trainingNode $ tensorflowModel model)
  -- aft <- forwardRunSession model inp
  -- liftIO $ putStrLn $ "Input/Output: " <> show inp <> " - " ++ show lab ++ "\tBefore/After: " <> show bef ++ " - " ++ show aft


copyValuesFromTo :: TensorflowModel' -> TensorflowModel' -> MonadBorl IO ()
copyValuesFromTo from to = do
  let fromVars = neuralNetworkVariables $ tensorflowModel from
      toVars = neuralNetworkVariables $ tensorflowModel to
  if length fromVars /= length toVars
    then error "cannot copy values to models with different length of neural network variables"
    else void $ do
    restoreModelWithLastIO to
    restoreModelWithLastIONoBuild from
    Tensorflow $ zipWithM TF.assign (neuralNetworkVariables $ tensorflowModel to) (neuralNetworkVariables $ tensorflowModel from) >>= TF.run_
    Tensorflow $ void $ saveModelWithLastIO from
    Tensorflow $ saveModelWithLastIO to


saveModelWithLastIO :: TensorflowModel' -> TF.Session TensorflowModel'
saveModelWithLastIO model =
  case lastInputOutputTuple model of
    Nothing     -> error "No last IO in saveModelWithLastIO"
    Just (i, o) -> saveModel model [i] [o]

saveModel :: TensorflowModel' -> Input -> Output -> TF.Session TensorflowModel'
saveModel model inp lab = do
  let tempDir = getCanonicalTemporaryDirectory >>= flip createTempDirectory ""
  basePath <- maybe (liftIO tempDir) return (checkpointBaseFileName model)
  let pathModel = B8.pack $ basePath ++ "/" ++ modelName
      pathTrain = B8.pack $ basePath ++ "/" ++ trainName
  let inRef = getRef (inputLayerName $ tensorflowModel model)
      labRef = getRef (labelLayerName $ tensorflowModel model)
  let inpT = encodeInputBatch inp
      labT = encodeLabelBatch lab
  let resetLastIO mdl = mdl {lastInputOutputTuple = Just (last inp, last lab)}
  unless (null $ neuralNetworkVariables $ tensorflowModel model) $ TF.save pathModel (neuralNetworkVariables $ tensorflowModel model) >>= TF.run_
  unless (null $ trainingVariables $ tensorflowModel model) $
    TF.save pathTrain (trainingVariables $ tensorflowModel model) >>= TF.runWithFeeds_ [TF.feed inRef inpT, TF.feed labRef labT]
  return $
    if isJust (checkpointBaseFileName model)
      then resetLastIO model
      else resetLastIO $ model {checkpointBaseFileName = Just basePath}

restoreModelWithLastIO :: TensorflowModel' -> MonadBorl IO ()
restoreModelWithLastIO model =
  case lastInputOutputTuple model of
    Nothing     -> error "No last IO in restoreModelWithLastIO"
    Just (i, o) -> restoreModel model [i] [o]

restoreModelWithLastIONoBuild :: TensorflowModel' -> MonadBorl IO ()
restoreModelWithLastIONoBuild model =
  case lastInputOutputTuple model of
    Nothing     -> error "No last IO in restoreModelWithLastIO"
    Just (i, o) -> restoreModelNoBuild model [i] [o]


restoreModel :: TensorflowModel' -> Input -> Output -> MonadBorl IO ()
restoreModel tfModel inp lab = do
  void $ Tensorflow $ tensorflowModelBuilder tfModel -- Build model (creates needed nodes)
  restoreModelNoBuild tfModel inp lab

restoreModelNoBuild :: TensorflowModel' -> Input -> Output -> MonadBorl IO ()
restoreModelNoBuild tfModel inp lab = Tensorflow $ do
  basePath <- maybe (error "cannot restore from unknown location: checkpointBaseFileName is Nothing") return (checkpointBaseFileName tfModel)
  let pathModel = B8.pack $ basePath ++ "/" ++ modelName
      pathTrain = B8.pack $ basePath ++ "/" ++ trainName
  let inRef = getRef (inputLayerName $ tensorflowModel tfModel)
      labRef = getRef (labelLayerName $ tensorflowModel tfModel)
  let inpT = encodeInputBatch inp
      labT = encodeLabelBatch lab
  unless (null $ trainingVariables $ tensorflowModel tfModel) $
    mapM (TF.restore pathTrain) (trainingVariables $ tensorflowModel tfModel) >>= TF.runWithFeeds_ [TF.feed inRef inpT, TF.feed labRef labT]
  unless (null $ neuralNetworkVariables $ tensorflowModel tfModel) $ mapM (TF.restore pathModel) (neuralNetworkVariables $ tensorflowModel tfModel) >>= TF.run_

