{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Strict              #-}
module ML.BORL.NeuralNetwork.Tensorflow where

import           Control.DeepSeq
import           Control.Monad         (unless, void, zipWithM)
import qualified Data.ByteString       as BS
import qualified Data.ByteString.Char8 as B8
import           Data.List             (genericLength)
import           Data.Maybe            (fromMaybe, isJust)
import           Data.Serialize
import           Data.Serialize.Text   ()
import           Data.Text             (Text)
import qualified Data.Vector           as V
import           System.Directory
import           System.IO.Temp
import           System.IO.Unsafe
import           System.Random


import qualified TensorFlow.Core       as TF
import qualified TensorFlow.Ops        as TF hiding (initializedVariable,
                                              zeroInitializedVariable)
import qualified TensorFlow.Output     as TF (ControlNode (..), NodeName (..))
import qualified TensorFlow.Tensor     as TF (tensorNodeName, tensorRefFromName)


import           ML.BORL.Types

import           Debug.Trace


type Outputs = [[Float]]
type Inputs = [[Float]]
type Labels = [[Float]]


data TensorflowModel = TensorflowModel
  { inputLayerName         :: Text                     -- ^ Input layer name for feeding input.
  , outputLayerName        :: Text                     -- ^ Output layer name for predictions.
  , labelLayerName         :: Text                     -- ^ Labels input layer name for training.
  , trainingNode           :: TF.ControlNode           -- ^ Training node.
  , neuralNetworkVariables :: [TF.Tensor TF.Ref Float] -- ^ Neural network variables for saving and restoring.
  , trainingVariables      :: [TF.Tensor TF.Ref Float] -- ^ Training data/settings for saving and restoring.
  }

instance NFData TensorflowModel where
  rnf (TensorflowModel i o l !_ !_ !_) = rnf i `seq` rnf o `seq` rnf l

data TensorflowModel' = TensorflowModel'
  { tensorflowModel        :: TensorflowModel
  , checkpointBaseFileName :: Maybe FilePath
  , lastInputOutputTuple   :: Maybe ([Float], [Float])
  , tensorflowModelBuilder :: TF.Session TensorflowModel
  }

instance Serialize TensorflowModel' where
  put tf@(TensorflowModel' (TensorflowModel inp out label train nnVars trVars) _ lastIO builder) = do
    put inp >> put out >> put label >> put (getTensorControlNodeName train) >> put lastIO
    put $ map getTensorRefNodeName nnVars
    put $ map getTensorRefNodeName trVars
    let (mBasePath, bytesModel, bytesTrain) =
          unsafePerformIO $ do
            let basePath = fromMaybe (error "cannot read tensorflow model") (checkpointBaseFileName tf) -- models have been saved during conversion
                pathModel = basePath ++ "/" ++ modelName
                pathTrain = basePath ++ "/" ++ trainName
            bModel <- liftSimple $ BS.readFile pathModel
            bTrain <- liftSimple $ BS.readFile pathTrain
            return (checkpointBaseFileName tf, bModel, bTrain)
    put mBasePath
    put bytesModel
    put bytesTrain
  get = do
    inp <- get
    out <- get
    label <- get
    train <- getControlNodeTensorFromName <$> get
    lastIO <- get
    nnVars <- map getRefTensorFromName <$> get
    trVars <- map getRefTensorFromName <$> get
    mBasePath <- get
    bytesModel <- get
    bytesTrain <- get
    return $ force $
      unsafePerformIO $ do
        basePath <- maybe (getCanonicalTemporaryDirectory >>= flip createTempDirectory "") (\b -> createDirectoryIfMissing True b >> return b) mBasePath
        let pathModel = basePath ++ "/" ++ modelName
            pathTrain = basePath ++ "/" ++ trainName
        BS.writeFile pathModel bytesModel
        BS.writeFile pathTrain bytesTrain
        let fakeBuilder = TF.runSession $ return $ TensorflowModel inp out label train nnVars trVars
        return $ TensorflowModel' (TensorflowModel inp out label train nnVars trVars) (Just basePath) lastIO fakeBuilder


instance NFData TensorflowModel' where
  rnf (TensorflowModel' m f l !_) = rnf m `seq` rnf f `seq` rnf l


trainMaxVal :: Float
trainMaxVal = 0.98

getRef :: Text -> TF.Tensor TF.Ref Float
getRef = TF.tensorFromName

modelName :: String
modelName = "model"

trainName :: String
trainName = "train"

getTensorControlNodeName :: TF.ControlNode -> Text
getTensorControlNodeName = TF.unNodeName . TF.unControlNode

getTensorRefNodeName :: TF.Tensor TF.Ref a -> Text
getTensorRefNodeName = TF.unNodeName . TF.tensorNodeName

getRefTensorFromName :: Text -> TF.Tensor TF.Ref a
getRefTensorFromName = TF.tensorRefFromName

getControlNodeTensorFromName :: Text -> TF.ControlNode
getControlNodeTensorFromName = TF.ControlNode . TF.NodeName

encodeInputBatch :: Inputs -> TF.TensorData Float
encodeInputBatch xs = TF.encodeTensorData [genericLength xs, genericLength (head xs)] (V.fromList $ mconcat xs)

encodeLabelBatch :: Labels -> TF.TensorData Float
encodeLabelBatch xs = TF.encodeTensorData [genericLength xs, genericLength (head xs)] (V.fromList $ mconcat xs)

forwardRun :: (MonadBorl' m) => TensorflowModel' -> Inputs -> m Outputs
forwardRun model inp =
  liftTf $
  let inRef = getRef (inputLayerName $ tensorflowModel model)
      outRef = getRef (outputLayerName $ tensorflowModel model)
      inpT = encodeInputBatch inp
      nrOuts = length inp
   in do res <- V.toList <$> TF.runWithFeeds [TF.feed inRef inpT] outRef
         return $ separate (length res `div` nrOuts) res []
  where
    separate _ [] acc = reverse acc
    separate len xs acc
      | length xs < len = error $ "error in separate (in Tensorflow.forwardRun), not enough values: " ++ show xs ++ " - len: " ++ show len
      | otherwise = separate len (drop len xs) (take len xs : acc)

backwardRunRepMemData :: (MonadBorl' m) => TensorflowModel' -> [(([Double], ActionIndex), Double)] -> m ()
backwardRunRepMemData model values = do
  let inputs = map (map realToFrac.fst.fst) values
  outputs <- forwardRun model inputs
  let minmax = max (-trainMaxVal) . min trainMaxVal
  let labels = zipWith (\((_,idx), val) outp -> replace idx (minmax $ realToFrac val) outp) values outputs
  -- Simple $ zipWithM_ (\(((_,idx),val), o) (inp,l) -> putStrLn $ show idx ++ ": " ++ show inp ++ " \tout: " ++ show o ++ " l: " ++ show l ++ " val: " ++ show val) (zip values outputs) (zip inputs labels)
  backwardRun model inputs labels

-- | Train tensorflow model with checks.
backwardRun :: (MonadBorl' m) => TensorflowModel' -> Inputs -> Labels -> m ()
backwardRun model inp lab
  | null inp || any null inp || null lab = error $ "Empty input in backwardRun not allowed! inp: " ++ show inp ++ ", lab: " ++ show lab
  | otherwise =
    let inRef = getRef (inputLayerName $ tensorflowModel model)
        labRef = getRef (labelLayerName $ tensorflowModel model)
        inpT = encodeInputBatch inp
        labT = encodeLabelBatch lab
    in liftTf $ TF.runWithFeeds_ [TF.feed inRef inpT, TF.feed labRef labT] (trainingNode $ tensorflowModel model)
       -- aft <- forwardRunSession model inp
       -- Simple $ putStrLn $ "Input/Output: " <> show inp <> " - " ++ show lab ++ "\tBefore/After: " <> show bef ++ " - " ++ show aft

-- | Copies values from one model to the other.
copyValuesFromTo :: (MonadBorl' m) => TensorflowModel' -> TensorflowModel' -> m ()
copyValuesFromTo from to = do
  let fromVars = neuralNetworkVariables $ tensorflowModel from
      toVars = neuralNetworkVariables $ tensorflowModel to
  if length fromVars /= length toVars
    then error "cannot copy values to models with different length of neural network variables"
    else void $ liftTf $ zipWithM TF.assign toVars fromVars >>= TF.run_


saveModelWithLastIO :: (MonadBorl' m) => TensorflowModel' -> m TensorflowModel'
saveModelWithLastIO model =
  case lastInputOutputTuple model of
    Nothing     -> error "No last IO in saveModelWithLastIO"
    Just (i, o) -> saveModel model [i] [o]

saveModel :: (MonadBorl' m) => TensorflowModel' -> Inputs -> Labels -> m TensorflowModel'
saveModel model inp lab = do
  let tempDir = getCanonicalTemporaryDirectory >>= flip createTempDirectory ""
  basePath <- maybe (liftSimple tempDir) return (checkpointBaseFileName model)
  let pathModel = B8.pack $ basePath ++ "/" ++ modelName
      pathTrain = B8.pack $ basePath ++ "/" ++ trainName
  let inRef = getRef (inputLayerName $ tensorflowModel model)
      labRef = getRef (labelLayerName $ tensorflowModel model)
  let inpT = encodeInputBatch inp
      labT = encodeLabelBatch lab
  let resetLastIO mdl = mdl {lastInputOutputTuple = Just (last inp, last lab)}
  unless (null $ neuralNetworkVariables $ tensorflowModel model) $
    liftTf $ TF.save pathModel (neuralNetworkVariables $ tensorflowModel model) >>= TF.run_
  unless (null $ trainingVariables $ tensorflowModel model) $
    liftTf $ TF.save pathTrain (trainingVariables $ tensorflowModel model) >>= TF.runWithFeeds_ [TF.feed inRef inpT, TF.feed labRef labT]
  -- res <- map V.toList <$> (Tensorflow $ TF.runWithFeeds [TF.feed inRef inpT, TF.feed labRef labT] (trainingVariables $ tensorflowModel model))
  -- Simple $ putStrLn $ "Training variables (saveModel): " <> show res

  return $
    if isJust (checkpointBaseFileName model)
      then resetLastIO model
      else resetLastIO $ model {checkpointBaseFileName = Just basePath}

restoreModelWithLastIO :: (MonadBorl' m) => TensorflowModel' -> m ()
restoreModelWithLastIO model =
  case lastInputOutputTuple model of
    Nothing     -> error "No last IO in restoreModelWithLastIO"
    Just (i, o) -> restoreModel model [i] [o]

buildTensorflowModel :: (MonadBorl' m) => TensorflowModel' -> m ()
buildTensorflowModel tfModel = void $ liftTf $ tensorflowModelBuilder tfModel -- Build model (creates needed nodes)


restoreModel :: (MonadBorl' m) => TensorflowModel' -> Inputs -> Labels -> m ()
restoreModel tfModel inp lab = liftTf $ do
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

  -- res <- map V.toList <$> TF.runWithFeeds [TF.feed inRef inpT, TF.feed labRef labT] (trainingVariables $ tensorflowModel tfModel)
  -- liftIO $ putStrLn $ "Training variables (restoreModel): " <> show res


