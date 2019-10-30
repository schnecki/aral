{-# LANGUAGE OverloadedLists   #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell   #-}
module ML.BORL.NeuralNetwork.TensorflowBuild
    ( buildModel
    , inputLayer1D
    , inputLayer
    , fullyConnected
    , trainingByAdam
    , trainingByAdamWith
    , randomParam
    ) where

import           ML.BORL.NeuralNetwork.Tensorflow

import           Control.Lens
import           Control.Monad                    (when)
import           Control.Monad.Trans.Class        (lift)
import           Control.Monad.Trans.State
import           Data.Int                         (Int32, Int64)
import           Data.Maybe                       (isJust, isNothing)
import           Data.String                      (fromString)
import           Data.Text                        (Text, pack, unpack)

import qualified TensorFlow.Build                 as TF (addNewOp, evalBuildT,
                                                         explicitName, opDef,
                                                         opDefWithName, opType, runBuildT,
                                                         summaries)
import qualified TensorFlow.Core                  as TF hiding (value)
-- import qualified TensorFlow.GenOps.Core                         as TF (square)
import qualified TensorFlow.GenOps.Core           as TF (abs, add, approximateEqual,
                                                         approximateEqual, assign, cast,
                                                         getSessionHandle,
                                                         getSessionTensor, identity',
                                                         lessEqual, matMul, mul,
                                                         readerSerializeState, relu, shape,
                                                         square, sub, tanh, tanh',
                                                         truncatedNormal)
import qualified TensorFlow.Minimize              as TF
-- import qualified TensorFlow.Ops                                 as TF (abs, add, assign,
--                                                                        cast, identity',
--                                                                        matMul, mul, relu,
--                                                                        sub,
--                                                                        truncatedNormal)
import qualified TensorFlow.BuildOp               as TF (OpParams)
import qualified TensorFlow.Ops                   as TF (initializedVariable,
                                                         initializedVariable', placeholder,
                                                         placeholder', reduceMean,
                                                         reduceSum, restore, save, scalar,
                                                         vector, zeroInitializedVariable,
                                                         zeroInitializedVariable')
import qualified TensorFlow.Tensor                as TF (Ref (..), collectAllSummaries,
                                                         tensorNodeName, tensorRefFromName,
                                                         tensorValueFromName)


data BuildInfo = BuildInfo
  { _inputName         :: Maybe Text
  , _outputName        :: Maybe Text
  , _labelName         :: Maybe Text
  , _maybeTrainingNode :: Maybe TF.ControlNode
  , _nnVars            :: [TF.Tensor TF.Ref Float]
  , _trainVars         :: [TF.Tensor TF.Ref Float]
  , _nrUnitsLayer      :: [[Int64]]
  , _lastTensor        :: Maybe (Int64, TF.Tensor TF.Value Float)
  , _nrLayers          :: Int
  }

makeLenses ''BuildInfo


batchSize :: Int64
batchSize = -1

inputTensorName :: Text
inputTensorName = "input"

layerIdxStartNr :: Int
layerIdxStartNr = 0

labLayerName :: Text
labLayerName = "labels"


inputLayer1D :: (TF.MonadBuild m) => Int64 -> StateT BuildInfo m ()
inputLayer1D numInputs = inputLayer [numInputs]

inputLayer :: (TF.MonadBuild m) => [Int64] -> StateT BuildInfo m ()
inputLayer shape = do
  let numInputs = product shape
  input <- lift $ TF.placeholder' (TF.opName .~ TF.explicitName inputTensorName) [batchSize, numInputs]
  lastTensor .= Just (numInputs, input)
  nrLayers .= layerIdxStartNr
  inputName .= Just inputTensorName


fullyConnected :: (TF.MonadBuild m) => [Int64] -> (TF.OpParams -> TF.Tensor TF.Build Float -> TF.Tensor TF.Build Float) -> StateT BuildInfo m ()
fullyConnected shape activationFunction = do
  layers <- gets (^. nrLayers)
  inputLayer <- gets (^. inputName)
  when (layers < layerIdxStartNr || isNothing inputLayer) $ error "You must start your model with an input layer"
  trainNode <- gets (^. maybeTrainingNode)
  when (isJust trainNode) $ error "You must create the NN before specifying the training nodes."
  let layerNr = layers + 1
  lastLayer <- gets (^. lastTensor)
  case lastLayer of
    Nothing -> error "No previous layer found on fullyConnected1D. Start with an input layer, see the `input1D` function"
    Just (previousNumUnits, previousTensor) -> do
      let numUnits = product shape
      hiddenWeights <- lift $ TF.initializedVariable' (TF.opName .~ fromString ("weights" ++ show layerNr)) =<< randomParam previousNumUnits [previousNumUnits, numUnits]
      hiddenBiases <- lift $ TF.zeroInitializedVariable' (TF.opName .~ fromString ("bias" ++ show layerNr)) [numUnits]
      let hiddenZ = (previousTensor `TF.matMul` hiddenWeights) `TF.add` hiddenBiases
      let outName = "out" <> pack (show layerNr)
      hidden <- lift $ TF.render $ activationFunction (TF.opName .~ TF.explicitName outName) hiddenZ
      nnVars %= (++ [hiddenWeights, hiddenBiases])
      lastTensor .= Just (numUnits, hidden)
      outputName .= Just outName
      nrLayers += 1
      nrUnitsLayer %=  (++ [[previousNumUnits, numUnits], [numUnits]])

trainingByAdam :: (TF.MonadBuild m) => StateT BuildInfo m ()
trainingByAdam = trainingByAdamWith TF.AdamConfig {TF.adamLearningRate = 0.01, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}


trainingByAdamWith :: (TF.MonadBuild m) => TF.AdamConfig -> StateT BuildInfo m ()
trainingByAdamWith adamConfig = do
  mOutput <- gets (^. outputName)
  when (isNothing mOutput) $ error "You must specify at least one layer, e.g. fullyConnected1D."
  lastLayer <- gets (^. lastTensor)
  -- Create training action.
  case lastLayer of
    Nothing -> error "No previous layer found in trainingByAdam. Start with an input layer, followed by at least one further layer."
    Just (_, previousTensor) -> do
      weights <- gets (^. nnVars)
      nrUnits <- gets (^. nrUnitsLayer)
      labels <- lift $ TF.placeholder' (TF.opName .~ TF.explicitName labLayerName) [batchSize]
      let loss = TF.square (previousTensor `TF.sub` labels)
            -- TF.reduceSum $ TF.square (previousTensor `TF.sub` labels)
      (trainStep, trVars) <- lift $ TF.minimizeWithRefs (TF.adamRefs' adamConfig) loss weights (map TF.Shape nrUnits)
      trainVars .= trVars
      maybeTrainingNode .= Just trainStep
      labelName .= Just labLayerName
      lastTensor .= Nothing

buildModel :: (TF.MonadBuild m) => StateT BuildInfo m () -> m TensorflowModel
buildModel builder = do
  buildInfo <- execStateT builder emptyBuildInfo
  case buildInfo of
    BuildInfo (Just inp) (Just out) (Just lab) (Just trainN) nnV trV _ _ _ -> return $ TensorflowModel inp out lab trainN nnV trV
    BuildInfo Nothing _ _ _ _ _ _ _ _ -> error "No input layer specified"
    BuildInfo _ Nothing _ _ _ _ _ _ _ -> error "No output layer specified"
    BuildInfo _ _ Nothing _ _ _ _ _ _ -> error "No training model specified"
    BuildInfo _ _ _ Nothing _ _ _ _ _  -> error "No training node specified (programming error in training action!)"

  where emptyBuildInfo = BuildInfo Nothing Nothing Nothing Nothing [] [] [] Nothing layerIdxStartNr

-- | Create tensor with random values where the stddev depends on the width.
randomParam :: (TF.MonadBuild m) => Int64 -> TF.Shape -> m (TF.Tensor TF.Build Float)
randomParam width (TF.Shape shape) = (`TF.mul` stddev) <$> TF.truncatedNormal (TF.vector shape)
  where
    stddev = TF.scalar (1 / sqrt (fromIntegral width))
