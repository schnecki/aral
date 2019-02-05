{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- This is example is a three-state MDP from Mahedevan 1996, Average Reward Reinforcement Learning - Foundations...
-- (Figure 2, p.166).

-- The provided solution is that a) the average reward rho=1 and b) the bias values are

-- when selection action a1 (A->B)
-- V(A) = 0.5
-- V(B) = -0.5
-- V(C) = 1.5

-- when selecting action a2 (A->C)
-- V(A) = -0.5
-- V(B) = -1.5
-- V(C) = 0.5

-- Thus the policy selecting a1 (going Left) is preferable.

module Main where

import           ML.BORL                                        hiding (actionFilter)

import           Helper

import           Control.DeepSeq                                (NFData)
import           Control.Lens
import           Control.Monad                                  (forM_, replicateM, when)
import           Control.Monad.IO.Class                         (liftIO)
import           Control.Monad.Reader
import           Data.ByteString                                (ByteString)
import qualified Data.ByteString                                as BS
import qualified Data.ByteString.Char8                          as B8
import           Data.Int                                       (Int32, Int64)
import           Data.Int                                       (Int32, Int64)
import           Data.List                                      (genericLength)
import           Data.Text                                      as T (Text, isInfixOf,
                                                                      isPrefixOf, unpack)
import qualified Data.Vector                                    as V
import           Debug.Trace
import           GHC.Exts                                       (fromList)
import           GHC.Exts                                       (fromList)
import           GHC.Generics
import           Grenade                                        hiding (train)
import           Grenade                                        hiding (train)
import           System.IO.Temp
import           System.Posix.Unistd                            (usleep)
import           System.Random                                  (randomIO)


import qualified Proto.Tensorflow.Core.Framework.Graph          as TF (GraphDef)
import qualified Proto.Tensorflow.Core.Framework.Graph_Fields   as TF (node)
import qualified Proto.Tensorflow.Core.Framework.NodeDef_Fields as TF (name, op, value)
import qualified TensorFlow.Build                               as TF (addNewOp,
                                                                       evalBuildT,
                                                                       explicitName, opDef,
                                                                       opDefWithName,
                                                                       opType, runBuildT,
                                                                       summaries)
import qualified TensorFlow.ControlFlow                         as TF (withControlDependencies)
import qualified TensorFlow.Core                                as TF hiding (value)
-- import qualified TensorFlow.GenOps.Core                         as TF (square)
import qualified TensorFlow.GenOps.Core                         as TF (abs, add,
                                                                       approximateEqual,
                                                                       approximateEqual,
                                                                       assign, cast,
                                                                       getSessionHandle,
                                                                       getSessionTensor,
                                                                       identity',
                                                                       lessEqual,
                                                                       lessEqual, matMul,
                                                                       mul,
                                                                       readerSerializeState,
                                                                       relu, shape, square,
                                                                       sub, tanh, tanh',
                                                                       truncatedNormal)
import qualified TensorFlow.Minimize                            as TF
import qualified TensorFlow.Nodes                               as TF (fetchTensorVector,
                                                                       getFetch, getNodes)
-- import qualified TensorFlow.Ops                                 as TF (abs, add, assign,
--                                                                        cast, identity',
--                                                                        matMul, mul, relu,
--                                                                        sub,
--                                                                        truncatedNormal)
import qualified TensorFlow.Ops                                 as TF (initializedVariable,
                                                                       initializedVariable',
                                                                       placeholder,
                                                                       placeholder',
                                                                       reduceMean,
                                                                       reduceSum, restore,
                                                                       save, scalar,
                                                                       vector,
                                                                       zeroInitializedVariable,
                                                                       zeroInitializedVariable')
import qualified TensorFlow.Tensor                              as TF (Ref (..),
                                                                       collectAllSummaries,
                                                                       tensorNodeName,
                                                                       tensorRefFromName,
                                                                       tensorValueFromName)
import qualified TensorFlow.Variable                            as TF (Variable, readValue)
import qualified TensorFlow.Variable                            as V (initializedVariable,
                                                                      initializedVariable',
                                                                      zeroInitializedVariable,
                                                                      zeroInitializedVariable')


type NN = Network '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 1, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 1, 'D1 1]

nnConfig :: NNConfig St
nnConfig = NNConfig
  { _toNetInp             = netInp
  , _replayMemory         = mkReplayMemory 10000
  , _trainBatchSize       = 1
  , _learningParams       = LearningParameters 0.005 0.0 0.0000
  , _prettyPrintElems     = [minBound .. maxBound] :: [St]
  , _scaleParameters      = scalingByMaxReward False 2
  , _updateTargetInterval = 5000
  , _trainMSEMax          = 0.015
  }


netInp :: St -> [Double]
netInp st = [scaleNegPosOne (minVal,maxVal) (fromIntegral $ fromEnum st)]

maxVal :: Double
maxVal = fromIntegral $ fromEnum (maxBound :: St)

minVal :: Double
minVal = fromIntegral $ fromEnum (minBound :: St)

-- | Create tensor with random values where the stddev depends on the width.
randomParam :: (TF.MonadBuild m) => Int64 -> TF.Shape -> m (TF.Tensor TF.Build Float)
randomParam width (TF.Shape shape) = (`TF.mul` stddev) <$> TF.truncatedNormal (TF.vector shape)
  where
    stddev = TF.scalar (1 / sqrt (fromIntegral width))

batchSize :: Int64
batchSize = -1                  -- Use -1 batch size to support variable sized batches.

numInputs :: Int64
numInputs = 2

modelBuilder :: (TF.MonadBuild m) => m TensorflowModel
modelBuilder = do

  -- Input layer.
  let inpLayerName = "input"
  input <- TF.placeholder' (TF.opName .~ TF.explicitName inpLayerName) (fromList [batchSize, numInputs])  -- Input layer.
  -- Hidden layer.
  let numUnits = 20
  hiddenWeights <- TF.initializedVariable' (TF.opName .~ "w1") =<< randomParam numInputs [numInputs, numUnits]
  hiddenBiases <- TF.zeroInitializedVariable' (TF.opName .~ "b1") [numUnits]
  let hiddenZ = (input `TF.matMul` hiddenWeights) `TF.add` hiddenBiases
  let hidden = TF.relu hiddenZ
  -- Logits
  outputWeights <- TF.initializedVariable' (TF.opName .~ "w2") =<< randomParam numInputs [numUnits, 1]
  outputBiases <- TF.zeroInitializedVariable' (TF.opName .~ "b2") [1]
  let outputs = (hidden `TF.matMul` outputWeights) `TF.add` outputBiases
  -- Output
  let outLayerName = "output"
  predictor <- TF.render $ TF.tanh' (TF.opName .~ TF.explicitName outLayerName) outputs

  -- Data Collection
  let weights = [hiddenWeights, hiddenBiases, outputWeights, outputBiases] :: [TF.Tensor TF.Ref Float]

  -- Create training action.
  let labLayerName = "labels"
  labels <- TF.placeholder' (TF.opName .~ TF.explicitName labLayerName) [batchSize]
  let loss = TF.reduceSum $ TF.square (predictor `TF.sub` labels)
      adamConfig = TF.AdamConfig { TF.adamLearningRate = 0.01 , TF.adamBeta1 = 0.9 , TF.adamBeta2 = 0.999 , TF.adamEpsilon = 1e-8 }
  (trainStep, trainVars) <- TF.minimizeWithRefs (TF.adamRefs' adamConfig) loss weights (map TF.Shape [[numInputs, numUnits], [numUnits], [numUnits,1],[1]])

  -- let correctPredictions = TF.abs (predictor `TF.sub` labels) `TF.lessEqual` TF.scalar 0.01
  -- let errRateName = "error"
  -- (_ :: TF.Tensor TF.Value Float) <- TF.render $ TF.identity' (TF.opName .~ TF.explicitName errRateName) $ 1 - TF.reduceMean (TF.cast correctPredictions)

  return TensorflowModel
    { inputLayerName = inpLayerName
    , outputLayerName = outLayerName
    , labelLayerName = labLayerName
    , trainingNode = trainStep
    , neuralNetworkVariables = weights
    , trainingVariables = trainVars
    }


main :: IO ()
main = do
  -- createModel >>= mapM_ testRun

  nn <- randomNetworkInitWith HeEtAl :: IO NN

  rl <- mkBORLUnichainGrenade initState actions actionFilter params decay nn nnConfig
  -- rl <- mkBORLUnichainTensorflow initState actions actionFilter params decay modelBuilder nnConfig
  -- let rl = mkBORLUnichainTabular initState actions actionFilter params decay
  askUser True usage cmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []


initState :: St
initState = A


-- | BORL Parameters.
params :: Parameters
params = Parameters
  { _alpha            = 0.2
  , _beta             = 0.25
  , _delta            = 0.25
  , _epsilon          = 1.0
  , _exploration      = 1.0
  , _learnRandomAbove = 0.0
  , _zeta             = 1.0
  , _xi               = 0.5
  }


-- | Decay function of parameters.
decay :: Period -> Parameters -> Parameters
decay t p@(Parameters alp bet del eps exp rand zeta xi)
  | t > 0 && t `mod` 200 == 0 =
    Parameters
      (max 0.0001 $ slow * alp)
      (f $ slower * bet)
      (f $ slower * del)
      (max 0.1 $ slow * eps)
      (f $ slow * exp)
      rand
      zeta -- (fromIntegral t / 20000) --  * zeta)
      xi -- (max 0 $ fromIntegral t / 40000) -- * xi)
  | otherwise = p
  where
    slower = 0.995
    slow = 0.95
    faster = 1.0 / 0.995
    f = max 0.001


-- State
data St = B | A | C deriving (Ord, Eq, Show, Enum, Bounded,NFData,Generic)
type R = Double
type P = Double

-- Actions
actions :: [Action St]
actions =
  [ Action moveLeft "left "
  , Action moveRight "right"]

actionFilter :: St -> [Bool]
actionFilter A = [True, True]
actionFilter B = [False, True]
actionFilter C = [True, False]


moveLeft :: St -> IO (Reward,St)
moveLeft s =
  return $
  case s of
    A -> (2, B)
    B -> (0, A)
    C -> (2, A)

moveRight :: St -> IO (Reward,St)
moveRight s =
  return $
  case s of
    A -> (0, C)
    B -> (0, A)
    C -> (2, A)


-- tensorNameFilter :: Text -> Bool
-- tensorNameFilter x = not $ Prelude.or ([T.isInfixOf "NoOp" x, T.isInfixOf "Save" x, T.isInfixOf "AssignVariableOp" x] :: [Bool])

-- tensorFloat :: Text -> Bool
-- tensorFloat x = not $ Prelude.or ([tensorInt32 x ] :: [Bool])

-- tensorInt32 :: Text -> Bool
-- tensorInt32 x = Prelude.or ([T.isInfixOf "Range" x] :: [Bool])

encodeImageBatch xs = TF.encodeTensorData [genericLength xs, 2] (V.fromList $ mconcat xs)
-- encodeLabelBatch xs = TF.encodeTensorData [genericLength xs] (V.fromList xs)

setCheckFile tempDir model = model { checkpointBaseFileName = Just tempDir }

prependName txt model = model { tensorflowModel = (tensorflowModel model)
        { inputLayerName = txt <> "/" <> (inputLayerName $ tensorflowModel model)
        , outputLayerName = txt <> "/" <> (outputLayerName $ tensorflowModel model)
        , labelLayerName = txt <> "/" <> (labelLayerName $ tensorflowModel model)
        }}


-- createModel :: IO [TensorflowModel']
-- createModel = do

--   let inp = encodeImageBatch [[0.7 :: Float,0.4]]
--       lab = encodeLabelBatch [0.41 :: Float]
--   -- SESSION 1

--   forM (["r1_target", "r1_worker"] :: [Text]) $ \scopeName -> TF.runSession $ do
--     let mkTensorflow' model = TensorflowModel' model Nothing Nothing (TF.withNameScope scopeName modelBuilder)
--     model <- setCheckFile ("/tmp/" <> T.unpack scopeName) . prependName scopeName . mkTensorflow' <$> TF.withNameScope scopeName modelBuilder
--     -- TF.addGraphDef graphDef
--     let inRef = TF.tensorFromName (inputLayerName $ tensorflowModel model) :: TF.Tensor TF.Ref Float
--         outRef = TF.tensorFromName (outputLayerName $ tensorflowModel model) :: TF.Tensor TF.Ref Float
--         labRef = TF.tensorFromName (labelLayerName $ tensorflowModel model) :: TF.Tensor TF.Ref Float

--     varVals :: [V.Vector Float] <- TF.runWithFeeds [TF.feed inRef inp, TF.feed labRef lab] (neuralNetworkVariables $ tensorflowModel model)
--     liftIO $ putStrLn $ "SESS 1 Weights: " ++ show (V.toList <$> varVals)
--     liftIO $ putStrLn $ "Saving model " <> T.unpack scopeName
--     saveModel model [[0,0]] [0]
--     liftIO $ putStrLn "Model saved"
--     return model

-- testRun :: TensorflowModel' -> IO ()
-- testRun model = do
--   -- print $ "TempDir: " ++ tempDir

--   let inp = encodeImageBatch [[0.7 :: Float,0.4]]
--       lab = encodeLabelBatch [0.41 :: Float]

--   let inRef = TF.tensorFromName (inputLayerName $ tensorflowModel model) :: TF.Tensor TF.Ref Float
--       outRef = TF.tensorFromName (outputLayerName $ tensorflowModel model) :: TF.Tensor TF.Ref Float
--       labRef = TF.tensorFromName (labelLayerName $ tensorflowModel model) :: TF.Tensor TF.Ref Float

--   TF.runSession $ do

--     -- Restore training config and weights afterwards as the first operations learns/modifies weights
--     restoreModel model [[0,0]] [0]
--     varVals :: [V.Vector Float] <- TF.run (neuralNetworkVariables $ tensorflowModel model)
--     liftIO $ putStrLn $ "SESS 2 Weights: " ++ show (V.toList <$> varVals)
--     bef <- head . V.toList <$> TF.runWithFeeds [TF.feed inRef inp] outRef
--     liftIO $ putStrLn $ "START SESS 2: " ++ show  bef

--     forM_ ([0..3000] :: [Int]) $ \i -> do
--       let bs = 1
--       (x1Data :: [Float]) <- liftIO $ replicateM bs randomIO
--       (x2Data :: [Float]) <- liftIO $ replicateM bs randomIO
--       let xData = [[x1,x2] | x1 <- x1Data, x2 <- x2Data ]
--       let yData = map (\(x1:x2:_) -> x1 * 0.3 + x2 * 0.5 + 0.0) xData
--       backwardRunSession model xData yData

--       when (i `mod` 100 == 0) $ do
--         res <- forwardRunSession model [[0.7, 0.4]]
--         liftIO $ putStrLn $ "Res/Value: " ++ show  res
--         varVals :: [V.Vector Float] <- TF.run (neuralNetworkVariables $ tensorflowModel model)
--         liftIO $ putStrLn $ "Weights 1: " ++ show (V.toList <$> varVals)
--         varVals :: [V.Vector Float] <- TF.run (neuralNetworkVariables $ tensorflowModel model)
--         liftIO $ putStrLn $ "Weights 2: " ++ show (V.toList <$> varVals)
--         res <- forwardRunSession model [[0.7, 0.4]]
--         liftIO $ putStrLn $ "Result: " ++ show res
--     saveModel model [[0.7,0.4]] [0.41]

--   -- Restore and read
--   TF.runSession $ do
--     restoreModel model [[0.7,0.4]] [0.41]
--     aft <- head . V.toList <$> TF.runWithFeeds [TF.feed inRef inp] outRef
--     res <- forwardRunSession model [[0.7, 0.4]]
--     liftIO $ putStrLn $ "END SESS 2: " ++ show  (aft, res)


