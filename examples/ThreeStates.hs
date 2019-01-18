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
import           Data.Text                                      as T (Text, isPrefixOf)
import qualified Data.Vector                                    as V
import           Debug.Trace
import           GHC.Exts                                       (fromList)
import           GHC.Exts                                       (fromList)
import           GHC.Generics
import           Grenade                                        hiding (train)
import           Grenade                                        hiding (train)
import           System.IO.Temp
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
                                                                       relu, square, sub,
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
  , _trainMSEMax          = 0.05
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

type Output = Float
type Input = Float

data Model = Model
  { inputLayerName  :: Text
  , outputLayerName :: Text
  , allVariables    :: [TF.Tensor TF.Ref Float]
  , trainingNode    :: TF.ControlNode
  }

  -- , tensorTrain :: TF.ControlNode
  -- , train :: TF.ControlNode       -- ^ node (tensor)
  --         -> TF.TensorData Input  -- ^ images
  --         -> TF.TensorData Output -- ^ correct values
  --         -> TF.Session ()
  -- , infer :: TF.Tensor TF.Value Float     -- ^ tensor
  --         -> TF.TensorData Input          -- ^ images
  --         -> TF.Session (V.Vector Output) -- ^ predictions
  -- -- , errorRate :: TF.TensorData Input      -- ^ images
  -- --             -> TF.TensorData Output     -- ^ train values
  -- --             -> TF.Session Float

    -- , tensorTrain = trainStep
    -- , train = \trainStep imFeed lFeed -> TF.runWithFeeds_ [TF.feed images imFeed , TF.feed labels lFeed] trainStep
    -- , infer = \tensor imFeed -> TF.runWithFeeds [TF.feed images imFeed] tensor
    -- , errorRate = \imFeed lFeed -> TF.unScalar <$> TF.runWithFeeds [TF.feed images imFeed , TF.feed labels lFeed] errorRateTensor


batchSize :: Int64
batchSize = -1                  -- Use -1 batch size to support variable sized batches.

numInputs :: Int64
numInputs = 2

modelBuilder :: (TF.MonadBuild m) => m Model
modelBuilder = do

  -- Input layer.
  let inpLayerName = "input"
  input <- TF.placeholder' (TF.opName .~ TF.explicitName inpLayerName) (fromList [batchSize, numInputs])  -- Input layer.
  -- Hidden layer.
  let numUnits = 2
  hiddenWeights <- TF.initializedVariable' (TF.opName .~ "w1") =<< randomParam numInputs (fromList [numInputs, numUnits])
  hiddenBiases <- TF.zeroInitializedVariable' (TF.opName .~ "b1") (fromList [numUnits])
  let hiddenZ = (input `TF.matMul` hiddenWeights) `TF.add` hiddenBiases
  let hidden = TF.relu hiddenZ
  -- Logits
  outputWeights <- TF.initializedVariable' (TF.opName .~ "w2") =<< randomParam numInputs (fromList [numUnits, 1])
  outputBiases <- TF.zeroInitializedVariable' (TF.opName .~ "b2") (fromList [1])
  let outputs = (hidden `TF.matMul` outputWeights) `TF.add` outputBiases
  -- Output
  let outLayerName = "output"
  predictor <- TF.render $ TF.identity' (TF.opName .~ TF.explicitName outLayerName) $ TF.reduceMean $ TF.relu outputs

  -- Data Collection
  let weights = [hiddenWeights, hiddenBiases, outputWeights, outputBiases]

  -- Create training action.
  -- labels <- TF.placeholder [batchSize]
  -- let loss = TF.reduceSum $ TF.square (logits `TF.sub` labels)
  --     adamConfig = TF.AdamConfig { TF.adamLearningRate = 0.01 , TF.adamBeta1 = 0.9 , TF.adamBeta2 = 0.999 , TF.adamEpsilon = 1e-8 }
  -- trainStep <- TF.minimizeWith (TF.adam' adamConfig) loss vars

  -- , train = \trainStep imFeed lFeed -> TF.runWithFeeds_ [TF.feed images imFeed , TF.feed labels lFeed] trainStep

  -- let correctPredictions = TF.abs (predictor `TF.sub` labels) `TF.lessEqual` TF.scalar 0.01
  -- errorRateTensor <- TF.render $ 1 - TF.reduceMean (TF.cast correctPredictions)

  return Model
    { inputLayerName = inpLayerName
    , outputLayerName = outLayerName
    -- , trainingNode = trainStep
    , allVariables = weights
    }


-- | Convert a simple graph to GraphDef, load it, run it, and check the output.
testGraphDefExec :: IO ()
testGraphDefExec = do
    let graphDef = TF.asGraphDef $ TF.render $ TF.scalar (5 :: Float) * 10
    TF.runSession $ do
        TF.addGraphDef graphDef
        x <- TF.run $ TF.tensorValueFromName "Mul_2"
        liftIO $ print (TF.unScalar x :: Float)

tensorNameFilter :: Text -> Bool
tensorNameFilter x = not $ Prelude.or ([T.isPrefixOf "NoOp" x, T.isPrefixOf "AssignVariableOp" x] :: [Bool])

tensorFloat :: Text -> Bool
tensorFloat x = not $ Prelude.or ([tensorInt32 x ] :: [Bool])

tensorInt32 :: Text -> Bool
tensorInt32 x = Prelude.or ([T.isPrefixOf "Range" x] :: [Bool])


main :: IO ()
main = do

  let encodeImageBatch xs = TF.encodeTensorData [genericLength xs, 2] (V.fromList $ mconcat xs)
      encodeLabelBatch xs = TF.encodeTensorData [genericLength xs] (V.fromList xs)

  tempDir <- getCanonicalTemporaryDirectory >>= flip createTempDirectory ""
  print $ "TempDir: " ++ tempDir
  let path = B8.pack $ tempDir ++ "/model"
  -- testSaveRestore tempDir
  -- testGraphDefExec

  let graphDef = TF.asGraphDef modelBuilder
      namesPredictor = graphDef ^.. TF.node.traversed.TF.name

      -- tensorNames ::


      -- allTensors :: [TF.Tensor TF.Ref Float]
      -- allTensors = map TF.tensorFromName $ filter tensorNameFilter namesPredictor


  let outputTensor = head (graphDef ^. TF.node)
      outputTensorName = outputTensor ^. TF.name
      inputTensorName = last (graphDef ^. TF.node)^. TF.name
      -- outRef,inRef :: TF.Tensor TF.Ref Float
      -- outRef = TF.tensorFromName outputTensorName
      -- inRef = TF.tensorFromName inputTensorName

  -- putStrLn $ "outputTensorName: " ++ show outputTensorName
  putStrLn (show namesPredictor)


  let inp = encodeImageBatch [[0.7 :: Float,0.4]]
  -- SESSION 1
  TF.runSession $ do
    model <- modelBuilder
    -- TF.addGraphDef graphDef

    let inRef = TF.tensorFromName (inputLayerName model) :: TF.Tensor TF.Ref Float
        outRef = TF.tensorFromName (outputLayerName model) :: TF.Tensor TF.Ref Float
        -- vars = map TF.tensorFromName (allVariables model) :: [TF.Tensor TF.Ref Float]

    bef <- head . V.toList <$> TF.runWithFeeds [TF.feed inRef inp] outRef
    liftIO $ print bef
    TF.save path (allVariables model) >>= TF.run_ -- WithFeeds_ [TF.feed inRef inp]

  -- SESSION 2
  TF.runSession $ do
    model <- modelBuilder
    let inRef = TF.tensorFromName (inputLayerName model) :: TF.Tensor TF.Ref Float
        outRef = TF.tensorFromName (outputLayerName model) :: TF.Tensor TF.Ref Float


    mapM (TF.restore path) (allVariables model) >>= TF.runWithFeeds_ [TF.feed inRef (encodeImageBatch [[0::Float,0]])]

    -- forM_ ([0..1000] :: [Int]) $ \i -> do
      -- (x1Data :: [Float]) <- liftIO $ replicateM 1 randomIO
      -- (x2Data :: [Float]) <- liftIO $ replicateM 1 randomIO
      -- let xData = [[x1,x2] | x1 <- x1Data, x2 <- x2Data ]
    do
      let xData = [[0.7 :: Float, 0.4]]
      let yData = map (\(x1:x2:_) -> x1 * 0.3 + x2 * 0.5) xData

      let images = encodeImageBatch xData
          labels = encodeLabelBatch yData

      bef <- head . V.toList <$> TF.runWithFeeds [TF.feed inRef images] outRef
      liftIO $ print bef

      -- train model (tensorTrain model) images labels
      -- aft <- head . V.toList <$> infer model predictor images


      -- when (i `mod` 100 == 0) $ do
      --   liftIO $ putStrLn $ "Before vs After: " ++ show bef ++ " " ++ show aft ++ " [Actual: " ++ show (head yData) ++ "]"
      --   varVals :: [V.Vector Float] <- TF.run (TF.readValue <$> weights model)
      --   liftIO $ putStrLn $ "Weights: " ++ show (V.toList <$> varVals)

      --   err <- errorRate model images labels
      --   liftIO . putStrLn $ "training error " ++ show (err * 100)

  -- TF.runSession $ do
  --     let x1Data = [0,0.5,1]
  --     let x2Data = [0,0.5,1]
  --     let xData = [[x1,x2] | x1 <- x1Data, x2 <- x2Data ]
  --     let yData = map (\(x1:x2:_) -> x1 * 0.3 + x2 * 0.5) xData

  --     let images = encodeImageBatch xData

  --     bef <- head . V.toList <$> infer model images

  --     liftIO $ putStrLn $ "Vals: " ++ show bef ++ " [Actual: " ++ show (head yData) ++ "]"

  -- TF.runSession $ do
  --     model <- TF.build tensorflow
  --     let x1Data = [0,0.5,1]
  --     let x2Data = [0,0.5,1]
  --     let xData = [[x1,x2] | x1 <- x1Data, x2 <- x2Data ]
  --     let yData = map (\(x1:x2:_) -> x1 * 0.3 + x2 * 0.5) xData

  --     let images = encodeImageBatch xData

  --     bef <- head . V.toList <$> infer model images

  --     liftIO $ putStrLn $ "Vals: " ++ show bef ++ " [Actual: " ++ show (head yData) ++ "]"


  -- nn <- randomNetworkInitWith HeEtAl :: IO NN


  -- let rl = mkBORLUnichainGrenade initState actions actionFilter params decay nn nnConfig
  -- -- let rl = mkBORLUnichainTabular initState actions actionFilter params decay
  -- askUser True usage cmds rl   -- maybe increase learning by setting estimate of rho

  -- where cmds = []
  --       usage = []

testSaveRestore :: FilePath -> IO ()
testSaveRestore dirPath = do
  let path = B8.pack $ dirPath ++ "/checkpoint"
      var :: TF.MonadBuild m => m (TF.Tensor TF.Ref Float)
      var = TF.initializedVariable =<< randomParam numInputs (fromList [numInputs])
  TF.runSession $ do
    v <- var
    TF.assign v (TF.vector [134, 256]) >>= TF.run_
    TF.run v >>= \x -> liftIO (print (x :: V.Vector Float))
    TF.save path [v] >>= TF.run_
  (result :: V.Vector Float) <-
    TF.runSession $ do
      v <- var
      TF.restore path v >>= TF.run_
      TF.run v
  liftIO $ print result


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
  , _learnRandomAbove = 0.1
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
      (fromIntegral t / 20000) --  * zeta)
      (max 0 $ fromIntegral t / 40000) -- * xi)
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
