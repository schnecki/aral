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
import           Data.Int                                       (Int32, Int64)
import           Data.Int                                       (Int32, Int64)
import           Data.List                                      (genericLength)
import qualified Data.Vector                                    as V
import qualified Data.Vector                                    as V
import           Debug.Trace
import           GHC.Exts                                       (fromList)
import           GHC.Exts                                       (fromList)
import           GHC.Generics
import           Grenade                                        hiding (train)
import           Grenade                                        hiding (train)
import qualified Proto.Tensorflow.Core.Framework.Graph_Fields   as TF (node)
import qualified Proto.Tensorflow.Core.Framework.NodeDef_Fields as TF (name, op, value)
import qualified Proto.Tensorflow.Core.Protobuf.Saver           as TF
import           System.Random                                  (randomIO)
import           System.Random                                  (randomIO)
import qualified TensorFlow.Build                               as TF (addNewOp,
                                                                       explicitName, opDef,
                                                                       opDefWithName,
                                                                       opType)
import qualified TensorFlow.ControlFlow                         as TF (withControlDependencies)
import qualified TensorFlow.Core                                as TF hiding (value)
import qualified TensorFlow.Core                                as TF
import qualified TensorFlow.GenOps.Core                         as TF (approximateEqual,
                                                                       lessEqual, square)
import qualified TensorFlow.GenOps.Core                         as TF (approximateEqual,
                                                                       getSessionHandle,
                                                                       getSessionTensor,
                                                                       lessEqual,
                                                                       readerSerializeState,
                                                                       square)
import qualified TensorFlow.Minimize                            as TF
import qualified TensorFlow.Minimize                            as TF
import qualified TensorFlow.Nodes                               as TF (fetchTensorVector,
                                                                       getFetch, getNodes)
import qualified TensorFlow.Ops                                 as TF hiding
                                                                       (initializedVariable,
                                                                       zeroInitializedVariable)
import qualified TensorFlow.Ops                                 as TF hiding
                                                                       (initializedVariable,
                                                                       zeroInitializedVariable)
import qualified TensorFlow.Tensor                              as TF (collectAllSummaries,
                                                                       tensorNodeName,
                                                                       tensorValueFromName)
import qualified TensorFlow.Tensor                              as TF (Ref (..),
                                                                       collectAllSummaries,
                                                                       tensorValueFromName)
import           TensorFlow.Variable                            as TF hiding (zeroInitializedVariable')
import qualified TensorFlow.Variable                            as V
import qualified TensorFlow.Variable                            as TF

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
randomParam :: Int64 -> TF.Shape -> TF.Build (TF.Tensor TF.Build Float)
randomParam width (TF.Shape shape) = (`TF.mul` stddev) <$> TF.truncatedNormal (TF.vector shape)
  where
    stddev = TF.scalar (1 / sqrt (fromIntegral width))

type Output = Float
type Input = Float

data Model = Model
  {
  allWeights      :: [TF.Variable Float]
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
  , tensorPredict :: TF.Tensor TF.Value Float
  -- , tensorTrain :: TF.ControlNode
  }

batchSize :: Int64
batchSize = -1                  -- Use -1 batch size to support variable sized batches.

numInputs :: Int64
numInputs = 2

predictorAndWeights :: TF.Build (TF.Tensor TF.Value Float, [TF.Variable Float])
predictorAndWeights = do
  -- input <- TF.placeholder (fromList [batchSize, numInputs])  --
  -- nodename <- TF.addNewOp $
  -- let ops = set TF.opName (TF.explicitName "asdf")
  -- predictor <- do
  --   let ret = TF.scalar' ops (5 :: Float) * 10
  --   -- let tp = ret ^. TF.node
  --   -- let tp = undefined
  --   -- TF.addNewOp $ TF.opDefWithName (TF.explicitName "asdf") tp
  --   TF.render ret
  -- let weights = []

  -- Input layer.
  images <- TF.placeholder (fromList [batchSize, numInputs])  -- Input layer.
            -- Hidden layer.
  let numUnits = 2
  hiddenWeights <- TF.initializedVariable =<< randomParam numInputs (fromList [numInputs, numUnits])
  hiddenBiases <- TF.zeroInitializedVariable (fromList [numUnits])
  let hiddenZ = (images `TF.matMul` TF.readValue hiddenWeights) `TF.add` TF.readValue hiddenBiases
  let hidden = TF.relu hiddenZ
               -- Logits
  logitWeights <- TF.initializedVariable =<< randomParam numInputs (fromList [numUnits, 1])
  logitBiases <- TF.zeroInitializedVariable (fromList [1])
  let logits = (hidden `TF.matMul` TF.readValue logitWeights) `TF.add` TF.readValue logitBiases
  let weights = [hiddenWeights, hiddenBiases, logitWeights, logitBiases]
  predictor <- TF.render $ TF.reduceMean $ TF.relu logits

  return (predictor, weights)


modelBuilder :: TF.Build Model
modelBuilder = do

  -- Create training action.

  -- labels <- TF.placeholder [batchSize]
  -- let loss = TF.reduceSum $ TF.square (logits `TF.sub` labels)
  --     adamConfig = TF.AdamConfig { TF.adamLearningRate = 0.01 , TF.adamBeta1 = 0.9 , TF.adamBeta2 = 0.999 , TF.adamEpsilon = 1e-8 }
  -- trainStep <- TF.minimizeWith (TF.adam' adamConfig) loss wghts

  -- let correctPredictions = TF.abs (predict `TF.sub` labels) `TF.lessEqual` TF.scalar 0.01
  -- errorRateTensor <- TF.render $ 1 - TF.reduceMean (TF.cast correctPredictions)
  (predictor, weights) <- TF.build predictorAndWeights
  return Model
    { allWeights = weights
    -- , train = \trainStep imFeed lFeed -> TF.runWithFeeds_ [TF.feed images imFeed , TF.feed labels lFeed] trainStep
    -- , infer = \tensor imFeed -> TF.runWithFeeds [TF.feed images imFeed] tensor
    -- , errorRate = \imFeed lFeed -> TF.unScalar <$> TF.runWithFeeds [TF.feed images imFeed , TF.feed labels lFeed] errorRateTensor
    , tensorPredict = predictor
    -- , tensorTrain = trainStep
    }

testSaveRestore :: Test
testSaveRestore = testCase "testSaveRestore" $
    withSystemTempDirectory "" $ \dirPath -> do
        let path = B8.pack $ dirPath ++ "/checkpoint"
            var :: TF.MonadBuild m => m (TF.Tensor TF.Ref Float)
            var = TF.zeroInitializedVariable' (TF.opName .~ "a")
                                        (TF.Shape [])
        TF.runSession $ do
            v <- var
            TF.assign v 134 >>= TF.run_
            TF.save path [v] >>= TF.run_
        result <- TF.runSession $ do
            v <- var
            TF.restore path v >>= TF.run_
            TF.run v
        liftIO $ TF.Scalar 134 @=? result


main :: IO ()
main = do

  let encodeImageBatch xs = TF.encodeTensorData [genericLength xs, 2] (V.fromList $ mconcat xs)
      encodeLabelBatch xs = TF.encodeTensorData [genericLength xs] (V.fromList xs)

  let predictorDef = TF.asGraphDef (tensorPredict <$> modelBuilder)
      outputTensorName = head (predictorDef ^. TF.node)^. TF.name
      inputTensorName = last (predictorDef ^. TF.node)^. TF.name
      namesPredictor = predictorDef ^.. TF.node.traversed.TF.name

  print $ head (predictorDef ^. TF.node)
  putStrLn $ "OpName: " ++ show outputTensorName

  putStrLn $ "NamesPredictor: " ++ show namesPredictor

  -- TF.runSession $ do
  --     TF.addGraphDef predictorDef
  --     (x :: V.Vector Float) <- TF.run $ do TF.tensorValueFromName outputTensorName
  --     liftIO $ print x

  -- TF.runSession $ do
  --     TF.addGraphDef predictorDef
  --     (x :: V.Vector Float) <- TF.run $ do TF.tensorValueFromName outputTensorName
  --     liftIO $ print x


  TF.runSession $ do
    TF.addGraphDef predictorDef

    let (inputLayer :: TF.Tensor TF.Value Float) = TF.tensorValueFromName inputTensorName
    let (predictor :: TF.Tensor TF.Value Float) = TF.tensorValueFromName outputTensorName

    forM_ ([0..1000] :: [Int]) $ \i -> do
      (x1Data :: [Float]) <- liftIO $ replicateM 1 randomIO
      (x2Data :: [Float]) <- liftIO $ replicateM 1 randomIO
      let xData = [[x1,x2] | x1 <- x1Data, x2 <- x2Data ]
      let yData = map (\(x1:x2:_) -> x1 * 0.3 + x2 * 0.5) xData

      let images = encodeImageBatch xData
          labels = encodeLabelBatch yData

      bef <- head . V.toList <$> TF.runWithFeeds [TF.feed inputLayer images] predictor
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


  nn <- randomNetworkInitWith HeEtAl :: IO NN


  let rl = mkBORLUnichainGrenade initState actions actionFilter params decay nn nnConfig
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
