{-# LANGUAGE FlexibleContexts  #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedLists   #-}

-- | This modules is used only to test whether the haskell-tensorflow interaction works. Refs:
-- https://github.com/tensorflow/haskell and
-- https://nbviewer.jupyter.org/github/lgastako/public-notebooks/blob/master/TensorFlow%20MNIST%20Working.ipynb
module Main where


import           Control.Monad                       (forM_, replicateM, replicateM_, when)
import           System.Random                       (randomIO)
import           Test.HUnit                          (assertBool)

import           Control.Monad.IO.Class              (liftIO)
import           Data.Int                            (Int32, Int64)
import           Data.List                           (genericLength)
import qualified Data.Text.IO                        as T
import qualified Data.Vector                         as V

import qualified TensorFlow.Core                     as TF
import qualified TensorFlow.GenOps.Core              as TF (square)
import qualified TensorFlow.Minimize                 as TF
import qualified TensorFlow.Ops                      as TF hiding (initializedVariable,
                                                            zeroInitializedVariable)
import qualified TensorFlow.Variable                 as TF

import           TensorFlow.Examples.MNIST.InputData
import           TensorFlow.Examples.MNIST.Parse

numPixels :: Int64
numPixels = 28*28 :: Int64
numLabels :: Int64
numLabels = 10    :: Int64


-- | Linear Regression
fit :: [Float] -> [Float] -> TF.Session (Float, Float)
fit xData yData = do
    -- Create tensorflow constants for x and y.
    let x = TF.vector xData
        y = TF.vector yData
    -- Create scalar variables for slope and intercept.
    w <- TF.initializedVariable 0
    b <- TF.initializedVariable 0
    -- Define the loss function.
    let yHat = (x `TF.mul` TF.readValue w) `TF.add` TF.readValue b
        loss = TF.square (yHat `TF.sub` y)
    -- Optimize with gradient descent.
    trainStep <- TF.minimizeWith (TF.gradientDescent 0.001) loss [w, b]
    replicateM_ 1000 (TF.run trainStep)
    -- Return the learned parameters.
    (TF.Scalar w', TF.Scalar b') <- TF.run (TF.readValue w, TF.readValue b)
    return (w', b')


-- | Create tensor with random values where the stddev depends on the width.
randomParam :: Int64 -> TF.Shape -> TF.Build (TF.Tensor TF.Build Float)
randomParam width (TF.Shape shape) = (`TF.mul` stddev) <$> TF.truncatedNormal (TF.vector shape)
  where
    stddev = TF.scalar (1 / sqrt (fromIntegral width))

-- Types must match due to model structure.
type LabelType = Int32

data Model = Model {
      train :: TF.TensorData Float  -- ^ images
            -> TF.TensorData LabelType
            -> TF.Session ()
    , infer :: TF.TensorData Float  -- ^ images
            -> TF.Session (V.Vector LabelType)  -- ^ predictions
    , errorRate :: TF.TensorData Float  -- ^ images
                -> TF.TensorData LabelType
                -> TF.Session Float
    }

createModel :: TF.Build Model
createModel = do
    -- Use -1 batch size to support variable sized batches.
    let batchSize = -1

    -- Inputs.
    images <- TF.placeholder [batchSize, numPixels]

    -- Hidden layer.
    let numUnits = 200

    hiddenWeights <- TF.initializedVariable =<< randomParam numPixels [numPixels, numUnits]
    hiddenBiases  <- TF.zeroInitializedVariable [numUnits]
    let hiddenZ = (images `TF.matMul` TF.readValue hiddenWeights) `TF.add` TF.readValue hiddenBiases
    let hidden = TF.relu hiddenZ

    -- Logits.
    logitWeights <- TF.initializedVariable =<< randomParam numUnits [numUnits, numLabels]
    logitBiases  <- TF.zeroInitializedVariable [numLabels]
    let logits = (hidden `TF.matMul` TF.readValue logitWeights) `TF.add` TF.readValue logitBiases
    predict <- TF.render $ TF.cast (TF.argMax (TF.softmax logits) (TF.scalar (1 :: LabelType)) :: TF.Tensor TF.Build Int32)

    -- Create training action.
    labels <- TF.placeholder [batchSize]
    let labelVecs = TF.oneHot labels (fromIntegral numLabels) 1 0
        loss      = reduceMean $ fst $ TF.softmaxCrossEntropyWithLogits logits labelVecs
        params    = [hiddenWeights, hiddenBiases, logitWeights, logitBiases]
    trainStep <- TF.minimizeWith TF.adam loss params

    let correctPredictions = TF.equal predict labels
    errorRateTensor <- TF.render $ 1 - reduceMean (TF.cast correctPredictions)

    return Model {
          train = \imFeed lFeed -> TF.runWithFeeds_ [
                TF.feed images imFeed
              , TF.feed labels lFeed
              ] trainStep
        , infer = \imFeed -> TF.runWithFeeds [TF.feed images imFeed] predict
        , errorRate = \imFeed lFeed -> TF.unScalar <$> TF.runWithFeeds [
                TF.feed images imFeed
              , TF.feed labels lFeed
              ] errorRateTensor
        }

main :: IO ()
main = do
  putStrLn "Simple linear regression test"
  putStrLn "-----------------------------"
  xData <- replicateM 100 randomIO
  let yData = [x * 3 + 8 | x <- xData]
  -- Fit linear regression model.
  (w, b) <-   TF.runSession $ fit xData yData
  putStrLn $ "w: " ++ show w
  putStrLn $ "b: " ++ show b
  assertBool "w == 3" (abs (3 - w) < 0.001)
  assertBool "b == 8" (abs (8 - b) < 0.001)


  TF.runSession $ do
    liftIO $ putStrLn "\n\nNeural Network Test"
    liftIO $ putStrLn      "------------------"
    -- Read training and test data.
    trainingImages <- liftIO (readMNISTSamples =<< trainingImageData)
    trainingLabels <- liftIO (readMNISTLabels =<< trainingLabelData)
    testImages <- liftIO (readMNISTSamples =<< testImageData)
    testLabels <- liftIO (readMNISTLabels =<< testLabelData)
    -- Create the model.
    model <- TF.build createModel
    -- Functions for generating batches.
    let encodeImageBatch xs = TF.encodeTensorData [genericLength xs, numPixels] (fromIntegral <$> mconcat xs)
        encodeLabelBatch xs = TF.encodeTensorData [genericLength xs] (fromIntegral <$> V.fromList xs)
        batchSize = 200
        selectBatch i xs = take batchSize $ drop (i * batchSize) (cycle xs)
    -- Train.
    forM_ ([0 .. 1000] :: [Int]) $ \i -> do
      let images = encodeImageBatch (selectBatch i trainingImages)
          labels = encodeLabelBatch (selectBatch i trainingLabels)
      train model images labels
      when (i `mod` 100 == 0) $ do
        err <- errorRate model images labels
        liftIO . putStrLn $ "training error " ++ show (err * 100)
    liftIO . putStrLn $ ""
    -- Test.
    testErr <- errorRate model (encodeImageBatch testImages) (encodeLabelBatch testLabels)
    liftIO . putStrLn $ "test error " ++ show (testErr * 100)
    -- Show some predictions.
    testPreds <- infer model (encodeImageBatch testImages)
    let numPredictions = 20
    liftIO $
      forM_ ([0 .. (numPredictions - 1)] :: [Int]) $ \i -> do
        putStrLn ""
        T.putStrLn $ drawMNIST $ testImages !! i
        putStrLn $ "expected " ++ show (testLabels !! i)
        putStrLn $ "     got " ++ show (testPreds V.! i)

-- Ganked from tensorflow-mnist/app/Main.hs
reduceMean :: TF.Tensor TF.Build Float -> TF.Tensor TF.Build Float
reduceMean xs = TF.mean xs (TF.scalar (0 :: Int32))


