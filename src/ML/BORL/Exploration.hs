{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
module ML.BORL.Exploration
    ( ExplorationStrategy (..)
    , TemperatureInitFactor
    , softmax
    ) where

import           Control.DeepSeq
import           Data.List                     (genericLength)
import           Data.Serialize
import           GHC.Generics

import           ML.BORL.NeuralNetwork.Scaling

type TemperatureInitFactor = Double -- ^ Will be decayed by multiplying with the exploration value of the parameters.

data ExplorationStrategy
  = EpsilonGreedy                          -- ^ Use Epsilon greedy algorithm
  | SoftmaxBoltzmann TemperatureInitFactor -- ^ Choose actions based on learned values. The initial temperature factor
                                           -- will be decayed by multiplying with the exploration value of the
                                           -- parameters.
  deriving (Show, Eq, Ord, NFData, Generic, Serialize)


-- | This normalises the input and returns a softmax using the Bolzmann distribution with given temperature.
softmax :: Double -> [Double] -> [Double]
softmax _ [] = []
softmax temp xs
  | all (== head xs) (tail xs) = replicate (length xs) (1 / genericLength xs)
  | otherwise = map (/ max eps s) xs'
  where
    normed = normalise xs
    xs' = map (exp . (/ max eps temp)) normed
    s = sum xs'
    eps = 1e-2

-- | Normalise the input list to (-1, 1).
normalise :: [Double] -> [Double]
normalise [] = error "empty input to normalise in ML.BORL.Exploration"
normalise xs = map (scaleZeroOneValue (minV, maxV)) xs
  where minV = minimum xs
        maxV = maximum xs
