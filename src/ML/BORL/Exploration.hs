{-# LANGUAGE BangPatterns   #-}
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

type TemperatureInitFactor = Float -- ^ Will be decayed by multiplying with the exploration value of the parameters.

data ExplorationStrategy
  = Greedy                                  -- ^ Use greedy action selection
  | EpsilonGreedy                           -- ^ Use Epsilon greedy algorithm
  | SoftmaxBoltzmann !TemperatureInitFactor -- ^ Choose actions based on learned values. The initial temperature factor
                                            -- will be decayed by multiplying with the exploration value of the
                                            -- parameters.
  deriving (Show, Eq, Ord, NFData, Generic, Serialize)


-- | This normalises the input and returns a softmax using the Bolzmann distribution with given temperature.
softmax :: (Ord n, Floating n) => n -> [n] -> [n]
softmax _ [] = []
softmax temp xs
  | all (== head xs) (tail xs) = replicate (length xs) (1 / genericLength xs)
  | otherwise = map (/ max eps s) xs'
  where
    normed = normalise xs
    xs' = map (exp . (/ max eps temp) . subtract 1) normed
    s = sum xs'
    eps = 1e-3

-- | Normalise the input list to (-1, 1).
normalise :: (Ord n, Fractional n) => [n] -> [n]
normalise [] = error "empty input to normalise in ML.BORL.Exploration"
normalise xs = map (scaleZeroOneFloat (minV, maxV)) xs
  where minV = minimum xs
        maxV = maximum xs


-- | This (actual correct version withouth normalisation) does not work as the probabilities are to close to 1.
softmaxNonNormalised :: (Ord n, Floating n) => n -> [n] -> [n]
softmaxNonNormalised _ [] = []
softmaxNonNormalised temp xs = map (/ max eps s) xs'
  where
    maxVal = maximum xs
    xs' = map (exp . (/ max eps temp) . subtract maxVal) xs
    s = sum xs'
    eps = 1e-4
