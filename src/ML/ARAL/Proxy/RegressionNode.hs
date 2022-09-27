{-# LANGUAGE DeriveAnyClass    #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE OverloadedStrings #-}
module ML.ARAL.Proxy.RegressionNode
    ( Observation (..)
    , RegressionNode (..)
    , addGroundTruthValue
    , trainRegressionNode
    , applyRegrssionNode
    ) where


import           Control.DeepSeq
import qualified Data.Map.Strict               as M
import           Data.Serialize
import qualified Data.Vector                   as VB
import qualified Data.Vector.Storable          as VS
import           GHC.Generics
import           System.Random

import           ML.ARAL.NeuralNetwork.Scaling
import           Numeric.Regression.Linear

data Observation =
  Observation
    { inputValues :: VB.Vector Double
    , outputValue :: Double
    }
  deriving (Eq, Show, Generic, Serialize, NFData)


data RegressionNode =
  RegressionNode
    { observations           :: M.Map Int (VB.Vector Observation) -- Int for a scaled output value
    , stepSize               :: Double
    , maxObservationsPerStep :: Int
    , coefficients           :: VB.Vector Double
    }
  deriving (Eq, Show, Generic, Serialize, NFData)

addGroundTruthValue :: Observation -> RegressionNode -> RegressionNode
addGroundTruthValue obs@(Observation _ out) (RegressionNode m step maxObs coefs) = RegressionNode m' step maxObs coefs
  where
    key = floor (out * transf)
    transf = 1 / step
    m' = M.alter (Just . VB.take maxObs . maybe (VB.singleton obs) (obs `VB.cons`)) key m


trainRegressionNode :: RegressionNode -> RegressionNode
trainRegressionNode old@(RegressionNode m step maxObs coefs) = do
  if length allObs < 100
    then old
    else RegressionNode m step maxObs (last $ regress ys xs coefs :: Model VB.Vector Double)
  where
    allObs :: VB.Vector Observation
    allObs = VB.concat (M.elems m)
    xs :: VB.Vector (VB.Vector Double)
    xs = VB.map inputValues allObs
    ys :: VB.Vector Double
    ys = VB.map outputValue allObs


applyRegrssionNode :: RegressionNode -> VB.Vector Double -> Double
applyRegrssionNode (RegressionNode _ _ _ coefs) =
  error "TODO"
