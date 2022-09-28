{-# LANGUAGE DeriveAnyClass    #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections     #-}
module ML.ARAL.Proxy.RegressionNode
    ( Observation (..)
    , RegressionNode (..)
    , RegressionLayer
    , randRegressionNode
    , addGroundTruthValueLayer
    , trainRegressionLayer
    , applyRegressionLayer
    ) where

import           Control.DeepSeq
import           Control.Monad
import qualified Data.Map.Strict               as M
import           Data.Serialize
import qualified Data.Vector                   as VB
import qualified Data.Vector.Storable          as VS
import           Debug.Trace
import           GHC.Generics
import           System.IO
import           System.Random

import           ML.ARAL.NeuralNetwork.Scaling
import           ML.ARAL.Types
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

-- | One Regression for each actions.
type RegressionLayer = VB.Vector RegressionNode


randRegressionNode :: Int -> IO RegressionNode
randRegressionNode lenS = RegressionNode M.empty 0.1 300 . VB.fromList <$> replicateM (lenS + 1) (randomRIO (-0.1, 0.1 :: Double))


addGroundTruthValueNode :: Observation -> RegressionNode -> RegressionNode
addGroundTruthValueNode obs@(Observation _ out) (RegressionNode m step maxObs coefs) = RegressionNode m' step maxObs coefs
  where
    key = floor (out * transf)
    transf = 1 / step
    m' = M.alter (Just . VB.take maxObs . maybe (VB.singleton obs) (obs `VB.cons`)) key m

addGroundTruthValueLayer :: [Observation] -> RegressionLayer -> RegressionLayer
addGroundTruthValueLayer obs = VB.zipWith addGroundTruthValueNode (VB.fromList obs)


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


-- | Train regression layger (= all nodes).
trainRegressionLayer :: RegressionLayer -> RegressionLayer
trainRegressionLayer = VB.map trainRegressionNode


applyRegressionNode :: RegressionNode -> VS.Vector Double -> Double
applyRegressionNode (RegressionNode _ _ _ coefs) inps
  | VB.length coefs - 1 /= VS.length inps = error $ "applyRegressionNode: Expected number of coefficients is not correct: " ++ show (VB.length coefs, VS.length inps)
  | otherwise = VS.sum (VS.zipWith (*) (VB.convert coefs) inps) + VB.last coefs


-- | Apply regression layer to given inputs
applyRegressionLayer :: RegressionLayer -> ActionIndex -> VS.Vector Double -> Double
applyRegressionLayer regNodes actIdx = applyRegressionNode (regNodes VB.! actIdx)
  -- trace ("inps: " ++ show inps)


  -- -- let (RegressionNode _ _ _ coefs) = regNodes VB.! actIdx
  --  error "TODO"
