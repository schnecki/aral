{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedLists            #-}
{-# LANGUAGE OverloadedStrings          #-}

-- | !!! IMPORTANT !!!
--
-- REQUIREMENTS: python 3.4 and gym (https://gym.openai.com/docs/#installation)
--
--
--  ArchLinux Commands:
--  --------------------
--  $ yay -S python34                # for yay see https://wiki.archlinux.org/index.php/AUR_helpers
--  $ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
--  $ python3.4 get-pip.py
--  $ pip3.4 install gym --user
--
--
--
--
module Main where

import           ML.BORL
import           ML.Gym


import           Helper

import           Control.Arrow          (first, second)
import           Control.DeepSeq        (NFData)
import qualified Control.Exception      as E
import           Control.Lens
import           Control.Lens           (set, (^.))
import           Control.Monad          (foldM_, forM, forM_, void)
import           Control.Monad          (foldM, unless, when)
import           Control.Monad.IO.Class (liftIO)
import           Data.List              (genericLength)
import qualified Data.Text              as T
import           GHC.Generics
import           GHC.Int                (Int32, Int64)
import           Grenade
import           System.Environment     (getArgs)
import           System.Exit
import           System.IO
import           System.Random

import qualified TensorFlow.Build       as TF (addNewOp, evalBuildT, explicitName, opDef,
                                               opDefWithName, opType, runBuildT, summaries)
import qualified TensorFlow.Core        as TF hiding (value)
import qualified TensorFlow.GenOps.Core as TF (abs, add, approximateEqual,
                                               approximateEqual, assign, cast,
                                               getSessionHandle, getSessionTensor,
                                               identity', lessEqual, matMul, mul,
                                               readerSerializeState, relu, relu', shape,
                                               square, sub, tanh, tanh', truncatedNormal)
import qualified TensorFlow.Minimize    as TF
import qualified TensorFlow.Ops         as TF (initializedVariable, initializedVariable',
                                               placeholder, placeholder', reduceMean,
                                               reduceSum, restore, save, scalar, vector,
                                               zeroInitializedVariable,
                                               zeroInitializedVariable')
import qualified TensorFlow.Tensor      as TF (Ref (..), collectAllSummaries,
                                               tensorNodeName, tensorRefFromName,
                                               tensorValueFromName)


maxX,maxY :: Int
maxX = 4                        -- [0..maxX]
maxY = 4                        -- [0..maxY]


type NN = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 5, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 5, 'D1 5]

nnConfig :: Gym -> Double -> NNConfig St
nnConfig gym maxRew = NNConfig
  { _toNetInp              = netInp gym
  , _replayMemoryMaxSize   = 10000
  , _trainBatchSize        = 32
  , _grenadeLearningParams = LearningParameters 0.01 0.9 0.0001
  , _prettyPrintElems      = ppSts
  , _scaleParameters       = scalingByMaxAbsReward False maxRew
  , _updateTargetInterval  = 1000
  , _trainMSEMax           = Nothing
  }

  where range = getGymRangeFromSpace $ observationSpace gym
        (lows, highs) = gymRangeToDoubleLists range
        vals = zipWith (\lo hi -> [lo, lo+(hi-lo)/3..hi]) lows highs
        ppSts = take 1000 $ combinations vals

combinations :: [[a]] -> [[a]]
combinations []       = []
combinations [xs] = map return xs
combinations (xs:xss) = concatMap (\x -> map (x:) ys) xs
  where ys = combinations xss

type St = [Double]


netInp :: Gym -> St -> [Double]
netInp gym = zipWith3 (curry scaleNegPosOne) lows highs
  where range = getGymRangeFromSpace $ observationSpace gym
        (lows, highs) = gymRangeToDoubleLists range

  -- [scaleNegPosOne (0, fromIntegral maxX) $ fromIntegral $ fst (getCurrentIdx st),
  --   scaleNegPosOne (0, fromIntegral maxY) $ fromIntegral $ snd (getCurrentIdx st)]


modelBuilder :: (TF.MonadBuild m) => Integer -> Integer -> m TensorflowModel
modelBuilder nrInp nrOut =
  buildModel $
  inputLayer1D (fromIntegral nrInp) >> fullyConnected1D 10 TF.relu' >> fullyConnected1D 7 TF.relu' >> fullyConnected1D (fromIntegral nrOut) TF.tanh' >>
  trainingByAdam1DWith TF.AdamConfig {TF.adamLearningRate = 0.001, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}

action :: Gym -> Integer -> Action St
action gym idx = flip Action (T.pack $ show idx) $ \_ -> do
  res <- stepGym gym idx
  obs <- if episodeDone res
         then resetGym gym
         else return $ observation res
  -- putStrLn $ "rew: " ++ show (reward res)
  return (reward res, gymObservationToDoubleList obs)


main :: IO ()
main = do

  args <- getArgs
  let name | length args >= 1 = args!!0
           | otherwise = "CartPole-v0"
  let maxReward | length args >= 2  = read (args!!1)
                | otherwise = 1
  (obs, gym) <- initGym (T.pack name)
  let inputNodes = dimension (observationSpace gym)
      actionNodes = dimension (actionSpace gym)
      initState = gymObservationToDoubleList obs
      actions = map (action gym) [0..actionNodes-1]


  nn <- randomNetworkInitWith UniformInit :: IO NN
  -- rl <- mkBORLUnichainGrenade initState actions actFilter params decay nn (nnConfig gym maxReward)
  rl <- mkBORLUnichainTensorflow initState actions actFilter params decay (modelBuilder inputNodes actionNodes) (nnConfig gym maxReward) (Just (-1))
  -- let rl = mkBORLUnichainTabular initState actions actFilter params decay
  askUser True usage cmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []

-- | BORL Parameters.
params :: Parameters
params = Parameters
  { _alpha            = 0.5
  , _beta             = 0.05
  , _delta            = 0.04
  , _gamma            = 0.30
  , _epsilon          = 0.075
  , _exploration      = 0.8
  , _learnRandomAbove = 0.0
  , _zeta             = 1.0
  , _xi               = 0.2
  }

-- | Decay function of parameters.
decay :: Decay
decay t (psiRhoOld, psiVOld, psiWOld) (psiRhoNew, psiVNew, psiWNew) p@(Parameters alp bet del ga eps exp rand zeta xi)
  | t `mod` 200 == 0 =
    Parameters
      (max 0.03 $ slow * alp)
      (max 0.015 $ slow * bet)
      (max 0.015 $ slow * del)
      (max 0.01 $ slow * ga)
      (max 0.05 $ slow * eps) -- (0.5*bet)
      (max 0.01 $ slow * exp)
      rand
      zeta -- zeta
      -- (max 0.075 $ slower * xi)
      (0.5*bet)
  | otherwise = p
  where
    slower = 0.995
    slow = 0.98
    faster = 1.0 / 0.99
    f = max 0.01


actFilter :: St -> [Bool]
actFilter _  = repeat True
