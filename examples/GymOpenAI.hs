{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveAnyClass             #-}
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
import           Debug.Trace
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
  , _trainBatchSize        = 1
  , _grenadeLearningParams = LearningParameters 0.01 0.9 0.0001
  , _prettyPrintElems      = map (St 0) ppSts
  , _scaleParameters       = scalingByMaxAbsReward False maxRew
  , _updateTargetInterval  = 5000
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

data St = St Int [Double]
  deriving (Generic, NFData)

instance Ord St where
  compare (St _ xs) (St _ ys) = compare xs ys

instance Eq St where
  (==) (St _ xs) (St _ ys) = xs == ys

instance Show St where
  show (St _ xs) = show xs

netInp :: Gym -> St -> [Double]
netInp gym (St _ st)=
  -- trace ("lows: " ++ show lows)
  -- trace ("highs: " ++ show highs)
  zipWith3 (curry scaleNegPosOne) lows highs st
  where range = getGymRangeFromSpace $ observationSpace gym
        (lows, highs) = gymRangeToDoubleLists range

modelBuilder :: (TF.MonadBuild m) => Integer -> Integer -> m TensorflowModel
modelBuilder nrInp nrOut =
  buildModel $
  inputLayer1D (fromIntegral nrInp) >> fullyConnected1D 10 TF.relu' >> fullyConnected1D 7 TF.relu' >> fullyConnected1D (fromIntegral nrOut) TF.tanh' >>
  trainingByAdam1DWith TF.AdamConfig {TF.adamLearningRate = 0.01, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}

action :: Gym -> Integer -> Action St
action gym idx = flip Action (T.pack $ show idx) $ \(St nr _) -> do
  res <- stepGym gym idx
  (nr', rew, obs) <- if episodeDone res
                then do obs <- resetGym gym
                        appendFile "episodeSteps" (show nr ++ "\n")
                        -- putStrLn $ "Done: " ++ show (cut ranges $ gymObservationToDoubleList $ observation res)
                        return (0, -20, obs)
                else return (nr+1, reward res , observation res)
  return (rew, St nr' $ cut ranges $ gymObservationToDoubleList obs, episodeDone res)
  where ranges = gymRangeToDoubleLists $ getGymRangeFromSpace $ observationSpace gym


cut :: ([Double], [Double]) -> [Double] -> [Double]
-- cut _ xs = xs
cut (lows, highs) xs = zipWith3 splitInto lows highs xs
  where
    splitInto lo hi x = -- x * scale
      fromIntegral (round (gran * x)) / gran
      where scale = 1/(hi - lo)
            gran = 20


main :: IO ()
main = do

  args <- getArgs
  let name | length args >= 1 = args!!0
           | otherwise = "CartPole-v0"
  let maxReward | length args >= 2  = read (args!!1)
                | otherwise = 2.5
  (obs, gym) <- initGym (T.pack name)
  setMaxEpisodeSteps gym 10000
  let inputNodes = dimension (observationSpace gym)
      actionNodes = dimension (actionSpace gym)
      ranges = gymRangeToDoubleLists $ getGymRangeFromSpace $ observationSpace gym
      initState = St 0 (cut ranges (gymObservationToDoubleList obs))
      actions = map (action gym) [0..actionNodes-1]

  writeFile "episodeSteps" ""

  nn <- randomNetworkInitWith UniformInit :: IO NN
  -- rl <- mkBORLUnichainGrenade initState actions actFilter params decay nn (nnConfig gym maxReward)
  rl <- mkBORLUnichainTensorflow initState actions actFilter params decay (modelBuilder inputNodes actionNodes) (nnConfig gym maxReward) (Just 0)
  -- let rl = mkBORLUnichainTabular initState actions actFilter params decay (Just 0)
  askUser True usage cmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []

-- | BORL Parameters.
params :: Parameters
params = Parameters
  { _alpha            = 0.30
  , _beta             = 0.05
  , _delta            = 0.04
  , _gamma            = 1.0
  , _epsilon          = 0.75
  , _exploration      = 1.0
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
      (max 0.01 $ slow * eps)
      (max 0.01 $ slow * exp)
      rand
      zeta -- zeta
      (0.5*bet)
  | otherwise = p
  where
    slower = 0.995
    slow = 0.98
    faster = 1.0 / 0.99
    f = max 0.01


actFilter :: St -> [Bool]
actFilter _  = repeat True
