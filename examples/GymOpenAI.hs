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

import           Control.Arrow          (first, second, (***))
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
                                               identity', identityN', lessEqual, matMul,
                                               mul, readerSerializeState, relu, relu',
                                               shape, square, sub, tanh, tanh',
                                               truncatedNormal)
import qualified TensorFlow.Minimize    as TF
import qualified TensorFlow.Ops         as TF (initializedVariable, initializedVariable',
                                               placeholder, placeholder', reduceMean,
                                               reduceSum, restore, save, scalar, vector,
                                               zeroInitializedVariable,
                                               zeroInitializedVariable')
import qualified TensorFlow.Tensor      as TF (Ref (..), collectAllSummaries,
                                               tensorNodeName, tensorRefFromName,
                                               tensorValueFromName)


newtype St = St [Double]
  deriving (Show, Eq, Ord, Generic, NFData)


maxX,maxY :: Int
maxX = 4                        -- [0..maxX]
maxY = 4                        -- [0..maxY]


type NN = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 5, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 5, 'D1 5]

modelBuilder :: (TF.MonadBuild m) => Integer -> Integer -> m TensorflowModel
modelBuilder nrInp nrOut =
  buildModel $
  inputLayer1D (fromIntegral nrInp) >>
  fullyConnected1D (5 * (fromIntegral (nrOut `div` 3) + fromIntegral nrInp)) TF.relu' >>
  fullyConnected1D (3 * (fromIntegral (nrOut `div` 2) + fromIntegral (nrInp `div` 2))) TF.relu' >>
  -- fullyConnected1D (1 * (fromIntegral nrOut + fromIntegral (nrInp `div` 3))) TF.relu' >>
  fullyConnected1D (fromIntegral nrOut) TF.tanh' >>
  trainingByAdam1DWith TF.AdamConfig {TF.adamLearningRate = 0.001, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}


nnConfig :: Gym -> Double -> NNConfig St
nnConfig gym maxRew = NNConfig
  { _toNetInp              = netInp gym
  , _replayMemoryMaxSize   = 20000
  , _trainBatchSize        = 8
  , _grenadeLearningParams = LearningParameters 0.01 0.9 0.0001
  , _prettyPrintElems      = map St ppSts
  , _scaleParameters       =
      scalingByMaxAbsReward False 1.5
      -- scalingByMaxAbsReward False maxRew
    -- ScalingNetOutParameters (-1) 1 (-150) 150 0 1.5 0 1000
  , _updateTargetInterval  = 5000
  , _trainMSEMax           = Just 0.05
  }

  where range = getGymRangeFromSpace $ observationSpace gym
        (lows, highs) = (map (max (-5)) *** map (min 5)) (gymRangeToDoubleLists range)
        vals = zipWith (\lo hi -> map rnd [lo, lo+(hi-lo)/3..hi]) lows highs
        rnd x = fromIntegral (round (100*x)) / 100
        ppSts = take 1000 $ combinations vals

netInp :: Gym -> St -> [Double]
netInp gym (St st) = st
  -- trace ("lows: " ++ show lows)
  -- trace ("highs: " ++ show highs)
  -- zipWith3 (curry scaleNegPosOne) lows highs st
  where range = getGymRangeFromSpace $ observationSpace gym
        (lows, highs) = (map (max (-5)) *** map (min 5)) (gymRangeToDoubleLists range)


combinations :: [[a]] -> [[a]]
combinations []       = []
combinations [xs] = map return xs
combinations (xs:xss) = concatMap (\x -> map (x:) ys) xs
  where ys = combinations xss


action :: Gym -> Integer -> Action St
action gym idx = flip Action (T.pack $ show idx) $ \_ -> do
  res <- stepGym gym idx
  (rew, obs) <- if episodeDone res
                then do obs <- resetGym gym
                        return (reward res, obs)
                else return (reward res, observation res)
  return (rew, St $ gymObservationToDoubleList obs, episodeDone res)


stGen :: ([Double], [Double]) -> St -> St
stGen (lows, highs) (St xs) = St $ zipWith3 splitInto lows highs xs
  where
    splitInto lo hi x = -- x * scale
      fromIntegral (round (gran * x)) / gran
      where scale = 1/(hi - lo)
            gran = 2

main :: IO ()
main = do

  args <- getArgs
  let name | length args >= 1 = args!!0
           | otherwise = "CartPole-v0"
  let maxReward | length args >= 2  = read (args!!1)
                | otherwise = 1
  (obs, gym) <- initGym (T.pack name)
  setMaxEpisodeSteps gym 10000
  let inputNodes = spaceSize (observationSpace gym)
      actionNodes = spaceSize (actionSpace gym)
      ranges = gymRangeToDoubleLists $ getGymRangeFromSpace $ observationSpace gym
      initState = St (gymObservationToDoubleList obs)
      actions = map (action gym) [0..actionNodes-1]

      initValues = Just $ defInitValues { defaultRho = 0, defaultR1 = 1}
  putStrLn $ "Actions: " ++ show actions
  let algorithm = AlgBORL 0.2 0.9 (ByMovAvg 100) Normal True
  nn <- randomNetworkInitWith UniformInit :: IO NN
  -- rl <- mkUnichainGrenade initState actions actFilter params decay nn (nnConfig gym maxReward)
  -- rl <- mkUnichainTensorflow algorithm initState actions actFilter params decay (modelBuilder inputNodes actionNodes) (nnConfig gym maxReward) initValues
  -- let rl = mkUnichainTabular algorithm initState (stGen ranges) actions actFilter params decay initValues
  let rl = mkUnichainTabular algorithm initState (netInp gym) actions actFilter params decay initValues
  askUser True usage cmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []

-- | BORL Parameters.
params :: Parameters
params = Parameters
  { _alpha            = 0.30
  , _beta             = 0.25
  , _delta            = 0.04
  , _gamma            = 1.0
  , _epsilon          = 0.01
  , _exploration      = 1.0
  , _learnRandomAbove = 0.0
  , _zeta             = 1.0
  , _xi               = 0.2
  }

-- | Decay function of parameters.
decay :: Decay
decay t p@(Parameters alp bet del ga eps exp rand zeta xi)
  | t `mod` 200 == 0 =
    Parameters
      (max 0.03 $ slow * alp)
      (max 0.015 $ slow * bet)
      (max 0.015 $ slow * del)
      (max 0.01 $ slow * ga)
      (max 0.01 $ slow * eps)
      (max 0.01 $ slower * exp)
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
