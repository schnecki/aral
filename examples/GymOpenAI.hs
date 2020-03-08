{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveAnyClass             #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedLists            #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE TypeFamilies               #-}

-- | !!! IMPORTANT !!!
--
-- REQUIREMENTS: python 3.4 and gym (https://gym.openai.com/docs/#installation)
--
--
--  ArchLinux Commands:
--  --------------------
--  $ yay -S python                # for yay see https://wiki.archlinux.org/index.php/AUR_helpers
--  $ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
--  $ python get-pip.py --user
--  $ pip install gym --user
--
--
--
--
module Main where

import           ML.BORL
import           ML.Gym

import           Helper

import           Control.Arrow           ((***))
import           Control.Concurrent.MVar
import           Control.DeepSeq         (NFData)
import           Control.Lens
import           Control.Monad           (join, when)
import           Data.List               (genericLength, sort)
import           Data.Maybe              (fromMaybe)
import qualified Data.Text               as T
import           GHC.Generics
import           GHC.Int                 (Int64)
import           Grenade
import           System.Environment      (getArgs)
import           System.IO.Unsafe        (unsafePerformIO)

import qualified TensorFlow.Core         as TF hiding (value)
import qualified TensorFlow.GenOps.Core  as TF (relu', tanh')
import qualified TensorFlow.Minimize     as TF

import           Debug.Trace

type Render = Bool

data St = St Render [Double]
  deriving (Generic, NFData)

instance Eq St where
  (St _ xs1) == (St _ xs2) = xs1 == xs2

instance Ord St where
  (St _ xs1) `compare` (St _ xs2) = xs1 `compare` xs2

instance Show St where
  show (St _ xs) = showFloatList xs


maxX,maxY :: Int
maxX = 4                        -- [0..maxX]
maxY = 4                        -- [0..maxY]


type NN = Network
          '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 6, Tanh]
          '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 6, 'D1 6]


modelBuilder :: (TF.MonadBuild m) => Gym -> St -> Integer -> Int64 -> m TensorflowModel
modelBuilder gym initSt nrActions outCols =
  buildModel $
  inputLayer1D inpLen >> fullyConnected [10 * inpLen] TF.relu' >> fullyConnected [5 * inpLen] TF.relu' >> fullyConnected [5 * inpLen] TF.relu' >>
  fullyConnected [fromIntegral nrActions, outCols] TF.tanh' >>
  -- trainingByAdamWith TF.AdamConfig {TF.adamLearningRate = 0.001, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}
  trainingByGradientDescent 0.01
  where
    inpLen = genericLength (netInp False gym initSt)


nnConfig :: Gym -> Double -> NNConfig
nnConfig gym maxRew =
  NNConfig
    { _replayMemoryMaxSize = 10000
    , _trainBatchSize = 8
    , _grenadeLearningParams = LearningParameters 0.01 0.9 0.0001
    , _learningParamsDecay = ExponentialDecay Nothing 0.5 100000
    , _prettyPrintElems = map (zipWith3 (\l u -> scaleValue (Just (l, u))) lows highs) ppSts
    , _scaleParameters = scalingByMaxAbsRewardAlg alg False (1.25* maxRew)
    , _stabilizationAdditionalRho = 0.0
    , _stabilizationAdditionalRhoDecay = ExponentialDecay Nothing 0.05 100000
    , _updateTargetInterval = 15000
    , _updateTargetIntervalDecay = StepWiseIncrease (Just 500) 0.1 10000
    , _trainMSEMax = Nothing -- Just 0.05
    , _setExpSmoothParamsTo1 = True
    }
  where
    (lows, highs) = observationSpaceBounds gym
    mkParamList lo hi = sort $ takeMiddle 3 xs
      where xs = map rnd [lo, lo + (hi - lo) / 10 .. hi]
    vals | length lows > 4 = []
         | otherwise = zipWith mkParamList lows highs
    rnd x = fromIntegral (round (1000 * x)) / 1000
    ppSts = takeMiddle 20 $ combinations vals

takeMiddle :: Int -> [a] -> [a]
takeMiddle _ [] = []
takeMiddle nr xs
  | nr <= 0 = []
  | otherwise = xs !! idx : takeMiddle (nr - 1) (take idx xs ++ drop (idx + 1) xs)
  where
    idx = length xs `div` 2

combinations :: [[a]] -> [[a]]
combinations []       = []
combinations [xs] = map return xs
combinations (xs:xss) = concatMap (\x -> map (x:) ys) xs
  where ys = combinations xss


globalVar :: MVar (Maybe Double)
globalVar = unsafePerformIO $ newMVar Nothing
{-# NOINLINE globalVar #-}
setGlobalVar :: Maybe Double -> IO ()
setGlobalVar x = modifyMVar_ globalVar (return . const x)
{-# NOINLINE setGlobalVar #-}
getGlobalVar :: IO (Maybe Double)
getGlobalVar = join <$> tryReadMVar globalVar
{-# NOINLINE getGlobalVar #-}

action :: Gym -> Maybe T.Text -> Integer -> Action St
action gym mName idx =
  flip Action (fromMaybe (T.pack $ show idx) mName) $ \oldSt@(St render _) -> do
    res <- stepGymRender render gym idx
    rew <- rewardFunction gym oldSt (fromIntegral idx) res
    obs <-
      if episodeDone res
        then resetGym gym
        else return (observation res)
    return (rew, St render (gymObservationToDoubleList obs), episodeDone res)


rewardFunction :: Gym -> St -> ActionIndex -> GymResult -> IO (Reward St)
rewardFunction gym (St _ oldSt) actIdx (GymResult obs rew eps) =
  case alg of
    AlgDQN {} -> return $ Reward rew
    AlgDQNAvgRewAdjusted  {}
      | name gym == "CartPole-v1" -> do
        epsStep <- getElapsedSteps gym
        let rad = xs !! 3 -- (-0.418879, 0.418879) .. 24 degrees in rad
            pos = xs !! 1 -- (-4.8, 4.8)
        return $ Reward $ (100 *) $ 0.41887903213500977 - abs rad - 0.418879 / 4.8 * abs pos - ite (eps && epsStep /= Just maxEpsSteps) 1.0 0
    AlgDQNAvgRewAdjusted {}
      | name gym == "MountainCar-v0" -> do
        let pos = head xs
            height = sin (3 * pos) * 0.45 + 0.55
            velocity = xs !! 1
            oldPos = head oldSt
            oldVelocity = oldSt !! 1
        step <- fromMaybe 0 <$> getGlobalVar
        setGlobalVar (Just $ step + 1)
        let movGoal = min 0.5 (5e-6 * step - 0.3)
        epsStep <- getElapsedSteps gym
        return $ Reward $ (20 *) $ ite (eps && epsStep < Just maxEpsSteps) (* 1.2) id $ ite (pos > (-0.3) && velocity >= 0 || pos < (-0.3) && velocity <= 0) height 0
    AlgDQNAvgRewAdjusted  {}
      | name gym == "Acrobot-v1" ->
        let [cosS0, sinS0, cosS1, sinS1, thetaDot1, thetaDot2] = xs -- cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2
         in return $ Reward $ (* 20) $ -cosS0 - cos (acos cosS0 + acos cosS1)
                           -- -cos(s[0]) - cos(s[1] + s[0])
    AlgDQNAvgRewAdjusted {}
      | name gym == "Copy-v0" -> return $ Reward rew
    AlgDQNAvgRewAdjusted {}
      | name gym == "Pong-ram-v0" -> return $ Reward rew
  where
    xs = gymObservationToDoubleList obs
    ite True x _  = x
    ite False _ x = x
    maxEpsSteps = maximumEpisodeSteps (name gym)

rewardFunction gym _ _ _ = error $ "rewardFunction not yet defined for this environment: " ++ T.unpack (name gym)

maxReward :: Gym -> Double
maxReward _ | isAlgDqn alg = 10
maxReward gym | name gym == "CartPole-v1" = 50
              | name gym == "MountainCar-v0" = 200
              | name gym == "Copy-v0" = 1.0
              | name gym == "Acrobot-v1" = 50
              -- | name gym == "Pendulum-v0" =
              | name gym == "Pong-ram-v0" = 1
maxReward gym   = error $ "Max Reward (maxReward) function not yet defined for this environment: " ++ T.unpack (name gym)


-- | Scales values to (-1, 1).
netInp :: Bool -> Gym -> St -> [Double]
netInp isTabular gym (St _ st)
  | not isTabular = zipWith3 (\l u -> scaleValue (Just (l, u))) lowerBounds upperBounds st
  | isTabular = map (rnd . fst) $ filter snd (stSelector st)
  where
    rnd x = fromIntegral (round (x * 10)) / 10
    (lowerBounds, upperBounds) = observationSpaceBounds gym
    stSelector xs
      -- | name gym == "MountainCar-v0" = [(head xs, True), (5 * (xs !! 1), False)]
      | otherwise = zip xs (repeat True)

observationSpaceBounds :: Gym -> ([Double], [Double])
observationSpaceBounds gym = map (max (-maxVal)) *** map (min maxVal) $ gymRangeToDoubleLists $ getGymRangeFromSpace $ observationSpace gym
  where
    maxVal | name gym == "CartPole-v1" = 5
           | otherwise = 1000


mInverseSt :: Gym -> Maybe (NetInputWoAction -> Maybe (Either String St))
mInverseSt gym = Just $ \xs -> Just $ Right $ St True $ zipWith3 (\l u x -> unscaleValue (Just (l, u)) x) lowerBounds upperBounds xs
  where
    (lowerBounds, upperBounds) = observationSpaceBounds gym


instance RewardFuture St where
  type StoreType St = ()


alg :: Algorithm St
alg =
  -- algDQN
  -- AlgDQNAvgRewAdjusted Nothing 0.85 0.99 ByStateValues
  AlgDQNAvgRewAdjusted (Just 0.01) 0.85 1.0 ByStateValues


main :: IO ()
main = do
  args <- getArgs
  putStrLn $ "Received arguments: " ++ show args
  let name
        | not (null args) = T.pack (head args)
        | otherwise = "MountainCar-v0"
             -- "CartPole-v1"
  (obs, gym) <- initGym name
  let maxRew
        | length args >= 2 = read (args !! 1)
        | otherwise = maxReward gym
  putStrLn $ "Gym: " ++ show gym
  maxEpsSteps <- getMaxEpisodeSteps gym
  putStrLn $ "Default maximum episode steps: " ++ show maxEpsSteps
  setMaxEpisodeSteps gym (maximumEpisodeSteps name)
  let inputNodes = spaceSize (observationSpace gym)
      actionNodes = spaceSize (actionSpace gym)
      initState = St False (gymObservationToDoubleList obs)
      actNames = actionNames name
      actions = zipWith (action gym) (map Just actNames ++ repeat Nothing) [0 .. actionNodes - 1]
      initValues = Just $ defInitValues {defaultRho = 0, defaultRhoMinimum = 0, defaultR1 = 1}
  putStrLn $ "Actions: " ++ show actions
  putStrLn $ "Observation Space: " ++ show (observationSpaceInfo name)
  putStrLn $ "Enforced observation bounds: " ++ show (observationSpaceBounds gym)
  nn <- randomNetworkInitWith UniformInit :: IO NN
  -- rl <- mkUnichainGrenadeCombinedNet alg initState (netInp False gym) actions actFilter (params gym maxRew) (decay gym) nn (nnConfig gym maxRew) initValues
  rl <- mkUnichainTensorflowCombinedNet alg initState (netInp False gym) actions actFilter (params gym maxRew) (decay gym) (modelBuilder gym initState actionNodes) (nnConfig gym maxRew) initValues
  -- rl <- mkUnichainTensorflow alg initState (netInp False gym) actions actFilter (params gym maxRew) (decay gym) (modelBuilder gym initState actionNodes) (nnConfig gym maxRew) initValues
  ---let rl = mkUnichainTabular alg initState (netInp True gym) actions actFilter (params gym maxRew) (decay gym) initValues
  askUser (mInverseSt gym) True usage cmds qlCmds rl -- maybe increase learning by setting estimate of rho
  where
    cmds = []
    usage = []
    qlCmds = [ ("f", "flip rendering", s %~ (\(St r xs) -> St (not r) xs))]


 -- | BORL Parameters.
params :: Gym -> Double -> ParameterInitValues
params gym maxRew =
  Parameters
    { _alpha               = 0.03
    , _beta                = 0.01
    , _delta               = 0.005
    , _gamma               = 0.01
    , _epsilon             = eps
    , _explorationStrategy = EpsilonGreedy -- SoftmaxBoltzmann 10 -- EpsilonGreedy
    , _exploration         = 1.0
    , _learnRandomAbove    = 0.5
    , _zeta                = 0.03
    , _xi                  = 0.005
    , _disableAllLearning  = False
    -- ANN
    , _alphaANN            = 0.5 -- only used for multichain
    , _betaANN             = 0.5
    , _deltaANN            = 0.5
    , _gammaANN            = 0.5
    }
  where eps | name gym == "MountainCar-v0" = 0.25
            | otherwise = min 1.0 $ max 0.05 $ 0.005 * maxRew

decay :: Gym -> Decay
decay gym =
  decaySetupParameters
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-5) 0.5 30000
      , _beta             = ExponentialDecay (Just 1e-2) 0.5 50000
      , _delta            = ExponentialDecay (Just 1e-2) 0.5 150000
      , _gamma            = ExponentialDecay (Just 1e-2) 0.5 150000
      , _zeta             = ExponentialDecay (Just 0) 0.5 150000
      , _xi               = NoDecay
      -- Exploration
      , _epsilon          = NoDecay -- ExponentialDecay (Just 0.03) 0.05 15000
      , _exploration      = ExponentialDecay (Just minExp) 0.50 (expFact * 50000)
      , _learnRandomAbove = NoDecay
      -- ANN
      , _alphaANN         = ExponentialDecay Nothing 0.75 150000
      , _betaANN          = ExponentialDecay Nothing 0.75 150000
      , _deltaANN         = ExponentialDecay Nothing 0.75 150000
      , _gammaANN         = ExponentialDecay Nothing 0.75 150000
      }
  where minExp -- | name gym == "MountainCar-v0" = 0.15
               | otherwise = 0.01
        expFact
          | name gym == "MountainCar-v0" = 2
          | otherwise = 1


actFilter :: St -> [Bool]
actFilter _  = repeat True -- [True, False, True]

maximumEpisodeSteps :: T.Text -> Integer
maximumEpisodeSteps "CartPole-v1" = 50000
maximumEpisodeSteps _             = 10000

actionNames :: T.Text -> [T.Text]
actionNames "CartPole-v1"    = ["left ", "right"]
actionNames "MountainCar-v0" = ["left ", "cont ", "right"]
actionNames "Acrobot-v1"     = ["left ", "cont ", "right"]
-- actionNames "Pong-ram-v0" = []
actionNames _                = []

observationSpaceInfo :: T.Text -> [T.Text]
observationSpaceInfo "CartPole-v1" = ["Cart Position (-4.8, 4.8)", "Cart Velocity (-Inf, Inf)", "Pole Angle (-24 deg, 24 deg)", "Pole Velocity At Tip (-Inf, Inf)"]
observationSpaceInfo "MountainCar-v0" = ["Position (-1.2, 0.6)", "Velocity (-0.07, 0.07)"]
observationSpaceInfo "Acrobot-v1" = ["cos(theta1)", "sin(theta1)", "cos(theta2)", "sin(theta2)", "thetaDot1", "thetaDot2"]
observationSpaceInfo _ = ["unkown observation space description"]
