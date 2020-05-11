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

import           Control.Arrow           ((***))
import           Control.Concurrent.MVar
import           Control.DeepSeq         (NFData)
import           Control.Lens
import           Control.Monad           (join, when)
import           Data.Default
import           Data.List               (genericLength, sort)
import           Data.Maybe              (fromMaybe)
import qualified Data.Text               as T
import qualified Data.Vector.Storable    as V
import           GHC.Generics
import           GHC.Int                 (Int64)
import           Grenade
import           System.Environment      (getArgs)
import           System.IO.Unsafe        (unsafePerformIO)

import qualified HighLevelTensorflow     as TF

import           Helper
import           ML.BORL
import           ML.Gym


import           Debug.Trace

type Render = Bool

data St = St Render [Float]
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


type NN = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 5, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 5, 'D1 5]
type NNCombined = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 40, Relu, FullyConnected 40 40, Relu, FullyConnected 40 30, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 40, 'D1 40, 'D1 40, 'D1 40, 'D1 30, 'D1 30]
type NNCombinedAvgFree = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 6, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 6, 'D1 6]

modelBuilderGrenade :: Gym -> St -> Integer -> Integer -> IO SpecConcreteNetwork
modelBuilderGrenade gym initState lenActs cols =
  buildModel $
  inputLayer1D lenIn >>
  fullyConnected (20*lenIn) >> relu >> dropout 0.90 >>
  fullyConnected (10 * lenIn) >> relu >>
  fullyConnected (5 * lenIn) >> relu >>
  fullyConnected (2*lenOut) >> relu >>
  fullyConnected lenOut >> reshape (lenActs, cols, 1) >> tanhLayer
  where
    lenOut = lenActs * cols
    lenIn = fromIntegral $ V.length $ netInp False gym initState


modelBuilder :: (TF.MonadBuild m) => Gym -> St -> Integer -> Int64 -> m TF.TensorflowModel
modelBuilder gym initSt nrActions outCols =
  TF.buildModel $
  TF.inputLayer1D inpLen >>
  TF.fullyConnected [10 * inpLen] TF.relu' >>
  TF.fullyConnected [5 * inpLen] TF.relu' >>
  TF.fullyConnected [5 * inpLen] TF.relu' >>
  TF.fullyConnected [fromIntegral nrActions, outCols] TF.tanh' >>
  -- TF.fullyConnectedLinear [fromIntegral nrActions, outCols] >>
  TF.trainingByAdamWith TF.AdamConfig {TF.adamLearningRate = 0.001, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}
  -- TF.trainingByGradientDescent 0.01
  where
    inpLen = fromIntegral $ V.length $ netInp False gym initSt


nnConfig :: Gym -> Float -> NNConfig
nnConfig gym maxRew =
  NNConfig
    { _replayMemoryMaxSize = 10000
    , _replayMemoryStrategy = ReplayMemoryPerAction
    , _trainBatchSize = 8
    , _trainingIterations = 1
    , _grenadeLearningParams = OptAdam 0.001 0.9 0.999 1e-8 1e-3
    , _grenadeSmoothTargetUpdate = 0.01
    , _learningParamsDecay = ExponentialDecay Nothing 0.5 100000
    , _prettyPrintElems = map (V.fromList . zipWith3 (\l u -> scaleValue (Just (l, u))) lows highs) ppSts
    , _scaleParameters = scalingByMaxAbsRewardAlg alg False (1.25 * maxRew)
    , _grenadeDropoutFlipActivePeriod = 0
    , _grenadeDropoutOnlyInactiveAfter = 0
    , _updateTargetInterval = 10000
    , _updateTargetIntervalDecay = StepWiseIncrease (Just 500) 0.1 10000
    }
  where
    (lows, highs) = observationSpaceBounds gym
    mkParamList lo hi = sort $ takeMiddle 3 xs
      where
        xs = map rnd [lo,lo + (hi - lo) / 10 .. hi]
    vals
      | length lows > 4 = []
      | otherwise = zipWith mkParamList lows highs
    rnd x = fromIntegral (round (1000 * x)) / 1000
    ppSts = takeMiddle 20 $ combinations vals

borlSettings :: Settings
borlSettings = def {_workersMinExploration = [0.4, 0.2, 0.1, 0.03], _useProcessForking = False, _nStep = 1, _mainAgentSelectsGreedyActions = True}


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


globalVar :: MVar (Maybe Float)
globalVar = unsafePerformIO $ newMVar Nothing
{-# NOINLINE globalVar #-}
setGlobalVar :: Maybe Float -> IO ()
setGlobalVar x = modifyMVar_ globalVar (return . const x)
{-# NOINLINE setGlobalVar #-}
getGlobalVar :: IO (Maybe Float)
getGlobalVar = join <$> tryReadMVar globalVar
{-# NOINLINE getGlobalVar #-}

action :: Maybe T.Text -> Integer -> Action St
action mName idx =
  flip Action (fromMaybe (T.pack $ show idx) mName) $ \agentType oldSt@(St render _) -> do
    gym <- getGym (fromEnum agentType)
    res <- stepGymRender render gym idx
    rew <- rewardFunction gym oldSt (fromIntegral idx) res
    obs <-
      if episodeDone res
        then resetGym gym
        else return (observation res)
    return (rew, St render (gymObservationToFloatList obs), episodeDone res)


rewardFunction :: Gym -> St -> ActionIndex -> GymResult -> IO (Reward St)
rewardFunction gym (St _ oldSt) actIdx (GymResult obs rew eps) =
  case alg of
    AlgDQN {} -> return $ Reward $ realToFrac rew
    AlgDQNAvgRewAdjusted  {}
      | name gym == "CartPole-v1" -> do
        epsStep <- getElapsedSteps gym
        let rad = xs !! 3 -- (-0.418879, 0.418879) .. 24 degrees in rad
            pos = xs !! 1 -- (-4.8, 4.8)
        return $ Reward $ realToFrac $ (100 *) $ 0.41887903213500977 - abs rad - 0.418879 / 4.8 * abs pos - ite (eps && epsStep /= Just maxEpsSteps) 1.0 0
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
        return $ Reward $ realToFrac $ (20 *) $ ite (eps && epsStep < Just maxEpsSteps) (* 1.2) id $ ite (pos > (-0.3) && velocity >= 0 || pos < (-0.3) && velocity <= 0) height 0
    AlgDQNAvgRewAdjusted  {}
      | name gym == "Acrobot-v1" ->
        let [cosS0, sinS0, cosS1, sinS1, thetaDot1, thetaDot2] = xs -- cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2
         in return $ Reward $ realToFrac $ (* 20) $ -cosS0 - cos (acos cosS0 + acos cosS1)
                           -- -cos(s[0]) - cos(s[1] + s[0])
    AlgDQNAvgRewAdjusted {}
      | name gym == "Copy-v0" -> return $ Reward $ realToFrac rew
    AlgDQNAvgRewAdjusted {}
      | name gym == "Pong-ram-v0" -> return $ Reward $ realToFrac rew
  where
    xs = gymObservationToFloatList obs
    ite True x _  = x
    ite False _ x = x
    maxEpsSteps = maximumEpisodeSteps (name gym)
rewardFunction gym _ _ _ = error $ "rewardFunction not yet defined for this environment: " ++ T.unpack (name gym)

maxReward :: Gym -> Float
maxReward _ | isAlgDqn alg = 10
maxReward gym | name gym == "CartPole-v1" = 50
              | name gym == "MountainCar-v0" = 50 -- 200
              | name gym == "Copy-v0" = 1.0
              | name gym == "Acrobot-v1" = 50
              -- | name gym == "Pendulum-v0" =
              | name gym == "Pong-ram-v0" = 1
maxReward gym   = error $ "Max Reward (maxReward) function not yet defined for this environment: " ++ T.unpack (name gym)


-- | Scales values to (-1, 1).
netInp :: Bool -> Gym -> St -> V.Vector Float
netInp isTabular gym (St _ st)
  | not isTabular = V.fromList $ zipWith3 (\l u -> scaleValue (Just (l, u))) lowerBounds upperBounds st
  | isTabular = V.fromList $ map (rnd . fst) $ filter snd (stSelector st)
  where
    rnd x = fromIntegral (round (x * 10)) / 10
    (lowerBounds, upperBounds) = observationSpaceBounds gym
    stSelector xs
      -- | name gym == "MountainCar-v0" = [(head xs, True), (5 * (xs !! 1), False)]
      | otherwise = zip xs (repeat True)

observationSpaceBounds :: Gym -> ([Float], [Float])
observationSpaceBounds gym = map (max (-maxVal)) *** map (min maxVal) $ gymRangeToFloatLists $ getGymRangeFromSpace $ observationSpace gym
  where
    maxVal | name gym == "CartPole-v1" = 5
           | otherwise = 1000


mInverseSt :: Gym -> Maybe (NetInputWoAction -> Maybe (Either String St))
mInverseSt gym = Just $ \xs -> Just $ Right $ St True $ zipWith3 (\l u x -> unscaleValue (Just (l, u)) x) lowerBounds upperBounds (V.toList xs)
  where
    (lowerBounds, upperBounds) = observationSpaceBounds gym


instance RewardFuture St where
  type StoreType St = ()


alg :: Algorithm St
alg =
  -- algDQN
  -- AlgDQNAvgRewAdjusted Nothing 0.85 0.99 ByStateValues
  AlgDQNAvgRewAdjusted 0.85 1.0 ByStateValues

-- | Gyms.
gymsMVar :: MVar [Gym]
gymsMVar = unsafePerformIO $ newMVar mempty
{-# NOINLINE gymsMVar #-}

addGym :: Gym -> IO ()
addGym v = modifyMVar_ gymsMVar (return . (++ [v]))

getGym :: Int -> IO Gym
getGym k = maybe (error "Could not get gym in getGym in GymOpenAI.hs") (!!k) <$> tryReadMVar gymsMVar

getName :: IO T.Text
getName = do
  args <- getArgs
  let name
        | not (null args) = T.pack (head args)
        | otherwise = "MountainCar-v0"
             -- "CartPole-v1"
  return name

mkInitSt :: St -> AgentType -> IO St
mkInitSt st MainAgent = return st
mkInitSt _ _          = (\(_,_,st) -> st) <$> mkInitSt'

mkInitSt' :: IO (Gym, Integer, St)
mkInitSt' = do
  name <- getName
  (obs, gym) <- initGym name
  addGym gym
  putStrLn $ "Added Gym: " ++ show gym
  maxEpsSteps <- getMaxEpisodeSteps gym
  putStrLn $ "Default maximum episode steps: " ++ show maxEpsSteps
  setMaxEpisodeSteps gym (maximumEpisodeSteps name)
  let actionNodes = spaceSize (actionSpace gym)
      initState = St False (gymObservationToFloatList obs)
  return (gym, actionNodes, initState)


main :: IO ()
main = do
  args <- getArgs
  putStrLn $ "Received arguments: " ++ show args
  (gym, actionNodes, initState) <- mkInitSt'
  let maxRew
        | length args >= 2 = read (args !! 1)
        | otherwise = maxReward gym
  name <- getName
  let actNames = actionNames name
  let actions = zipWith action (map Just actNames ++ repeat Nothing) [0 .. actionNodes - 1]
  let actFilter :: St -> V.Vector Bool
      actFilter _ = V.replicate (length actions) True
      initValues = Just $ defInitValues {defaultRho = 0, defaultRhoMinimum = 0, defaultR1 = 1}
  putStrLn $ "Actions: " ++ show actions
  putStrLn $ "Observation Space: " ++ show (observationSpaceInfo name)
  putStrLn $ "Enforced observation bounds: " ++ show (observationSpaceBounds gym)
  nn <- randomNetworkInitWith HeEtAl :: IO NN
  -- rl <- mkUnichainGrenadeCombinedNet alg initState (netInp False gym) actions actFilter (params gym maxRew) (decay gym) nn (nnConfig gym maxRew) borlSettings initValues
  -- rl <- mkUnichainTensorflowCombinedNet alg (mkInitSt initState) (netInp False gym) actions actFilter (params gym maxRew) (decay gym) (modelBuilder gym initState actionNodes) (nnConfig gym maxRew) borlSettings initValues
  -- rl <- mkUnichainTensorflow alg initState (netInp False gym) actions actFilter (params gym maxRew) (decay gym) (modelBuilder gym initState actionNodes) (nnConfig gym maxRew) borlSettings initValues
  -- let rl = mkUnichainTabular alg initState (netInp True gym) actions actFilter (params gym maxRew) (decay gym) borlSettings initValues
  rl <-  mkUnichainGrenadeCombinedNet alg (mkInitSt initState) (netInp False gym) actions actFilter (params gym maxRew) (decay gym) (modelBuilderGrenade gym initState actionNodes) (nnConfig gym maxRew) borlSettings initValues
  askUser (mInverseSt gym) True usage cmds qlCmds rl -- maybe increase learning by setting estimate of rho
  where
    cmds = []
    usage = []
    qlCmds = [("f", "flip rendering", s %~ (\(St r xs) -> St (not r) xs))]
 -- | BORL Parameters.
params :: Gym -> Float -> ParameterInitValues
params gym maxRew =
  Parameters
    { _alpha               = 0.03
    , _alphaRhoMin         = 2e-5
    , _beta                = 0.01
    , _delta               = 0.005
    , _gamma               = 0.01
    , _epsilon             = [eps, 0.01]
    , _exploration         = 1.0
    , _learnRandomAbove    = 0.5
    , _zeta                = 0.03
    , _xi                  = 0.005
    }
  where eps | name gym == "MountainCar-v0" = 0.25
            | otherwise = min 1.0 $ max 0.05 $ 0.005 * maxRew

decay :: Gym -> ParameterDecaySetting
decay gym =
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-5) 0.5 30000
      , _alphaRhoMin      = NoDecay
      , _beta             = ExponentialDecay (Just 1e-2) 0.5 50000
      , _delta            = ExponentialDecay (Just 1e-2) 0.5 150000
      , _gamma            = ExponentialDecay (Just 1e-2) 0.5 150000
      , _zeta             = ExponentialDecay (Just 0) 0.5 150000
      , _xi               = NoDecay
      -- Exploration
      , _epsilon          = [NoDecay] -- ExponentialDecay (Just 0.03) 0.05 15000
      , _exploration      = ExponentialDecay (Just minExp) 0.50 (round $ expFact * 50000)
      , _learnRandomAbove = NoDecay
      }
  where minExp | name gym == "MountainCar-v0" = 0.001
               | otherwise = 0.01
        expFact
          | name gym == "MountainCar-v0" = 1.3
          | otherwise = 1


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
