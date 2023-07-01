{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveAnyClass             #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs               #-}
{-# LANGUAGE OverloadedLists            #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE TemplateHaskell            #-}
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
--  $ pip install gym==0.21 --user
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
import           Control.Monad.IO.Class
import           Data.Default
import           Data.List               (genericLength, sort)
import           Data.Maybe              (fromMaybe)
import           Data.Serialize
import qualified Data.Text               as T
import qualified Data.Vector.Storable    as V
import           EasyLogger
import           GHC.Generics
import           GHC.Int                 (Int64)
import           Grenade
import           System.Environment      (getArgs)
import           System.IO
import           System.IO.Unsafe        (unsafePerformIO)

import           Experimenter
import           Helper
import           ML.ARAL                 as B
import           ML.Gym

import           Debug.Trace


expSetup :: ARAL St Act -> ExperimentSetting
expSetup borl =
  ExperimentSetting
    { _experimentBaseName = "gym_" <> unsafePerformIO getName
    , _experimentInfoParameters = [isNN]
    , _experimentRepetitions = 30
    , _preparationSteps = 500000
    , _evaluationWarmUpSteps = 0
    , _evaluationSteps = 10000
    , _evaluationReplications = 1
    , _evaluationMaxStepsBetweenSaves = Just 20000
    }
  where
    isNN = ExperimentInfoParameter "Is neural network" (isNeuralNetwork (borl ^. proxies . v))

evals :: [StatsDef s]
evals =
  [
    Name "Exp Mean of Repl. Mean Reward" $ Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Stats $ Sum OverPeriods (Of "reward")
  , Name "Repl. Mean Reward" $ Mean OverReplications $ Stats $ Sum OverPeriods (Of "reward")
  , Name "Exp StdDev of Repl. Mean Reward" $ StdDev OverExperimentRepetitions $ Stats $ Mean OverReplications $ Stats $ Sum OverPeriods (Of "reward")
  -- , Mean OverExperimentRepetitions $ Stats $ StdDev OverReplications $ Stats $ Sum OverPeriods (Of "reward")
  , Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Last (Of "avgRew")
  , Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Last (Of "avgEpisodeLength")
  , Name "Exp Mean of Repl. Mean Steps to Goal" $ Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Last (Of "avgEpisodeLength")
  , Name "Repl. Mean Steps to Goal" $ Mean OverReplications $ Last (Of "avgEpisodeLength")
  , Name "Exp StdDev of Repl. Mean Steps to Goal" $ StdDev OverExperimentRepetitions $ Stats $ Mean OverReplications $ Last (Of "avgEpisodeLength")
  -- , Mean OverExperimentRepetitions $ Stats $ StdDev OverReplications $ Last (Of "avgEpisodeLength")
  ]

instance ExperimentDef (ARAL St Act)
  -- type ExpM (ARAL St Act) = TF.SessionT IO
                                          where
  type ExpM (ARAL St Act) = IO
  type InputValue (ARAL St Act) = ()
  type InputState (ARAL St Act) = ()
  type Serializable (ARAL St Act) = ARALSerialisable St Act
  serialisable = toSerialisable
  deserialisable :: Serializable (ARAL St Act) -> ExpM (ARAL St Act) (ARAL St Act)
  deserialisable x = fromSerialisable actionFun actFilter (netInp True gym) x
    where actFilter :: St -> [V.Vector Bool]
          actFilter _ = [V.replicate (fromIntegral actionNodes) True]
          actionNodes = spaceSize (actionSpace gym)
          gym = unsafePerformIO $ getGym 0
  generateInput _ _ _ _ = return ((), ())
  runStep phase rl _ _ = do
    rl' <- stepM rl
    let inverseSt -- | isAnn rl = mInverseSt
                  | otherwise = Nothing
    when (rl' ^. t `mod` 10000 == 0) $ liftIO $ prettyARALHead True inverseSt rl' >>= print
    let (eNr, eSteps) = rl ^. episodeNrStart
        eLength = fromIntegral eSteps / max 1 (fromIntegral eNr)
        p = Just $ fromIntegral $ rl' ^. t
        val l = realToFrac $ head $ fromValue (rl' ^?! l)
        results | phase /= EvaluationPhase =
                  [
                  -- StepResult "reward" p (realToFrac (rl' ^?! lastRewards._head))
                  -- , StepResult "avgEpisodeLength" p eLength
                  ]
                | otherwise =
                  [ StepResult "reward" p (realToFrac $ rl' ^?! lastRewards._head)
                  , StepResult "avgRew" p (realToFrac $ V.head (rl' ^?! proxies . rho . proxyScalar))
                  , StepResult "psiRho" p (val $ psis . _1)
                  , StepResult "psiV" p (val $ psis . _2)
                  , StepResult "psiW" p (val $ psis . _3)
                  , StepResult "avgEpisodeLength" p eLength
                  , StepResult "avgEpisodeLengthNr" (Just $ fromIntegral eNr) eLength
                  ] -- ++
                  -- concatMap
                  --   (\s ->
                  --      map (\a -> StepResult (T.pack $ show (s, a)) p (M.findWithDefault 0 (tblInp s, a) (rl' ^?! proxies . r1 . proxyTable))) (filteredActionIndexes actions actFilter s))
                  --   (sort [(minBound :: St) .. maxBound])
    return (results, rl')
  parameters _ =
    [ParameterSetup "algorithm" (set algorithm) (view algorithm) (Just $ const $ return
                                                                  [ AlgARAL 0.8 1.0 ByStateValues
                                                                  , AlgARAL 0.8 0.999 ByStateValues
                                                                  , AlgARAL 0.8 0.99 ByStateValues
                                                                  -- don't use -- , AlgDQN 0.99 EpsilonSensitive
                                                                  -- don't use -- , AlgDQN 0.5 EpsilonSensitive
                                                                  , AlgDQN 0.999 Exact
                                                                  , AlgDQN 0.99 Exact
                                                                  , AlgDQN 0.50 Exact
								  , AlgRLearning


                                                                  ---------------------------
                                                                  ---- Sensitivity for Gamma:
                                                                  ---------------------------
                                                                  --   AlgARAL 0.9 1.0 ByStateValues
                                                                  -- , AlgARAL 0.9 0.999 ByStateValues
                                                                  -- , AlgARAL 0.9 0.99 ByStateValues
                                                                  -- , AlgARAL 0.5 1.0 ByStateValues
                                                                  -- , AlgARAL 0.5 0.999 ByStateValues
                                                                  -- , AlgARAL 0.5 0.99 ByStateValues

                                                                  ]) Nothing Nothing Nothing
      -------------------------------
      ---- Sensitivity for Learn Rate
      -------------------------------

    -- , ParameterSetup "init lr" (set (B.parameters . B.gamma)) (view (B.parameters . B.gamma))
    --   (Just $ const $ return [0.2, 0.3]) Nothing Nothing Nothing
    --   -- (Just $ const $ return [0.025, 0.05, 0.1]) Nothing Nothing Nothing
    ]
  beforeEvaluationHook _ _ _ _ rl = return $ set episodeNrStart (0, 0) $ set (B.parameters . exploration) 0.00 $ set (B.settings . disableAllLearning) True rl


type Render = Bool

data St = St Render [Double]
  deriving (Generic, NFData, Serialize)

instance Eq St where
  (St _ xs1) == (St _ xs2) = xs1 == xs2

instance Ord St where
  (St _ xs1) `compare` (St _ xs2) = xs1 `compare` xs2

instance Show St where
  show (St _ xs) = showDoubleList xs


maxX,maxY :: Int
maxX = 4                        -- [0..maxX]
maxY = 4                        -- [0..maxY]


type NN = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 5, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 5, 'D1 5]
type NNCombined = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 40, Relu, FullyConnected 40 40, Relu, FullyConnected 40 30, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 40, 'D1 40, 'D1 40, 'D1 40, 'D1 30, 'D1 30]
type NNCombinedAvgFree = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 6, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 6, 'D1 6]

modelBuilderGrenade :: Integer -> (Integer, Integer) -> IO SpecConcreteNetwork
modelBuilderGrenade lenIn (lenActs, cols) =
  buildModelWith (def { cpuBackend = BLAS }) def $
  inputLayer1D lenIn >>
  -- fullyConnected (20*lenIn) >> relu >> dropout 0.90 >>
  fullyConnected (10 * lenIn) >> relu >>
  fullyConnected (5 * lenIn) >> relu >>
  fullyConnected (2*lenOut) >> relu >>
  fullyConnected lenOut >> reshape (lenActs, cols, 1) >> tanhLayer
  where
    lenOut = lenActs * cols


nnConfig :: Gym -> Double -> NNConfig
nnConfig gym maxRew =
  NNConfig
    { _replayMemoryMaxSize = 1000
    , _replayMemoryStrategy = ReplayMemoryPerAction
    , _trainBatchSize = 8
    , _trainingIterations = 1
    , _grenadeLearningParams = OptAdam 0.001 0.9 0.999 1e-8 1e-3
    , _grenadeSmoothTargetUpdate = 0.01
    , _grenadeSmoothTargetUpdatePeriod = 1
    , _learningParamsDecay = ExponentialDecay Nothing 0.5 100000
    , _prettyPrintElems = -- map (netInp True gym) ppSts
        map (V.fromList . zipWith3 (\l u -> scaleMinMax (l, u)) lows highs) ppSts
    , _scaleParameters = scalingByMaxAbsRewardAlg alg False (1.25 * maxRew)
    , _scaleOutputAlgorithm = ScaleMinMax
    , _cropTrainMaxValScaled = Just 0.98
    , _grenadeDropoutFlipActivePeriod = 0
    , _grenadeDropoutOnlyInactiveAfter = 0
    , _clipGradients = ClipByGlobalNorm 0.01
    , _autoNormaliseInput = True
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


globalVar :: MVar (Maybe Double)
globalVar = unsafePerformIO $ newMVar Nothing
{-# NOINLINE globalVar #-}
setGlobalVar :: Maybe Double -> IO ()
setGlobalVar x = modifyMVar_ globalVar (return . const x)
{-# NOINLINE setGlobalVar #-}
getGlobalVar :: IO (Maybe Double)
getGlobalVar = join <$> tryReadMVar globalVar
{-# NOINLINE getGlobalVar #-}

actionNrVar :: MVar Int
actionNrVar = unsafePerformIO $ newMVar (-1)
{-# NOINLINE actionNrVar #-}
setActionNrVar :: Int -> IO ()
setActionNrVar x = modifyMVar_ actionNrVar (return . const x)
{-# NOINLINE setActionNrVar #-}
getActionNrVar :: IO Int
getActionNrVar = fromMaybe (error "empty actionNrVar in getActionNrVar") <$> tryReadMVar actionNrVar
{-# NOINLINE getActionNrVar #-}


-- Actions
data Act = Act Int
  deriving (Show, Eq, Ord, NFData, Generic, Serialize)

instance Enum Act where
  fromEnum (Act n) = n
  toEnum n = Act n

instance Bounded Act where
  minBound = toEnum 0
  maxBound = toEnum $ unsafePerformIO getActionNrVar - 1

actionFun :: ActionFunction St Act
actionFun agentType oldSt@(St render _) [Act idx] = do
  gym <- getGym (fromEnum agentType)
  res <- stepGymRender render gym (fromIntegral idx)
  rew <- rewardFunction gym oldSt (fromIntegral idx) res
  obs <-
    if episodeDone res
      then resetGym gym
      else return (observation res)
  return (rew, St render (gymObservationToDoubleList obs), episodeDone res)


rewardFunction :: Gym -> St -> ActionIndex -> GymResult -> IO (Reward St)
rewardFunction gym (St _ oldSt) actIdx (GymResult obs rew eps) =
  case name gym of
    "CartPole-v1" -> do
        epsStep <- getElapsedSteps gym
        let velTip = xs !! 3 -- (-inf, inf)
            vel = xs !! 1    -- (-inf, inf)
            ang = xs !! 2    -- (-0.418879, 0.418879) .. 24 degrees in rad
            pos = xs !! 0    -- (-4.8, 4.8)
        return $ Reward (10 - 10 * abs ang / 0.418879)
          -- rew -- realToFrac $ (100 *) $ 0.41887903213500977 - abs rad - 0.418879 / 4.8 * abs pos - ite (eps && epsStep /= Just maxEpsSteps) 1.0 0
        -- return $ Reward $ realToFrac $ (100 *) $ 0.41887903213500977 - abs rad - 0.418879 / 4.8 * abs pos - ite (eps && epsStep /= Just maxEpsSteps) 1.0 0
    "MountainCar-v0" -> do
        let pos = head xs
            height = sin (3 * pos) * 0.45 + 0.55
            velocity = xs !! 1
            oldPos = head oldSt
            oldVelocity = oldSt !! 1
        step <- fromMaybe 0 <$> getGlobalVar
        setGlobalVar (Just $ step + 1)
        let movGoal = min 0.5 (5e-6 * step - 0.3)
        epsStep <- getElapsedSteps gym
        -- return $ Reward $ realToFrac $ (20 *) $ ite (eps && epsStep < Just maxEpsSteps) (* 1.2) (* 1) $ ite (pos > (-0.3) && velocity >= 0 || pos < (-0.3) && velocity <= 0) height 0
        return $ Reward height
    "Acrobot-v1" ->
        let [cosS0, sinS0, cosS1, sinS1, thetaDot1, thetaDot2] = xs -- cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2
         in return $ Reward $ realToFrac $ (* 20) $ -cosS0 - cos (acos cosS0 + acos cosS1)
                           -- -cos(s[0]) - cos(s[1] + s[0])
    "Copy-v0" -> return $ Reward $ realToFrac rew
    "Pong-ram-v0" -> return $ Reward $ realToFrac rew
    _ -> return $ Reward $ realToFrac rew
  where
    xs = gymObservationToDoubleList obs
    ite True x _  = x
    ite False _ x = x
    maxEpsSteps = maximumEpisodeSteps (name gym)
rewardFunction gym _ _ _ = error $ "rewardFunction not yet defined for this environment: " ++ T.unpack (name gym)

maxReward :: Gym -> Double
maxReward _ | isAlgDqn alg = 10
maxReward gym | name gym == "CartPole-v1" = 1
              | name gym == "MountainCar-v0" = 25 -- 200
              | name gym == "Copy-v0" = 1.0
              | name gym == "Acrobot-v1" = 50
              -- | name gym == "Pendulum-v0" =
              | name gym == "Pong-ram-v0" = 1
maxReward gym   = error $ "Max Reward (maxReward) function not yet defined for this environment: " ++ T.unpack (name gym)


-- | Scales values to (-1, 1).
netInp :: Bool -> Gym -> St -> V.Vector Double
netInp isTabular gym (St _ st)
  | not isTabular = V.fromList $ zipWith3 (\l u -> scaleMinMax (l, u)) lowerBounds upperBounds st
  | isTabular =
    V.fromList $ zipWith3 (\l u v -> fromIntegral . round $ ((10 * v) / (u - max 0 l))) lowerBounds upperBounds st
    -- V.fromList $ map (rnd 10 . fst) $ filter snd (stSelector st)
  where
    rnd n x = fromIntegral (round (x * n)) / n
    (lowerBounds, upperBounds) = observationSpaceBounds gym
    stSelector xs
      -- | name gym == "MountainCar-v0" = [(head xs, True), (5 * (xs !! 1), False)]
      | otherwise = zip xs (repeat True)

observationSpaceBounds :: Gym -> ([Double], [Double])
observationSpaceBounds gym = map (max (-maxVal)) *** map (min maxVal) $ gymRangeToDoubleLists $ getGymRangeFromSpace $ observationSpace gym
  where
    maxVal | name gym == "CartPole-v1" = 20
           | otherwise = 25


mInverseSt :: Gym -> Maybe (NetInputWoAction -> Maybe (Either String St))
mInverseSt gym = Nothing -- Just $ \xs -> Just $ Right $ St True $ zipWith3 (\l u x -> unscaleMinMax (l, u) x) lowerBounds upperBounds (V.toList xs)
  where
    (lowerBounds, upperBounds) = observationSpaceBounds gym


instance RewardFuture St where
  type StoreType St = ()


alg :: Algorithm St
alg =
  -- algDQN
  -- AlgARAL Nothing 0.85 0.99 ByStateValues
  AlgARAL 0.85 1.0 ByStateValues

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
      initState = St False (gymObservationToDoubleList obs)
  return (gym, actionNodes, initState)


main :: IO ()
main = do
  $(initLogger) (LogFile "package.log")
  setMinLogLevel LogWarning
  enableARALLogging (LogFile "package.log")
  name <- getName
  putStrLn $ "Using environment: " <> T.unpack name
  putStr "Experiment or user mode [User mode]? Enter e for experiment mode, and u for user mode: " >> hFlush stdout
  l <- getLine
  case l of
    -- "l"   -> lpMode
    "e"   -> experimentMode
    "exp" -> experimentMode
    _     -> usermode

experimentMode :: IO ()
experimentMode = do
  args <- getArgs
  putStrLn $ "Received arguments: " ++ show args
  (gym, actionNodes, initState) <- mkInitSt'
  setActionNrVar (fromIntegral actionNodes)
  let maxRew
        | length args >= 2 = read (args !! 1)
        | otherwise = maxReward gym
  name <- getName
  let actNames = actionNames name
  -- let actions = zipWith actionFun (map Just actNames ++ repeat Nothing) [0 .. actionNodes - 1]
  let actFilter :: St -> [V.Vector Bool]
      actFilter _ = [V.replicate (fromIntegral actionNodes) True]
      initValues = Just $ defInitValues {defaultRho = 0, defaultRhoMinimum = 0, defaultR1 = 1}
  putStrLn $ "Actions Count: " ++ show actionNodes
  putStrLn $ "Observation Space: " ++ show (observationSpaceInfo name)
  putStrLn $ "Enforced observation bounds: " ++ show (observationSpaceBounds gym)

  let databaseSetup = DatabaseSetting "host=localhost dbname=experimenter user=experimenter password= port=5432" 10
  ---
  rl <- mkUnichainTabular alg (mkInitSt initState) (netInp True gym) actionFun actFilter (params gym maxRew) (decay gym) borlSettings initValues
  -- rl <- mkUnichainTabular (AlgARAL 0.8 1.0 ByStateValues) (liftInitSt initState) tblInp actionFun actFilter params decay borlSettings (Just initVals)
  (changed, res) <- runExperiments liftIO databaseSetup expSetup () rl
  let runner = liftIO
  ---
  putStrLn $ "Any change: " ++ show changed
  evalRes <- genEvalsConcurrent 6 runner databaseSetup res evals
     -- print (view evalsResults evalRes)
  writeAndCompileLatex databaseSetup evalRes
  writeCsvMeasure databaseSetup res NoSmoothing ["reward", "avgEpisodeLength"]


usermode :: IO ()
usermode = do
  args <- getArgs
  putStrLn $ "Received arguments: " ++ show args
  (gym, actionNodes, initState) <- mkInitSt'
  setActionNrVar (fromIntegral actionNodes)
  let maxRew
        | length args >= 2 = read (args !! 1)
        | otherwise = maxReward gym
  name <- getName
  let actNames = actionNames name
  -- let actions = zipWith actionFun (map Just actNames ++ repeat Nothing) [0 .. actionNodes - 1]
  let actFilter :: St -> [V.Vector Bool]
      actFilter _ = [V.replicate (fromIntegral actionNodes) True]
      initValues = Just $ defInitValues {defaultRho = 0, defaultRhoMinimum = 0, defaultR1 = 1}
  putStrLn $ "Actions Count: " ++ show actionNodes
  putStrLn $ "Observation Space: " ++ show (observationSpaceInfo name)
  putStrLn $ "Enforced observation bounds: " ++ show (observationSpaceBounds gym)
  -- rl <- mkUnichainGrenadeCombinedNet alg initState (netInp False gym) actions actFilter (params gym maxRew) (decay gym) (modelBuilderGrenade gym initState actionNodes) (nnConfig gym maxRew) borlSettings initValues
  rl <- mkUnichainTabular alg (mkInitSt initState) (netInp True gym) actionFun actFilter (params gym maxRew) (decay gym) borlSettings initValues
  -- rl <-  mkUnichainGrenade alg (mkInitSt initState) (netInp False gym) actionFun actFilter (params gym maxRew) (decay gym) modelBuilderGrenade (nnConfig gym maxRew) borlSettings initValues
  askUser (mInverseSt gym) True usage cmds qlCmds rl -- maybe increase learning by setting estimate of rho
  where
    cmds = []
    usage = []
    qlCmds = [("f", "flip rendering", return . (s %~ (\(St r xs) -> St (not r) xs)))]


 -- | ARAL Parameters.
params :: Gym -> Double -> ParameterInitValues
params gym maxRew =
  Parameters
    { _alpha               = 0.01
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
  where eps | name gym == "MountainCar-v0" = 0.025
            | otherwise = 0.025 -- min 1.0 $ max 0.05 $ 0.005 * maxRew

decay :: Gym -> ParameterDecaySetting
decay gym =
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-5) 0.5 15000
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
observationSpaceInfo "CartPole-v1"    = ["Cart Position (-4.8, 4.8)", "Cart Velocity (-Inf, Inf)", "Pole Angle (-24 deg, 24 deg)", "Pole Velocity At Tip (-Inf, Inf)"]
observationSpaceInfo "MountainCar-v0" = ["Position (-1.2, 0.6)", "Velocity (-0.07, 0.07)"]
observationSpaceInfo "Acrobot-v1"     = ["cos(theta1)", "sin(theta1)", "cos(theta2)", "sin(theta2)", "thetaDot1", "thetaDot2"]
observationSpaceInfo _                = ["unkown observation space description"]
