{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveAnyClass             #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs               #-}
{-# LANGUAGE MultiParamTypeClasses      #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE TemplateHaskell            #-}
{-# LANGUAGE TupleSections              #-}
{-# LANGUAGE TypeFamilies               #-}
--
-- | See https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
--
--  Run `tail -f render.out` to render the RL process on the command line.
--  Run `sh plot.sh` to plot the output plots via `gnuplot`.
module Main where

import           ML.ARAL                as B
import           ML.ARAL.Logging

import           EasyLogger
import           Experimenter

import           Data.Default
import           Helper

import           Control.Arrow          (first, second, (***))
import           Control.DeepSeq        (NFData)
import           Control.Lens
import           Control.Lens           (set, (^.))
import           Control.Monad          (foldM, forM, liftM, unless, void, when)
import           Control.Monad.IO.Class (liftIO)
import           Data.Default
import           Data.Function          (on)
import           Data.IORef
import           Data.List              (elemIndex, genericLength, groupBy,
                                         sort, sortBy)
import qualified Data.Map.Strict        as M
import           Data.Maybe
import           Data.Serialize
import qualified Data.Text              as T
import           Data.Text.Encoding     as E
import qualified Data.Vector            as VB
import qualified Data.Vector.Storable   as V
import           GHC.Generics
import           GHC.Int                (Int32, Int64)
import           GHC.TypeLits
import           Prelude                hiding (Left, Right)
import           System.Directory
import           System.FilePath.Posix  ((</>))
import           System.IO
import           System.IO.Unsafe
import           System.Random
import           Text.Printf

import           Debug.Trace


-- ### Observation Space

-- The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

-- | Num | Observation                          | Min  | Max | Unit         |
-- |-----|--------------------------------------|------|-----|--------------|
-- | 0   | position of the car along the x-axis | -Inf | Inf | position (m) |
-- | 1   | velocity of the car                  | -Inf | Inf | position (m) |

-- ### Action Space

-- There are 3 discrete deterministic actions:

-- | Num | Observation             | Value | Unit         |
-- |-----|-------------------------|-------|--------------|
-- | 0   | Accelerate to the left  | Inf   | position (m) |
-- | 1   | Don't accelerate        | Inf   | position (m) |
-- | 2   | Accelerate to the right | Inf   | position (m) |


max_steps = 200
min_position = -1.2
max_position = 0.6
min_height = minimum $ map heightPos [min_position - 0.01,min_position .. max_position]
max_height = maximum $ map heightPos [min_position - 0.01,min_position .. max_position]
max_speed = 0.07
goal_position = 0.5
goal_velocity = 0

force = 0.001
gravity = 0.0025

-- low = [self.min_position, -self.max_speed]
-- high = [self.max_position, self.max_speed]


-- -- action_space = spaces.Discrete(3)
-- observation_space = [self.low, self.high]

data Act = Left | NoOp | Right
  deriving (Show, Eq, Ord, Enum, Bounded, Generic, NFData, Serialize)

actIdx :: Act -> Int
actIdx Left  = 0
actIdx NoOp  = 1
actIdx Right = 2

data St =
  St { stPosition :: Double
     , stVelocity :: Double
     , stStep     :: Int
     }
  deriving (Show, Eq, Ord, NFData, Generic, Serialize)

reset :: IO St
reset = St <$> randomRIO (-0.6, -0.4) <*> pure 0 <*> pure 0


actionFun :: ARAL St Act -> AgentType -> St -> [Act] -> IO (Reward St, St, EpisodeEnd)
actionFun aral agentType (St position velocity step) [action] = do

  let velocity' =
        clamp (-max_speed, max_speed) $
        velocity + fromIntegral (actIdx action - 1) * force + cos (3 * position) * (-gravity)
      position' = clamp (min_position, max_position) $ position + velocity'

  let velocity''
        | position' == min_position && velocity' < 0 = 0
        | otherwise = velocity'
      terminated = position' >= goal_position && velocity'' >= goal_velocity
      reward = -1.0
  st' <- if terminated --  || step >= max_steps
         then reset
         else return $ St position' velocity'' (step+1)
  when (agentType == MainAgent) $ void $ render st'
  let currentHeight = heightPos position'
  let rewardNew
        | terminated = 10
        | position < -0.5 && velocity < 0 || position > -0.5 && velocity > 0 = Reward $ -10 + 20 * ((currentHeight - min_height) / (max_height - min_height))
        | otherwise = -10
  return (if terminated then 10 else rewardNew, st', terminated)

  where clamp (low, high) a = min high (max a low)

height :: St -> Double
height = heightPos . stPosition

heightPos :: Double -> Double
heightPos pos = sin (3 * pos) * 0.45 + 0.55


expSetup :: ARAL St Act -> ExperimentSetting
expSetup borl =
  ExperimentSetting
    { _experimentBaseName = "mountaincar_rew"
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
  -- ++
  -- concatMap
  --   (\s -> map (\a -> Mean OverReplications $ First (Of $ E.encodeUtf8 $ T.pack $ show (s, a))) (filteredActionIndexes actions actFilter s))
  --   (sort [(minBound :: St) .. maxBound])
  -- ++
  -- [ Id $ EveryXthElem 10 $ Of "avgRew"
  -- , Mean OverReplications $ EveryXthElem 100 (Of "avgRew")
  -- , StdDev OverReplications $ EveryXthElem 100 (Of "avgRew")
  -- , Mean OverReplications (Stats $ Mean OverPeriods (Of "avgRew"))
  -- , Mean OverReplications $ EveryXthElem 100 (Of "psiRho")
  -- , StdDev OverReplications $ EveryXthElem 100 (Of "psiRho")
  -- , Mean OverReplications $ EveryXthElem 100 (Of "psiV")
  -- , StdDev OverReplications $ EveryXthElem 100 (Of "psiV")
  -- , Mean OverReplications $ EveryXthElem 100 (Of "psiW")
  -- , StdDev OverReplications $ EveryXthElem 100 (Of "psiW")]

  -- where
  --   filterRow f = filter (f . fst . getCurrentIdx)


instance RewardFuture St where
  type StoreType St = ()


instance ExperimentDef (ARAL St Act)
  -- type ExpM (ARAL St Act) = TF.SessionT IO
                                          where
  type ExpM (ARAL St Act) = IO
  type InputValue (ARAL St Act) = ()
  type InputState (ARAL St Act) = ()
  type Serializable (ARAL St Act) = ARALSerialisable St Act
  serialisable = toSerialisable
  deserialisable :: Serializable (ARAL St Act) -> ExpM (ARAL St Act) (ARAL St Act)
  deserialisable = fromSerialisable actionFun actFilter tblInp
  generateInput _ _ _ _ = return ((), ())
  runStep phase rl _ _ = do
    rl' <- stepM rl
    let inverseSt = Nothing
    when (rl' ^. t `mod` 10000 == 0) $ liftIO $ prettyARALHead True inverseSt rl' >>= print
    let (eNr, eSteps) = rl ^. episodeNrStart
        eLength = fromIntegral (expSetup rl ^. evaluationSteps) / max 1 (fromIntegral eNr)
        p = Just $ fromIntegral $ rl' ^. t
        val l = realToFrac $ head $ fromValue (rl' ^?! l)
        results | phase /= EvaluationPhase = []
                | otherwise =
                  [ StepResult "reward" p (realToFrac $ rl' ^?! lastRewards._head)
                  , StepResult "avgRew" p (realToFrac $ V.head (rl' ^?! proxies . rho . proxyScalar))
                  -- , StepResult "psiRho" p (val $ psis . _1)
                  -- , StepResult "psiV" p (val $ psis . _2)
                  -- , StepResult "psiW" p (val $ psis . _3)
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
                                                                  [ -- AlgARAL 0.8 1.0 ByStateValues
                                                                  -- , AlgARAL 0.8 0.999 ByStateValues
                                                                  -- , AlgARAL 0.8 0.99 ByStateValues
                                                                  -- , AlgDQN 0.99 EpsilonSensitive
                                                                  -- , AlgDQN 0.5 EpsilonSensitive
                                                                    AlgDQN 0.999 Exact
                                                                  , AlgDQN 0.99 Exact
                                                                  , AlgDQN 0.50 Exact
								  -- , AlgRLearning
                                                                  ]) Nothing Nothing Nothing

    , ParameterSetup "init lr" (set (B.parameters . gamma)) (view (B.parameters . gamma))
      (Just $ const $ return [0.025, 0.05, 0.1, 0.2]) Nothing Nothing Nothing
    ]

  -- beforeEvaluationHook :: ExperimentNumber -> RepetitionNumber -> ReplicationNumber -> GenIO -> a -> ExpM a a
  beforeEvaluationHook expNr repetNr repNr _ rl = do
    mapM_ (moveFileToSubfolder rl expNr repetNr) (["reward", "stateValues", "stateValuesAgents", "queueLength", "episodeLength"] :: [FilePath])
    let rl' = set episodeNrStart (0, 0) $ set (B.parameters . exploration) 0.00 $ set (B.settings . disableAllLearning) True rl
    st' <- reset
    return $ set s st' rl'


nnConfig :: NNConfig
nnConfig =
  NNConfig
    { _replayMemoryMaxSize = 1000
    , _replayMemoryStrategy = ReplayMemoryPerAction -- ReplayMemorySingle
    , _trainBatchSize = 20
    , _trainingIterations = 1
    , _grenadeLearningParams = OptAdam 0.005 0.9 0.999 1e-8 1e-3
    , _grenadeSmoothTargetUpdate = 0.01
    , _grenadeSmoothTargetUpdatePeriod = 100
    , _learningParamsDecay = ExponentialDecay Nothing 0.05 100000
    , _prettyPrintElems = [] -- map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsRewardAlg (AlgARAL 0.8 1.0 ByStateValues) False 6
    , _scaleOutputAlgorithm = ScaleMinMax
    , _cropTrainMaxValScaled = Just 0.98
    , _grenadeDropoutFlipActivePeriod = 10000
    , _grenadeDropoutOnlyInactiveAfter = 10^5
    , _clipGradients = NoClipping -- ClipByGlobalNorm 0.01
    , _autoNormaliseInput = True
    }


borlSettings :: Settings
borlSettings =
  def
    { _workersMinExploration = [0.1, 0.2, 0.3]
    , _nStep = 1
    , _independentAgents = 1
    , _samplingSteps = 4
    , _overEstimateRho = False -- True
    }

-- | ARAL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha               = 0.2
    , _alphaRhoMin = 2e-5
    , _beta                = 0.01
    , _delta               = 0.005
    , _gamma               = 0.025
    , _epsilon             = 0.025

    , _exploration         = 1.0
    , _learnRandomAbove    = 1.5
    , _zeta                = 0.03
    , _xi                  = 0.005

    }

-- | Decay function of parameters.
decay :: ParameterDecaySetting
decay =
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-5) 0.5 15000  -- 5e-4
      , _alphaRhoMin      = NoDecay
      , _beta             = ExponentialDecay (Just 1e-4) 0.5 150000
      , _delta            = ExponentialDecay (Just 5e-4) 0.5 150000
      , _gamma            = ExponentialDecay (Just 1e-3) 0.5 150000 -- 1e-3
      , _zeta             = ExponentialDecay (Just 0) 0.5 150000
      , _xi               = NoDecay
      -- Exploration
      , _epsilon          = pure NoDecay -- ExponentialDecay (Just 5.0) 0.5 150000
      , _exploration      = ExponentialDecay (Just 0.01) 0.8 10000 -- ExponentialDecay (Just 0.01) 0.50 100000
      , _learnRandomAbove = NoDecay
      }

initVals :: InitValues
initVals = InitValues 0 (-10) 0 0 0 0

main :: IO ()
main = do
  $(initLogger) (LogFile "package.log")
  setMinLogLevel LogWarning -- LogDebug -- LogInfo
  -- enableARALLogging (LogFile "package.log")

  writeFile "render.out" ""

  putStr "Experiment or user mode [User mode]? Enter e for experiment mode, or u for user mode: " >> hFlush stdout
  l <- getLine
  case l of
    -- "l"   -> lpMode
    "e"   -> experimentMode
    "exp" -> experimentMode
    _     -> usermode


experimentMode :: IO ()
experimentMode = do
  writeIORef renderEnabled False
  let databaseSetup = DatabaseSetting "host=localhost dbname=experimenter user=schnecki password= port=5432" 10
  ---
  rl <- mkUnichainTabular (AlgARAL 0.8 1.0 ByStateValues) (const reset) tblInp actionFun actFilter params decay borlSettings (Just initVals)
  (changed, res) <- runExperiments liftIO databaseSetup expSetup () rl
  let runner = liftIO
  ---
  putStrLn $ "Any change: " ++ show changed
  evalRes <- genEvalsConcurrent 6 runner databaseSetup res evals
     -- print (view evalsResults evalRes)
  writeAndCompileLatex databaseSetup evalRes
  writeCsvMeasure databaseSetup res NoSmoothing ["reward", "avgEpisodeLength"]


mRefState :: Maybe (St, ActionIndex)
mRefState = Nothing
-- mRefState = Just (fromIdx (goalX, goalY), 0)


moveFileToSubfolder :: ARAL St Act -> Int -> Int -> FilePath -> IO ()
moveFileToSubfolder rl expNr repNr filename = do
  createDirectoryIfMissing True dirName
  doesFileExist filename >>= \exists -> when exists (renameFile filename fpSub)
  where
    dirName = "results" </> T.unpack (view experimentBaseName . expSetup $ rl) </> "files"
    fpSub = dirName </> "exp" ++ show expNr ++ "_rep" ++ show repNr ++ "_" ++ filename


renderEnabled :: IORef Bool
renderEnabled = unsafePerformIO $ newIORef True
{-# NOINLINE renderEnabled #-}


usermode :: IO ()
usermode = do
  writeIORef renderEnabled True
  $(initLogger) LogStdOut
  setMinLogLevel LogAll

  alg <- chooseAlg mRefState

  -- Approximate all fucntions using a single neural network
  -- rl <- mkUnichainHasktorchAsSAM (Just (1, 0.03)) [minBound..maxBound] alg (liftInitSt initState) netInp actionFun actFilter params decay modelBuilderHasktorch nnConfig borlSettings (Just initVals)
  -- rl <- mkUnichainHasktorchAsSAMAC True Nothing [minBound..maxBound] alg (liftInitSt initState) netInp actionFun actFilter params decay modelBuilderHasktorch nnConfig borlSettings (Just initVals)

  -- Use a table to approximate the function (tabular version)
  rl <- mkUnichainTabular alg (const reset) tblInp actionFun actFilter params decay borlSettings (Just initVals)

  -- let inverseSt | isAnn rl = Just mInverseSt
  --               | otherwise = Nothing

  askUser Nothing True usage cmds [cmdDrawGrid] rl -- maybe increase learning by setting estimate of rho
  where
    cmds = zipWith (\u a -> (fst u, maybe [0] return (elemIndex a actions))) usage [ Left, Right]
    usage = [("j", "Move left"), ("l", "Move right")]
    cmdDrawGrid = ("d", "Render", \rl -> render (rl ^. s) >>= putStrLn >> return rl)


modelBuilderHasktorch :: Integer -> (Integer, Integer) -> MLPSpec
modelBuilderHasktorch lenIn (lenActs, cols) = MLPSpec [lenIn, 20, 10, 10, lenOut] (HasktorchActivation HasktorchRelu []) (Just HasktorchTanh)
  where
    lenOut = lenActs * cols


-- netInp :: St -> V.Vector Double
-- netInp (St x xDot theta thetaDot _) =
--   -- V.fromList [fromIntegral . fst . getCurrentIdx $ st, fromIntegral . snd . getCurrentIdx $ st]
--   V.fromList [scaleMinMax (0, fromIntegral maxX) $ fromIntegral $ fst (getCurrentIdx st),
--               scaleMinMax (0, fromIntegral maxY) $ fromIntegral $ snd (getCurrentIdx st)]

tblInp :: St -> V.Vector Double
tblInp (St pos vel _) =
  V.fromList
    [ min steps . max (-steps) $ fromInteger $ round $ (steps*) $ scaleMinMax (min_position, max_position) pos
    , min steps . max (-steps) $ signum vel
      -- fromInteger $ round $ (steps*) $ scaleMinMax (-max_speed, max_speed) vel
    ]
  where
    steps = 5 -- there are (2*steps+1) buckets


actions :: [Act]
actions = [minBound..maxBound]


actFilter :: St -> [V.Vector Bool]
actFilter _ = [V.fromList [True, True, True]]


-- allStateInputs :: M.Map NetInputWoAction St
-- allStateInputs = M.fromList $ zip (map netInp [minBound..maxBound]) [minBound..maxBound]


render :: St -> IO String
render (St pos _ _) = do
  enabled <- readIORef renderEnabled
  if not enabled
    then return ""
    else do
      let stepX = 0.025
      let stepY = 0.03
      let xs = [min_position - stepX,min_position .. max_position + stepX]
      let ys = [max_height + stepY,max_height .. min_height - stepY]
      let h = heightPos pos
          hGoal = heightPos goal_position
      let inStepX val posV = val - posV >= 0 && val - posV <= stepX
          inStepY val posV = val - posV >= 0 && val - posV <= stepY
      lines <-
        forM ys $ \y -> do
          line <-
            forM xs $ \x -> do
              let sign
                    | inStepX x pos && inStepY y h = 'x'
                 -- | x - pos >= 0 && x - pos <= step = 'x'
                    | inStepX x goal_position && inStepY y hGoal = 'G'
                 -- | x - goal_position >= 0 && x - goal_position <= step = 'G'
                    | inStepY y (heightPos x) = '-'
                    | otherwise = ' '
              return sign
          return (line ++ "\n")
      let out = clear ++ concat lines ++ "  Height: " ++ show h ++ "\n"
      appendFile "render.out" out
      return out

clear = replicate 10 '\n'
