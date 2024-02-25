{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveAnyClass             #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs               #-}
{-# LANGUAGE MultiParamTypeClasses      #-}
{-# LANGUAGE OverloadedLists            #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE TemplateHaskell            #-}
{-# LANGUAGE TupleSections              #-}
{-# LANGUAGE TypeFamilies               #-}

-- | Re-Implementation of CartPole-v1
-- see https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
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
import           Control.Monad          (foldM, liftM, unless, when)
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
import           Grenade
import           Prelude                hiding (Left, Right)
import           System.Directory
import           System.FilePath.Posix  ((</>))
import           System.IO
import           System.IO.Unsafe
import           System.Random
import           Text.Printf

import           Debug.Trace

gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
poleLength = 0.5  -- actually half the pole's length
polemass_length = masspole * poleLength
force_mag = 10.0
tau = 0.02  -- seconds between state updates
kinematics_integrator = "euler"

-- Angle at which to fail the episode
theta_threshold_radians = 12 * 2 * pi / 360
x_threshold = 2.4

-- Angle limit set to 2 * theta_threshold_radians so failing observation
-- is still within bounds.
-- high = [
--   x_threshold * 2,
--     np.finfo(np.float32).max,
--     theta_threshold_radians * 2,
--     np.finfo(np.float32).max,
--     ]

data Act = Left | Right
  deriving (Show, Eq, Ord, Enum, Bounded, Generic, NFData, Serialize)


data St =
  St { stX                     :: Double
     , stXDot                  :: Double
     , stTheta                 :: Double
     , stThetaDot              :: Double
     , stStepsBeyondTerminated :: Maybe Int
     }
  deriving (Show, Eq, Ord, NFData, Generic, Serialize)

reset :: IO St
reset = St <$> randomRIO (-0.05, 0.05) <*> randomRIO (-0.05, 0.05) <*> randomRIO (-0.05, 0.05) <*> randomRIO (-0.05, 0.05) <*> pure Nothing

actionFun :: ARAL St Act -> AgentType -> St -> [Act] -> IO (Reward St, St, EpisodeEnd)
actionFun _ _ (St x xDot theta thetaDot stepsBeyondTerminated) [action] = do
  let force =
        case action of
          Right -> force_mag
          Left  -> -force_mag
      costheta = cos theta
      sintheta = sin theta
      -- # For the interested reader:
      -- # https://coneural.org/florian/papers/05_cart_pole.pdf
      temp = (force + polemass_length * thetaDot ** 2 * sintheta) / total_mass
      thetaacc = (gravity * sintheta - costheta * temp) / (poleLength * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass))
      xacc = temp - polemass_length * thetaacc * costheta / total_mass
  let st' =
        if kinematics_integrator == "euler"
          then let x' = x + tau * xDot
                   xDot' = xDot + tau * xacc
                   theta' = theta + tau * thetaDot
                   thetaDot' = thetaDot + tau * thetaacc
                in St x' xDot' theta' thetaDot' Nothing
              -- semi-implicit euler
          else let xDot' = xDot + tau * xacc
                   x' = x + tau * xDot'
                   thetaDot' = thetaDot + tau * thetaacc
                   theta' = theta + tau * thetaDot'
                in St x' xDot' theta' thetaDot' Nothing
      terminated = x < (-x_threshold) || x > x_threshold || theta < (-theta_threshold_radians) || theta > theta_threshold_radians
      (reward, stepsBeyondTerminated')
        | not terminated = (1.0, Nothing)
        | otherwise =
          case stepsBeyondTerminated of
            Nothing -> (1.0, Just 0) -- Pole just fell!
            Just nr -> (0.0, Just $ nr + 1)
  let rewardNew = Reward . (12 - ) . toDegrees . abs . stTheta $ st'
      toDegrees = (360 / (2 * pi) *)
  if isJust stepsBeyondTerminated
    then (\st' -> (reward, st', terminated)) <$> reset
    else return (rewardNew, st' {stStepsBeyondTerminated = stepsBeyondTerminated'}, terminated)


expSetup :: ARAL St Act -> ExperimentSetting
expSetup borl =
  ExperimentSetting
    { _experimentBaseName = "cartpole_correct_epslen"
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
        -- eLength = fromIntegral eSteps / max 1 (fromIntegral eNr)
        eLength = fromIntegral (expSetup rl ^. evaluationSteps) / max 1 (fromIntegral eNr)
        p = Just $ fromIntegral $ rl' ^. t
        val l = realToFrac $ head $ fromValue (rl' ^?! l)
        results | phase /= EvaluationPhase =
                  [ StepResult "reward" p (realToFrac (rl' ^?! lastRewards._head))
                  , StepResult "avgEpisodeLength" p eLength
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
                                                                  -- , AlgDQN 0.99 EpsilonSensitive
                                                                  -- , AlgDQN 0.5 EpsilonSensitive
                                                                  , AlgDQN 0.999 Exact
                                                                  , AlgDQN 0.99 Exact
                                                                  , AlgDQN 0.50 Exact
								  , AlgRLearning
                                                                  ]) Nothing Nothing Nothing

    -- , ParameterSetup "init lr" (set (B.parameters . gamma) (view (B.parameters . gamma)))
    --   (Just $ const $ return [0.025, 0.05, 0.1]) Nothing Nothing Nothing
    ]

  -- beforeEvaluationHook :: ExperimentNumber -> RepetitionNumber -> ReplicationNumber -> GenIO -> a -> ExpM a a
  beforeEvaluationHook expNr repetNr repNr _ rl = do
    mapM_ (moveFileToSubfolder rl expNr repetNr) (["reward", "stateValues", "stateValuesAgents", "queueLength", "episodeLength"] :: [FilePath])
    return $ set episodeNrStart (0, 0) $ set (B.parameters . exploration) 0.00 $ set (B.settings . disableAllLearning) True rl


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
    }

-- | ARAL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha               = 0.10
    , _alphaRhoMin         = 2e-5
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
      , _epsilon          = [NoDecay] -- ExponentialDecay (Just 5.0) 0.5 150000
      , _exploration      = ExponentialDecay (Just 0.01) 0.8 10000 -- ExponentialDecay (Just 0.01) 0.50 100000
      , _learnRandomAbove = NoDecay
      }

initVals :: InitValues
initVals = InitValues 0 0 0 0 0 0

main :: IO ()
main = do
  $(initLogger) (LogFile "package.log")
  setMinLogLevel LogWarning -- LogDebug -- LogInfo
  -- enableARALLogging (LogFile "package.log")


  putStr "Experiment or user mode [User mode]? Enter e for experiment mode, or u for user mode: " >> hFlush stdout
  l <- getLine
  case l of
    -- "l"   -> lpMode
    "e"   -> experimentMode
    "exp" -> experimentMode
    _     -> usermode


experimentMode :: IO ()
experimentMode = do
  let databaseSetup = DatabaseSetting "host=localhost dbname=experimenter user=experimenter password= port=5432" 10
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


usermode :: IO ()
usermode = do
  $(initLogger) LogStdOut
  setMinLogLevel LogAll

  alg <- chooseAlg mRefState

  -- Approximate all fucntions using a single neural network
  -- rl <- mkUnichainGrenadeCombinedNet alg (liftInitSt initState) netInp actionFun actFilter params decay modelBuilderGrenade nnConfig borlSettings (Just initVals)
  -- rl <- mkUnichainGrenade alg (liftInitSt initState) netInp actionFun actFilter params decay modelBuilderGrenade nnConfig borlSettings (Just initVals)
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
    cmdDrawGrid = ("d", "Draw grid", \rl -> drawGrid rl >> return rl)


modelBuilderHasktorch :: Integer -> (Integer, Integer) -> MLPSpec
modelBuilderHasktorch lenIn (lenActs, cols) = MLPSpec [lenIn, 20, 10, 10, lenOut] (HasktorchActivation HasktorchRelu []) (Just HasktorchTanh)
  where
    lenOut = lenActs * cols


-- | The definition for a feed forward network using the dynamic module. Note the nested networks. This network clearly is over-engeneered for this example!
modelBuilderGrenade :: Integer -> (Integer, Integer) -> IO SpecConcreteNetwork
modelBuilderGrenade lenIn (lenActs, cols) =
  buildModelWith (NetworkInitSettings UniformInit HMatrix Nothing) def $
  inputLayer1D lenIn >>
  fullyConnected 20 >> relu >> -- dropout 0.90 >>
  fullyConnected 10 >> relu >>
  fullyConnected 10 >> relu >>
  fullyConnected lenOut >> reshape (lenActs, cols, 1) >> tanhLayer
  where
    lenOut = lenActs * cols


-- netInp :: St -> V.Vector Double
-- netInp (St x xDot theta thetaDot _) =
--   -- V.fromList [fromIntegral . fst . getCurrentIdx $ st, fromIntegral . snd . getCurrentIdx $ st]
--   V.fromList [scaleMinMax (0, fromIntegral maxX) $ fromIntegral $ fst (getCurrentIdx st),
--               scaleMinMax (0, fromIntegral maxY) $ fromIntegral $ snd (getCurrentIdx st)]

tblInp :: St -> V.Vector Double
tblInp (St x xDot theta thetaDot _) =
  V.fromList
    [ min steps . max (-steps) $ fromInteger $ round $ (steps*) $ scaleMinMax (-x_threshold, x_threshold) x		-- in (-2.4,2.4)
    , min steps . max (-steps) $ fromInteger $ round xDot -- $ (steps*) $ scaleMinMax (-vInf, vInf) xDot			-- in (-Inf, Inf)
    , min steps . max (-steps) $ fromInteger $ round $ (steps*) $ scaleMinMax (-12, 12) (360 / (2 * pi) * theta)	-- in (-12, 12)
    , min steps . max (-steps) $ fromInteger $ round thetaDot -- $ (steps*) $ scaleMinMax (-vInf, vInf) thetaDot			-- in (-Inf, Inf)
    ]
  where
    steps = 5.0 -- there are (2*steps+1)  buckets
    vInf  = 1


-- -- State
-- data St = St Int Int deriving (Eq, NFData, Generic, Serialize)

-- instance Ord St where
--   x <= y = fst (getCurrentIdx x) < fst (getCurrentIdx y) || (fst (getCurrentIdx x) == fst (getCurrentIdx y) && snd (getCurrentIdx x) < snd (getCurrentIdx y))

-- instance Show St where
--   show xs = show (getCurrentIdx xs)

-- instance Enum St where
--   fromEnum st = let (x,y) = getCurrentIdx st
--                 in x * (maxX + 1) + y
--   toEnum x = fromIdx (x `div` (maxX+1), x `mod` (maxX+1))

-- instance Bounded St where
--   minBound = fromIdx (0,0)
--   maxBound = fromIdx (maxX, maxY)


-- -- Actions
-- data Act = Random | Up | Down | Left | Right
--   deriving (Eq, Ord, Enum, Bounded, Generic, NFData, Serialize)

-- instance Show Act where
--   show Random = "random"
--   show Up     = "up    "
--   show Down   = "down  "
--   show Left   = "left  "
--   show Right  = "right "

actions :: [Act]
actions = [Left, Right]


actFilter :: St -> [V.Vector Bool]
actFilter _ = [V.fromList [True, True]]


-- allStateInputs :: M.Map NetInputWoAction St
-- allStateInputs = M.fromList $ zip (map netInp [minBound..maxBound]) [minBound..maxBound]


drawGrid :: ARAL St Act -> IO ()
drawGrid aral = do
  putStrLn "TODO"
  -- putStr "\n    "
  -- mapM_ (putStr . printf "%2d ") ([0 .. maxY] :: [Int])
  -- putStr "\n"
  -- mapM_
  --   (\x -> do
  --      putStr (printf "%2d: " x)
  --      mapM_ (drawField aral . St x) ([0 .. maxY] :: [Int])
  --      putStr "\n")
  --   ([0 .. maxX] :: [Int])


drawField :: ARAL St Act -> St -> IO ()
drawField aral s = do
  acts <- map snd . VB.toList <$> nextActionFor MainAgent aral Greedy s 0
  putStr $
    case acts of
      [0] -> " * "
      [1] -> " ^ "
      [2] -> " v "
      [3] -> " < "
      [4] -> " > "
      _   -> error "unexpected action"
