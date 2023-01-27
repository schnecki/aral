-- With the goal state being (0,0) an agent following the optimal policy needs on average 4 steps to the goal state.
-- Thus, it accumulates a reward of 4.0*4 + 10 = 26.0 every 5 steps. That is 5.2 as average reward in the optimal case.
--
-- for 1 agent: rho^\pi^* = 5.20
--

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
module Main where

import           ML.ARAL                  as B
import           ML.ARAL.Logging
import           RegNet
import           SolveLp

import           EasyLogger
import           Experimenter

import           Data.Default
import           Helper

import           Control.Arrow            (first, second, (***))
import           Control.DeepSeq          (NFData)
import           Control.Lens
import           Control.Lens             (set, (^.))
import           Control.Monad            (foldM, liftM, unless, when)
import           Control.Monad.IO.Class   (liftIO)
import           Data.Default
import           Data.Function            (on)
import           Data.IORef
import           Data.List                (elemIndex, genericLength, groupBy, sort, sortBy)
import qualified Data.Map.Strict          as M
import           Data.Serialize
import           Data.Singletons.TypeLits hiding (natVal)
import qualified Data.Text                as T
import           Data.Text.Encoding       as E
import qualified Data.Vector              as VB
import qualified Data.Vector.Storable     as V
import           GHC.Generics
import           GHC.Int                  (Int32, Int64)
import           GHC.TypeLits
import           Grenade
import           Prelude                  hiding (Left, Right)
import           System.IO
import           System.IO.Unsafe
import           System.Random
import           Text.Printf

import           Debug.Trace

maxX, maxY, goalX, goalY :: Int
maxX = 4                        -- [0..maxX]
maxY = 4                        -- [0..maxY]
goalX = 0
goalY = 0


expSetup :: ARAL St Act -> ExperimentSetting
expSetup borl =
  ExperimentSetting
    { _experimentBaseName = "gridworld-mini"
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

instance BorlLp St Act where
  lpActions _ = actions
  lpActionFunction = actionFun
  lpActionFilter _ = head . actFilter

policy :: Policy St Act
policy s a
  | s == fromIdx (goalX, goalY) && a == actRand =
    let distanceSA = concatMap filterDistance $ groupBy ((==) `on` fst) $ sortBy (compare `on` fst) stateActions
        pol = map ((, 1 / fromIntegral (length distanceSA)) . first fromIdx) distanceSA
     in pol
  | s == fromIdx (goalX, goalY) = []
  | a == actRand = []
  | otherwise = mkProbability $ filterChance $ filterDistance $ filter filterActRand [(step sa', actUp), (step sa', actLeft), (step sa', actRight), (step sa', actRand)]
  where
    sa' = ((row, col), a)
    step ((row, col), a)
      | a == actUp = (max 0 $ row - 1, col)
      | a == actDown = (min maxX $ row + 1, col)
      | a == actLeft = (row, max 0 $ col - 1)
      | a == actRight = (row, min maxY $ col + 1)
      | a == actRand = (row, col)
    row = fst $ getCurrentIdx s
    col = snd $ getCurrentIdx s
    actRand = head actions
    actUp = actions !! 1
    actDown = actions !! 2
    actLeft = actions !! 3
    actRight = actions !! 4
    states = [minBound .. maxBound] :: [St]
    stateActions = ((goalX, goalY), actRand) : map (first getCurrentIdx) [(s, a) | s <- states, a <- tail actions, s /= fromIdx (goalX, goalY)]
    filterActRand ((r, c), a)
      | r == goalX && c == goalY = a == actRand
      | otherwise = a /= actRand
    filterChance [x] = [x]
    filterChance xs = filter ((== maximum stepsToBorder) . mkStepsToBorder . step) xs
      where
        stepsToBorder :: [Int]
        stepsToBorder = map (mkStepsToBorder . step) xs :: [Int]
        mkStepsToBorder (r, c) = min (r `mod` maxX) (c `mod` maxY)
    filterDistance xs = filter ((== minimum dist) . mkDistance . step) xs
      where
        dist :: [Int]
        dist = map (mkDistance . step) xs
    mkDistance (r, c) = r + abs (c - goalY)
    mkProbability xs = map (\x -> (first fromIdx x, 1 / fromIntegral (length xs))) xs

fakeEpisodes :: ARAL St Act -> ARAL St Act -> ARAL St Act
fakeEpisodes rl rl'
  | rl ^. s == goal && rl ^. episodeNrStart == rl' ^. episodeNrStart = episodeNrStart %~ (\(nr, t) -> (nr + 1, t + 1)) $ rl'
  | otherwise = episodeNrStart %~ (\(nr, t) -> (nr, t + 1)) $ rl'


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
    let inverseSt | isAnn rl = Just mInverseSt
                  | otherwise = Nothing
    when (rl' ^. t `mod` 10000 == 0) $ liftIO $ prettyARALHead True inverseSt rl' >>= print
    let (eNr, eSteps) = rl ^. episodeNrStart
        eLength = fromIntegral eSteps / max 1 (fromIntegral eNr)
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
    return (results, fakeEpisodes rl rl')
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
                                                                  ]) Nothing Nothing Nothing]
  beforeEvaluationHook _ _ _ _ rl = return $ set episodeNrStart (0, 0) $ set (B.parameters . exploration) 0.00 $ set (B.settings . disableAllLearning) True rl

nnConfig :: NNConfig
nnConfig =
  NNConfig
    { _replayMemoryMaxSize = 1000
    , _replayMemoryStrategy = ReplayMemorySingle -- ReplayMemoryPerAction
    , _trainBatchSize = 8
    , _trainingIterations = 1
    , _grenadeLearningParams = OptAdam 0.005 0.9 0.999 1e-8 1e-3
    , _grenadeSmoothTargetUpdate = 0.01
    , _grenadeSmoothTargetUpdatePeriod = 100
    , _learningParamsDecay = ExponentialDecay Nothing 0.05 100000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
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
    , _samplingSteps = 8
    }

-- | ARAL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha               = 0.01
    , _alphaRhoMin = 2e-5
    , _beta                = 0.01
    , _delta               = 0.005
    , _gamma               = 0.01
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
  enableRegNetLogging LogStdOut


  putStr "Experiment or user mode [User mode]? Enter e for experiment mode, l for lp mode, and u for user mode: " >> hFlush stdout
  l <- getLine
  chooseRandomReward
  case l of
    "l"   -> lpMode
    "e"   -> experimentMode
    "exp" -> experimentMode
    _     -> usermode


experimentMode :: IO ()
experimentMode = do
  let databaseSetup = DatabaseSetting "host=localhost dbname=experimenter user=experimenter password= port=5432" 10
  ---
  rl <- mkUnichainTabular (AlgARAL 0.8 1.0 ByStateValues) (liftInitSt initState) tblInp actionFun actFilter params decay borlSettings (Just initVals)
  (changed, res) <- runExperiments liftIO databaseSetup expSetup () rl
  let runner = liftIO
  ---
  putStrLn $ "Any change: " ++ show changed
  evalRes <- genEvalsConcurrent 6 runner databaseSetup res evals
     -- print (view evalsResults evalRes)
  writeAndCompileLatex databaseSetup evalRes
  writeCsvMeasure databaseSetup res NoSmoothing ["reward", "avgEpisodeLength"]


lpMode :: IO ()
lpMode = do
  putStrLn "I am solving the system using linear programming to provide the optimal solution...\n"
  lpRes <- runBorlLpInferWithRewardRepet 100000 policy mRefState
  print lpRes
  mkStateFile 0.65 False True lpRes
  mkStateFile 0.65 False False lpRes
  putStrLn "NOTE: Above you can see the solution generated using linear programming. Bye!"


mRefState :: Maybe (St, ActionIndex)
mRefState = Nothing
-- mRefState = Just (fromIdx (goalX, goalY), 0)


usermode :: IO ()
usermode = do
  $(initLogger) LogStdOut
  enableRegNetLogging LogStdOut

  alg <- chooseAlg mRefState

  -- Approximate all fucntions using a single neural network
  -- rl <- mkUnichainGrenadeCombinedNet alg (liftInitSt initState) netInp actionFun actFilter params decay modelBuilderGrenade nnConfig borlSettings (Just initVals)
  -- rl <- mkUnichainGrenade alg (liftInitSt initState) netInp actionFun actFilter params decay modelBuilderGrenade nnConfig borlSettings (Just initVals)
  rl <- mkUnichainHasktorchAsSAM (Just (1, 0.03)) [minBound..maxBound] alg (liftInitSt initState) netInp actionFun actFilter params decay modelBuilderHasktorch nnConfig borlSettings (Just initVals)
  -- rl <- mkUnichainHasktorchAsSAM Nothing [minBound..maxBound] alg (liftInitSt initState) netInp actionFun actFilter params decay modelBuilderHasktorch nnConfig borlSettings (Just initVals)

  -- Use a table to approximate the function (tabular version)
  -- rl <- mkUnichainTabular alg (liftInitSt initState) tblInp actionFun actFilter params decay borlSettings (Just initVals)
  -- rl <- mkUnichainRegressionAs [minBound..maxBound] alg (liftInitSt initState) netInp actionFun actFilter params decay regConf nnConfig borlSettings (Just initVals)

  let inverseSt | isAnn rl = Just mInverseSt
                | otherwise = Nothing

  askUser inverseSt True usage cmds [cmdDrawGrid] rl -- maybe increase learning by setting estimate of rho
  where
    cmds = zipWith (\u a -> (fst u, maybe [0] return (elemIndex a actions))) usage [Up, Left, Down, Right]
    usage = [("i", "Move up"), ("j", "Move left"), ("k", "Move down"), ("l", "Move right")]
    cmdDrawGrid = ("d", "Draw grid", \rl -> drawGrid rl >> return rl)

regConf :: St -> RegressionConfig
regConf _ = def
  { regConfigBatchSize               = 64
  , regConfigGradModelErrorThreshold = 1e-5
  , regConfigGradDecentMaxSteps = 10
  , regConfigLearningAlgorithm       = -- StochasticGradientDescentAdam def Nothing
                                       GradientDescent
                                       -- AlternatingAlgorithms $ VB.fromList [ (10, 100, GradientDescent) ,(1000, 3, StochasticGradientDescentAdam def Nothing)]
  , regConfigMinCorrelation          = 0.01
  , regConfigStartup                 = def
  , regConfigClipOutput              = Nothing
  , regConfigModel                   =
    -- RegressionModels True $ VB.fromList [RegModelLayer True RegTermNonLinear $ VB.fromList [RegModelAll RegTermLinear, RegModelAll RegTermQuadratic]]
    RegressionModels True $ VB.fromList [RegModelAll RegTermLinear] -- ^ Models to use for Regression: Default: @RegressionModels True $ VB.fromList [RegModelAll RegTermLinear]@
  }


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


netInp :: St -> V.Vector Double
netInp st = V.fromList [fromIntegral . fst . getCurrentIdx $ st, fromIntegral . snd . getCurrentIdx $ st]
  -- V.fromList [scaleMinMax (0, fromIntegral maxX) $ fromIntegral $ fst (getCurrentIdx st), scaleMinMax (0, fromIntegral maxY) $ fromIntegral $ snd (getCurrentIdx st)]

tblInp :: St -> V.Vector Double
tblInp st = V.fromList [fromIntegral $ fst (getCurrentIdx st), fromIntegral $ snd (getCurrentIdx st)]

initState :: St
initState = fromIdx (maxX,maxY)

goal :: St
goal = fromIdx (goalX, goalY)

-- State
data St = St Int Int deriving (Eq, NFData, Generic, Serialize)

instance Ord St where
  x <= y = fst (getCurrentIdx x) < fst (getCurrentIdx y) || (fst (getCurrentIdx x) == fst (getCurrentIdx y) && snd (getCurrentIdx x) < snd (getCurrentIdx y))

instance Show St where
  show xs = show (getCurrentIdx xs)

instance Enum St where
  fromEnum st = let (x,y) = getCurrentIdx st
                in x * (maxX + 1) + y
  toEnum x = fromIdx (x `div` (maxX+1), x `mod` (maxX+1))

instance Bounded St where
  minBound = fromIdx (0,0)
  maxBound = fromIdx (maxX, maxY)


-- Actions
data Act = Random | Up | Down | Left | Right
  deriving (Eq, Ord, Enum, Bounded, Generic, NFData, Serialize)

instance Show Act where
  show Random = "random"
  show Up     = "up    "
  show Down   = "down  "
  show Left   = "left  "
  show Right  = "right "

actions :: [Act]
actions = [Random, Up, Down, Left, Right]


actionFun :: ARAL St Act -> AgentType -> St -> [Act] -> IO (Reward St, St, EpisodeEnd)
actionFun _ tp s [Random] = goalState moveRand tp s
actionFun _ tp s [Up]     = goalState moveUp tp s
actionFun _ tp s [Down]   = goalState moveDown tp s
actionFun _ tp s [Left]   = goalState moveLeft tp s
actionFun _ tp s [Right]  = goalState moveRight tp s
-- actionFun _ tp s [Random, Random] = goalState moveRand tp s
actionFun _ tp s [x, y] = do
  (r1, s1, e1) <- actionFun tp s [x]
  (r2, s2, e2) <- actionFun tp s1 [y]
  return ((r1 + r2) / 2, s2, e1 || e2)
actionFun _ _ _ xs        = error $ "Multiple/Unexpected actions received in actionFun: " ++ show xs

actFilter :: St -> [V.Vector Bool]
actFilter st
  | st == fromIdx (goalX, goalY) = replicate (borlSettings ^. independentAgents) (True `V.cons` V.replicate (length actions - 1) False)
actFilter _
  | borlSettings ^. independentAgents == 1 = [False `V.cons` V.replicate (length actions - 1) True]
actFilter _
  | borlSettings ^. independentAgents == 2 = [V.fromList [False, True, True, False, False], V.fromList [False, False, False, True, True]]
actFilter _ = error "Unexpected setup in actFilter in GridworldMini.hs"


moveRand :: AgentType -> St -> IO (Reward St, St, EpisodeEnd)
moveRand = moveUp

ioRefMaxR :: IORef Double
ioRefMaxR = unsafePerformIO $ newIORef 8
{-# NOINLINE ioRefMaxR  #-}


goalState :: (AgentType -> St -> IO (Reward St, St, EpisodeEnd)) -> AgentType -> St -> IO (Reward St, St, EpisodeEnd)
goalState f tp st = do
  x <- randomRIO (0, maxX :: Int)
  y <- randomRIO (0, maxY :: Int)
  maxRand <- readIORef ioRefMaxR
  r <- if maxRand <= 0 then return 0 else randomRIO (0, maxRand)
  let stepRew (Reward re, s, e) = (Reward $ re + r, s, e)
  case getCurrentIdx st of
    (x', y')
      | x' == goalX && y' == goalY -> -- return (Reward 10, fromIdx (x, y), True)
                                   return (Reward 10, fromIdx (x, y), False)
    _ -> stepRew <$> f tp st


moveUp :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveUp _ st
    | m == 0 = return (Reward (-1), st, False)
    | otherwise = return (Reward 0, fromIdx (m-1,n), False)
  where (m,n) = getCurrentIdx st

moveDown :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveDown _ st
    | m == maxX = return (Reward (-1), st, False)
    | otherwise = return (Reward 0, fromIdx (m+1,n), False)
  where (m,n) = getCurrentIdx st

moveLeft :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveLeft _ st
    | n == 0 = return (Reward (-1), st, False)
    | otherwise = return (Reward 0, fromIdx (m,n-1), False)
  where (m,n) = getCurrentIdx st

moveRight :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveRight _ st
    | n == maxY = return (Reward (-1), st, False)
    | otherwise = return (Reward 0, fromIdx (m,n+1), False)
  where (m,n) = getCurrentIdx st


-- Conversion from/to index for state

fromIdx :: (Int, Int) -> St
fromIdx (m,n) = St m n
  --  $ zipWith (\nr xs -> zipWith (\nr' ys -> if m == nr && n == nr' then 1 else 0) [0..] xs) [0..] base
  -- where base = replicate 5 [0,0,0,0,0]


allStateInputs :: M.Map NetInputWoAction St
allStateInputs = M.fromList $ zip (map netInp [minBound..maxBound]) [minBound..maxBound]

mInverseSt :: NetInputWoAction -> Maybe (Either String St)
mInverseSt xs = return <$> M.lookup xs allStateInputs

getCurrentIdx :: St -> (Int,Int)
getCurrentIdx (St x y ) = (x, y)


drawGrid :: ARAL St Act -> IO ()
drawGrid aral = do
  putStr "\n    "
  mapM_ (putStr . printf "%2d ") ([0 .. maxY] :: [Int])
  putStr "\n"
  mapM_
    (\x -> do
       putStr (printf "%2d: " x)
       mapM_ (drawField aral . St x) ([0 .. maxY] :: [Int])
       putStr "\n")
    ([0 .. maxX] :: [Int])


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


chooseRandomReward :: IO ()
chooseRandomReward = do
  putStr "Enter x for random reward in U(0, x) per step (Default: 0): "
  hFlush stdout
  nr <- getIOWithDefault 0
  writeIORef ioRefMaxR nr
