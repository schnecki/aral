{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveAnyClass             #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs               #-}
{-# LANGUAGE OverloadedLists            #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE TupleSections              #-}
{-# LANGUAGE TypeFamilies               #-}
module Main where

import           ML.BORL                  as B
import           SolveLp

import           Experimenter

import           Helper

import           Control.Arrow            (first, second, (***))
import           Control.DeepSeq          (NFData)
import           Control.Lens
import           Control.Lens             (set, (^.))
import           Control.Monad            (foldM, liftM, unless, when)
import           Control.Monad.IO.Class   (liftIO)
import           Data.Function            (on)
import           Data.List                (genericLength, groupBy, sort, sortBy)
import qualified Data.Map.Strict          as M
import           Data.Serialize
import           Data.Singletons.TypeLits hiding (natVal)
import qualified Data.Text                as T
import           Data.Text.Encoding       as E
import qualified Data.Vector.Storable     as V
import           GHC.Generics
import           GHC.Int                  (Int32, Int64)
import           GHC.TypeLits
import           Grenade
import           System.IO
import           System.Random

import qualified HighLevelTensorflow      as TF

import           Debug.Trace

maxX, maxY, goalX, goalY :: Int
maxX = 4                        -- [0..maxX]
maxY = 4                        -- [0..maxY]
goalX = 0
goalY = 0


expSetup :: BORL St -> ExperimentSetting
expSetup borl =
  ExperimentSetting
    { _experimentBaseName         = "gridworld-mini 28.1."
    , _experimentInfoParameters   = [isNN, isTf]
    , _experimentRepetitions      = 40
    , _preparationSteps           = 500000
    , _evaluationWarmUpSteps      = 0
    , _evaluationSteps            = 10000
    , _evaluationReplications     = 1
    , _maximumParallelEvaluations = 1
    }
  where
    isNN = ExperimentInfoParameter "Is neural network" (isNeuralNetwork (borl ^. proxies . v))
    isTf = ExperimentInfoParameter "Is tensorflow network" (isTensorflow (borl ^. proxies . v))

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

instance BorlLp St where
  lpActions = actions
  lpActionFilter = actFilter

policy :: Policy St
policy s a
  | s == fromIdx (goalX, goalY) && a == actRand = mkProbability $ concatMap filterDistance $ groupBy ((==) `on` fst) $ sortBy (compare `on` fst) stateActions
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
    stateActions =
      ((goalX, goalY), actRand) :
      map (first getCurrentIdx) [(s, a) | s <- states, a <- tail actions, s /= fromIdx (goalX, goalY) || (s == fromIdx (goalX, goalY) && actionName a == actionName actRand)]
    filterActRand ((r, c), a)
      | r == goalX && c == goalY = actionName a == actionName actRand
      | otherwise = actionName a /= actionName actRand
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

fakeEpisodes :: BORL St -> BORL St -> BORL St
fakeEpisodes rl rl'
  | rl ^. s == goal && rl ^. episodeNrStart == rl' ^. episodeNrStart = episodeNrStart %~ (\(nr, t) -> (nr + 1, t + 1)) $ rl'
  | otherwise = episodeNrStart %~ (\(nr, t) -> (nr, t + 1)) $ rl'


instance ExperimentDef (BORL St)
  -- type ExpM (BORL St) = TF.SessionT IO
                                          where
  type ExpM (BORL St) = IO
  type InputValue (BORL St) = ()
  type InputState (BORL St) = ()
  type Serializable (BORL St) = BORLSerialisable St
  serialisable = toSerialisable
  deserialisable :: Serializable (BORL St) -> ExpM (BORL St) (BORL St)
  deserialisable = fromSerialisable actions actFilter tblInp modelBuilder
  generateInput _ _ _ _ = return ((), ())
  runStep phase rl _ _ = do
    rl' <- stepM rl
    when (rl' ^. t `mod` 10000 == 0) $ liftIO $ prettyBORLHead True mInverseSt rl' >>= print
    let (eNr, eSteps) = rl ^. episodeNrStart
        eLength = fromIntegral eSteps / max 1 (fromIntegral eNr)
        p = Just $ fromIntegral $ rl' ^. t
        val l = realToFrac (rl' ^?! l)
        results | phase /= EvaluationPhase =
                  [ StepResult "reward" p (val $ lastRewards._head)
                  , StepResult "avgEpisodeLength" p eLength
                  ]
                | otherwise =
                  [ StepResult "reward" p (val $ lastRewards._head)
                  , StepResult "avgRew" p (val $ proxies . rho . proxyScalar)
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
                                                                  [ AlgDQNAvgRewAdjusted 0.8 1.0 ByStateValues
                                                                  , AlgDQNAvgRewAdjusted 0.8 0.999 ByStateValues
                                                                  , AlgDQNAvgRewAdjusted 0.8 0.99 ByStateValues
                                                                  -- , AlgDQN 0.99 EpsilonSensitive
                                                                  -- , AlgDQN 0.5 EpsilonSensitive
                                                                  , AlgDQN 0.999 Exact
                                                                  , AlgDQN 0.99 Exact
                                                                  , AlgDQN 0.50 Exact
                                                                  ]) Nothing Nothing Nothing]
  beforeEvaluationHook _ _ _ _ rl = return $ set episodeNrStart (0, 0) $ set (B.parameters . exploration) 0.00 $ set (B.settings . disableAllLearning) True rl

nnConfig :: NNConfig
nnConfig =
  NNConfig
    { _replayMemoryMaxSize = 10000
    , _replayMemoryStrategy = ReplayMemorySingle
    , _nStep = 1
    , _trainBatchSize = 8
    , _grenadeLearningParams = OptAdam 0.001 0.9 0.999 1e-8 -- OptSGD 0.01 0.0 0.0001
    , _learningParamsDecay = ExponentialDecay Nothing 0.05 100000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsReward False 6
    , _stabilizationAdditionalRho = 0.5
    , _stabilizationAdditionalRhoDecay = ExponentialDecay Nothing 0.05 100000
    , _updateTargetInterval = 1
    , _updateTargetIntervalDecay = NoDecay


    , _workersMinExploration = [0.3, 0.2, 0.1]
    }

-- | BORL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha               = 0.01
    , _alphaRhoMin = 2e-5
    , _beta                = 0.01
    , _delta               = 0.005
    , _gamma               = 0.01
    , _epsilon             = 0.25

    , _exploration         = 1.0
    , _learnRandomAbove    = 1.5
    , _zeta                = 0.03
    , _xi                  = 0.005

    }

-- | Decay function of parameters.
decay :: ParameterDecaySetting
decay =
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-5) 0.5 50000  -- 5e-4
      , _alphaRhoMin      = NoDecay
      , _beta             = ExponentialDecay (Just 1e-4) 0.5 150000
      , _delta            = ExponentialDecay (Just 5e-4) 0.5 150000
      , _gamma            = ExponentialDecay (Just 1e-3) 0.5 150000 -- 1e-3
      , _zeta             = ExponentialDecay (Just 0) 0.5 150000
      , _xi               = NoDecay
      -- Exploration
      , _epsilon          = [NoDecay] -- ExponentialDecay (Just 5.0) 0.5 150000
      , _exploration      = ExponentialDecay (Just 0.01) 0.50 100000
      , _learnRandomAbove = NoDecay
      }

initVals :: InitValues
initVals = InitValues 0 0 0 0 0 0

main :: IO ()
main = do
  putStr "Experiment or user mode [User mode]? Enter e for experiment mode, l for lp mode, and u for user mode: " >> hFlush stdout
  l <- getLine
  case l of
    "l"   -> lpMode
    "e"   -> experimentMode
    "exp" -> experimentMode
    _     -> usermode


experimentMode :: IO ()
experimentMode = do
  let databaseSetup = DatabaseSetting "host=192.168.1.110 dbname=ARADRL user=experimenter password=experimenter port=5432" 10
  ---
  rl <- mkUnichainTabular algBORL (liftInitSt initState) tblInp actions actFilter params decay (Just initVals)
  (changed, res) <- runExperiments runMonadBorlIO databaseSetup expSetup () rl
  let runner = runMonadBorlIO
  ---
  -- let mkInitSt = mkUnichainTensorflowM algBORL (liftInitSt initState) netInp actions actFilter params decay modelBuilder nnConfig (Just initVals)
  -- (changed, res) <- runExperimentsM runMonadBorlTF databaseSetup expSetup () mkInitSt
  -- let runner = runMonadBorlTF
  ---
  putStrLn $ "Any change: " ++ show changed
  evalRes <- genEvalsConcurrent 6 runner databaseSetup res evals
     -- print (view evalsResults evalRes)
  writeAndCompileLatex databaseSetup evalRes
  writeCsvMeasure databaseSetup res NoSmoothing ["reward", "avgEpisodeLength"]


lpMode :: IO ()
lpMode = do
  putStrLn "I am solving the system using linear programming to provide the optimal solution...\n"
  runBorlLpInferWithRewardRepet 200000 policy mRefState >>= print
  putStrLn "NOTE: Above you can see the solution generated using linear programming. Bye!"


mRefState :: Maybe (St, ActionIndex)
mRefState = Nothing
-- mRefState = Just (fromIdx (goalX, goalY), 0)

alg :: Algorithm St
alg =

  -- AlgBORLVOnly ByStateValues Nothing
        -- AlgDQN 0.99  EpsilonSensitive
        -- AlgDQN 0.50  EpsilonSensitive            -- does work
        -- algDQNAvgRewardFree
        AlgDQNAvgRewAdjusted 0.8 0.99 ByStateValues
  -- AlgBORL 0.5 0.8 ByStateValues mRefState

usermode :: IO ()
usermode = do

  -- Approximate all fucntions using a single neural network
  rl <- mkUnichainGrenadeCombinedNet alg (liftInitSt initState) netInp actions actFilter params decay (modelBuilderGrenade actions initState) nnConfig (Just initVals)

  -- Use an own neural network for every function to approximate
  -- rl <- (randomNetworkInitWith UniformInit :: IO NN) >>= \nn -> mkUnichainGrenade alg (liftInitSt initState) netInp actions actFilter params decay nn nnConfig (Just initVals)
  rl <- mkUnichainTensorflow alg (liftInitSt initState) netInp actions actFilter params decay modelBuilder nnConfig  (Just initVals)
  -- rl <- mkUnichainTensorflowCombinedNet alg (liftInitSt initState) netInp actions actFilter params decay modelBuilder nnConfig (Just initVals)

  -- Use a table to approximate the function (tabular version)
  -- let rl = mkUnichainTabular alg (liftInitSt initState) tblInp actions actFilter params decay (Just initVals)

  askUser mInverseSt True usage cmds [] rl -- maybe increase learning by setting estimate of rho
  where
    cmds =
      zipWith3
        (\n (s, a) na -> (s, (n, Action a na)))
        [0 ..]
        [("i", goalState moveUp), ("j", goalState moveDown), ("k", goalState moveLeft), ("l", goalState moveRight)]
        (tail names)
    usage = [("i", "Move up"), ("j", "Move left"), ("k", "Move down"), ("l", "Move right")]


type NN = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 5, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 5, 'D1 5]
type NNCombined = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 40, Relu, FullyConnected 40 40, Relu, FullyConnected 40 30, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 40, 'D1 40, 'D1 40, 'D1 40, 'D1 30, 'D1 30]
type NNCombinedAvgFree = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 10, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 10]

modelBuilder :: (TF.MonadBuild m) => Int64 -> m TF.TensorflowModel
modelBuilder colOut =
  TF.buildModel $
  TF.inputLayer1D inpLen >>
  TF.fullyConnected [20] TF.relu' >>
  TF.fullyConnected [10] TF.relu' >>
  TF.fullyConnected [10] TF.relu' >>
  TF.fullyConnected [genericLength actions, colOut] TF.tanh' >>
  TF.trainingByAdamWith TF.AdamConfig {TF.adamLearningRate = 0.001, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}
  -- trainingByGradientDescent 0.01
  where inpLen = fromIntegral $ V.length $ netInp initState

-- | The definition for a feed forward network using the dynamic module. Note the nested networks. This network clearly is over-engeneered for this example!
modelBuilderGrenade :: [Action a] -> St -> Integer -> IO SpecConcreteNetwork
modelBuilderGrenade actions initState cols =
  buildModel $
  inputLayer1D lenIn >>
  fullyConnected 20 >> relu >> dropout 0.90 >>
  fullyConnected 10 >> relu >>
  fullyConnected 10 >> relu >>
  fullyConnected lenOut >> reshape (lenActs, cols, 1) >> tanhLayer
  where
    lenOut = lenActs * cols
    lenIn = fromIntegral $ V.length (netInp initState)
    lenActs = genericLength actions


netInp :: St -> V.Vector Float
netInp st =
  V.fromList [scaleNegPosOne (0, fromIntegral maxX) $ fromIntegral $ fst (getCurrentIdx st), scaleNegPosOne (0, fromIntegral maxY) $ fromIntegral $ snd (getCurrentIdx st)]

tblInp :: St -> V.Vector Float
tblInp st = V.fromList [fromIntegral $ fst (getCurrentIdx st), fromIntegral $ snd (getCurrentIdx st)]

names = ["random", "up   ", "down ", "left ", "right"]

initState :: St
initState = fromIdx (maxX,maxY)

goal :: St
goal = fromIdx (goalX, goalY)

-- State
newtype St = St [[Integer]] deriving (Eq, NFData, Generic, Serialize)

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
actions :: [Action St]
actions = zipWith Action
  (map goalState [moveRand, moveUp, moveDown, moveLeft, moveRight])
  names

actFilter :: St -> V.Vector Bool
actFilter st
  | st == fromIdx (goalX, goalY) = True `V.cons` V.replicate (length actions - 1) False
actFilter _ = False `V.cons` V.replicate (length actions - 1) True


moveRand :: AgentType -> St -> IO (Reward St, St, EpisodeEnd)
moveRand = moveUp


goalState :: (AgentType -> St -> IO (Reward St, St, EpisodeEnd)) -> AgentType -> St -> IO (Reward St, St, EpisodeEnd)
goalState f tp st = do
  x <- randomRIO (0, maxX :: Int)
  y <- randomRIO (0, maxY :: Int)
  r <- randomRIO (0, 8 :: Float)
  let stepRew (Reward re, s, e) = (Reward $ re + r, s, e)
  case getCurrentIdx st of
    (x', y')
      | x' == goalX && y' == goalY ->
                                   -- return (Reward 10, fromIdx (x, y), True)
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
fromIdx (m,n) = St $ zipWith (\nr xs -> zipWith (\nr' ys -> if m == nr && n == nr' then 1 else 0) [0..] xs) [0..] base
  where base = replicate 5 [0,0,0,0,0]


allStateInputs :: M.Map NetInputWoAction St
allStateInputs = M.fromList $ zip (map netInp [minBound..maxBound]) [minBound..maxBound]

mInverseSt :: Maybe (NetInputWoAction -> Maybe (Either String St))
mInverseSt = Just $ \xs -> return <$> M.lookup xs allStateInputs

getCurrentIdx :: St -> (Int,Int)
getCurrentIdx (St st) =
  second (fst . head . filter ((==1) . snd)) $
  head $ filter ((1 `elem`) . map snd . snd) $
  zip [0..] $ map (zip [0..]) st


