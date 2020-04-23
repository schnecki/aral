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

import           ML.BORL
import           SolveLp

import           Experimenter

import           Helper

import           Control.Arrow            (first, second, (***))
import           Control.DeepSeq          (NFData)
import           Control.DeepSeq
import           Control.Lens
import           Control.Lens             (set, (^.))
import           Control.Monad            (foldM, liftM, unless, when)
import           Control.Monad.IO.Class   (liftIO)
import           Data.Default
import           Data.Function            (on)
import           Data.List                (genericLength, groupBy, sortBy)
import qualified Data.Map.Strict          as M
import           Data.Serialize
import           Data.Singletons.TypeLits hiding (natVal)
import qualified Data.Vector.Storable     as V
import           GHC.Generics
import           GHC.Int                  (Int64)
import           GHC.TypeLits
import           Grenade
import           System.IO
import           System.Random

import qualified HighLevelTensorflow      as TF


expSetup :: BORL St -> ExperimentSetting
expSetup borl =
  ExperimentSetting
    { _experimentBaseName         = "gridworld"
    , _experimentInfoParameters   = [isNN, isTf]
    , _experimentRepetitions      = 3
    , _preparationSteps           = 300000
    , _evaluationWarmUpSteps      = 0
    , _evaluationSteps            = 10000
    , _evaluationReplications     = 3
    , _maximumParallelEvaluations = 1
    }
  where
    isNN = ExperimentInfoParameter "Is neural network" (isNeuralNetwork (borl ^. proxies . v))
    isTf = ExperimentInfoParameter "Is tensorflow network" (isTensorflow (borl ^. proxies . v))

evals :: [StatsDef s]
evals =
  [ Id $ EveryXthElem 10 $ Of "avgRew"
  , Mean OverReplications $ EveryXthElem 100 (Of "avgRew")
  , StdDev OverReplications $ EveryXthElem 100 (Of "avgRew")
  , Mean OverReplications (Stats $ Mean OverPeriods (Of "avgRew"))
  , Mean OverReplications $ EveryXthElem 100 (Of "psiRho")
  , StdDev OverReplications $ EveryXthElem 100 (Of "psiRho")
  , Mean OverReplications $ EveryXthElem 100 (Of "psiV")
  , StdDev OverReplications $ EveryXthElem 100 (Of "psiV")
  , Mean OverReplications $ EveryXthElem 100 (Of "psiW")
  , StdDev OverReplications $ EveryXthElem 100 (Of "psiW")
  , Mean OverReplications $ Last (Of "avgEpisodeLength")
  , StdDev OverReplications $ Last (Of "avgEpisodeLength")
  ]


instance RewardFuture St where
  type StoreType St = ()

instance BorlLp St where
  lpActions = actions
  lpActionFilter = actFilter

policy :: Policy St
policy s a
  | s == fromIdx (0, 2) && a == actRand =
    map ((, 1 / fromIntegral (length stateActions)) . first fromIdx) $ concatMap filterDistance $ groupBy ((==) `on` fst) $ sortBy (compare `on` fst) stateActions
  | s == fromIdx (0, 2) = []
  | a == actRand = []
  | otherwise =
    mkProbability $
    filterChance $ filterDistance $ filter filterActRand [(step sa', actUp), (step sa', actLeft), (step sa', actRight), (step sa', actRand)]
  where
    sa' = ((row, col), a)
    step ((row, col), a)
      | a == actUp = (max 0 $ row - 1, col)
      | a == actDown = (min maxY $ row + 1, col)
      | a == actLeft = (row, max 0 $ col - 1)
      | a == actRight = (row, min maxX $ col + 1)
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
      ((0, 2), actRand) : map (first getCurrentIdx) [(s, a) | s <- states, a <- tail actions, s /= fromIdx (0, 2) || (s == fromIdx (0, 2) && actionName a == actionName actRand)]
    filterActRand ((r, c), a)
      | r == 0 && c == 2 = actionName a == actionName actRand
      | otherwise = actionName a /= actionName actRand
    filterColumn ((_, c), x)
      | c == 2 = actionName x == actionName actUp || actionName x == actionName actRand
      | c < 2 = actionName x == actionName actRight
          -- actionName x /= actionName actLeft
      | c > 2 = actionName x == actionName actLeft
        -- actionName x /= actionName actRight
      | otherwise = True
    filterChance [x] = [x]
    filterChance xs = filter ((== maximum stepsToBorder) . mkStepsToBorder . step) xs
      where stepsToBorder :: [Int]
            stepsToBorder = map (mkStepsToBorder . step) xs :: [Int]
            mkStepsToBorder (r, c) = min (r `mod` maxX) (c `mod` maxY)
    filterDistance xs = filter ((== minimum dist) . mkDistance . step) xs
      where
        dist :: [Int]
        dist = map (mkDistance . step) xs
    mkDistance (r, c) = r + abs (c - 2)
    mkProbability xs = map (\x -> (first fromIdx x, 1 / fromIntegral (length xs))) xs

instance ExperimentDef (BORL St) where
  type ExpM (BORL St) = TF.SessionT IO
  -- type ExpM (BORL St) = IO
  type InputValue (BORL St) = ()
  type InputState (BORL St) = ()
  type Serializable (BORL St) = BORLSerialisable St
  serialisable = toSerialisable
  deserialisable :: Serializable (BORL St) -> ExpM (BORL St) (BORL St)
  deserialisable = fromSerialisable actions actFilter netInp modelBuilder
  generateInput _ _ _ _ = return ((), ())
  runStep phase rl _ _ = do
      rl' <- stepM rl
      when (rl' ^. t `mod` 10000 == 0) $ liftIO $ prettyBORLHead True mInverseSt rl' >>= print
      let (eNr, eStart) = rl ^. episodeNrStart
          eLength = fromIntegral eStart / fromIntegral eNr
          val l = realToFrac (rl' ^?! l)
          results =
            [ StepResult "avgRew" (Just $ fromIntegral $ rl' ^. t) (val $ proxies . rho . proxyScalar)
            , StepResult "psiRho" (Just $ fromIntegral $ rl' ^. t) (val $ psis . _1)
            , StepResult "psiV" (Just $ fromIntegral $ rl' ^. t) (val $ psis . _2)
            , StepResult "psiW" (Just $ fromIntegral $ rl' ^. t) (val $ psis . _3)
            , StepResult "avgEpisodeLength" (Just $ fromIntegral $ rl' ^. t) eLength
            , StepResult "avgEpisodeLengthNr" (Just $ fromIntegral eNr) eLength
            ]
      return (results, rl')
  parameters _ =
    [ ParameterSetup
        "algorithm"
        (set algorithm)
        (view algorithm)
        (Just $ const $
         return
           [ AlgBORL defaultGamma0 defaultGamma1 (ByMovAvg 3000)  Nothing
           , AlgBORL defaultGamma0 defaultGamma1 (ByMovAvg 3000) Nothing
           , AlgBORLVOnly (ByMovAvg 3000) Nothing
           ])
        Nothing
        Nothing
        Nothing
    ]


nnConfig :: NNConfig
nnConfig =
  NNConfig
    { _replayMemoryMaxSize = 1
    , _replayMemoryStrategy = ReplayMemorySingle
    , _trainBatchSize = 1
    , _grenadeLearningParams = OptAdam 0.001 0.9 0.999 1e-8
       -- OptSGD 0.01 0.0 0.0001
    , _grenadeSmoothTargetUpdate = 0.01
    , _learningParamsDecay = NoDecay -- ExponentialDecay Nothing 0.05 100000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsReward False 6
    , _stabilizationAdditionalRho = 0.0
    , _stabilizationAdditionalRhoDecay = ExponentialDecay Nothing 0.05 100000
    , _updateTargetInterval = 10000
    , _updateTargetIntervalDecay = StepWiseIncrease (Just 500) 0.1 10000
    }

borlSettings :: Settings
borlSettings = def {_workersMinExploration = replicate 7 0.01 --  []} -- [0.4, 0.2, 0.1, 0.03]}
                   , _nStep = 4
                   , _workersUpdateInterval = 1000
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
    , _zeta                = 0.03
    , _xi                  = 0.005
    -- Exploration
    , _epsilon             = 0.25

    , _exploration         = 1.0
    , _learnRandomAbove    = 0.5

    }

-- | Decay function of parameters.
decay :: ParameterDecaySetting
decay =
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-5) 0.5 50000  -- 5e-4
      , _alphaRhoMin      = NoDecay
      , _beta             = ExponentialDecay (Just 1e-4) 0.5 150000
      , _delta            = ExponentialDecay (Just 5e-4) 0.5 150000
      , _gamma            = ExponentialDecay (Just 1e-3) 0.5 150000
      , _zeta             = ExponentialDecay (Just 0) 0.5 150000
      , _xi               = NoDecay
      -- Exploration
      , _epsilon          = [NoDecay] -- [ExponentialDecay (Just 0.050) 0.05 150000]
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
  let databaseSetup = DatabaseSetting "host=localhost dbname=experimenter2 user=experimenter password= port=5432" 10
  ---
  -- let rl = mkUnichainTabular algBORL (liftInitSt initState) netInp actions actFilter params decay Nothing
  -- (changed, res) <- runExperiments runMonadBorlIO databaseSetup expSetup () rl
  -- let runner = runMonadBorlIO
  ---
  let mkInitSt = mkUnichainTensorflowM algBORL (liftInitSt initState) netInp actions actFilter params decay modelBuilder nnConfig borlSettings (Just initVals)
  (changed, res) <- runExperimentsM runMonadBorlTF databaseSetup expSetup () mkInitSt
  let runner = runMonadBorlTF
  ---
  putStrLn $ "Any change: " ++ show changed
  evalRes <- genEvalsConcurrent 6 runner databaseSetup res evals
     -- print (view evalsResults evalRes)
  writeAndCompileLatex databaseSetup evalRes


lpMode :: IO ()
lpMode = do
  putStrLn "I am solving the system using linear programming to provide the optimal solution...\n"
  runBorlLpInferWithRewardRepet 100000 policy mRefState >>= print
  putStrLn "NOTE: Above you can see the solution generated using linear programming. Bye!"


mRefState :: Maybe (St, ActionIndex)
mRefState = Nothing
-- mRefState = Just (fromIdx (0,2), 0)

alg :: Algorithm St
alg =
       -- AlgBORLVOnly ByStateValues Nothing
        -- AlgDQN 0.99 EpsilonSensitive            -- does not work
        -- AlgDQN 0.50  EpsilonSensitive            -- does work
        -- algDQNAvgRewardFree
  AlgDQNAvgRewAdjusted 0.8 1.0 ByStateValues
  -- AlgBORL 0.5 0.8 ByStateValues mRefState


usermode :: IO ()
usermode = do

  -- Approximate all fucntions using a single neural network
  rl <- force <$> mkUnichainGrenadeCombinedNet alg (liftInitSt initState) netInp actions actFilter params decay (modelBuilderGrenade actions initState) nnConfig borlSettings (Just initVals)


  -- Use an own neural network for every function to approximate
  -- rl <- (randomNetworkInitWith UniformInit :: IO NN) >>= \nn -> mkUnichainGrenade alg (liftInitSt initState) netInp actions actFilter params decay nn nnConfig borlSettings (Just initVals)
  -- rl <- mkUnichainTensorflowCombinedNet alg (liftInitSt initState) netInp actions actFilter params decay modelBuilder nnConfig borlSettings (Just initVals)
  -- rl <- mkUnichainTensorflowCombinedNet alg (liftInitSt initState) netInp actions actFilter params decay modelBuilder nnConfig borlSettings (Just initVals)

  -- Use a table to approximate the function (tabular version)
  -- rl <- mkUnichainTabular alg (liftInitSt initState) tblInp actions actFilter params decay borlSettings (Just initVals)

  askUser mInverseSt True usage cmds [] rl -- maybe increase learning by setting estimate of rho
  where
    cmds =
      zipWith3
        (\n (s, a) na -> (s, (n, Action a na)))
        [1 ..]
        [("i", goalState moveUp), ("j", goalState moveDown), ("k", goalState moveLeft), ("l", goalState moveRight)]
        (tail names)
    usage = [("i", "Move up"), ("j", "Move left"), ("k", "Move down"), ("l", "Move right")]

maxX,maxY :: Int
maxX = 4                        -- [0..maxX]
maxY = 4                        -- [0..maxY]


modelBuilder :: (TF.MonadBuild m) => Int64 -> m TF.TensorflowModel
modelBuilder colOut =
  TF.buildModel $
  TF.inputLayer1D inpLen >>
  TF.fullyConnected [20] TF.relu' >>
  TF.fullyConnected [10] TF.relu' >>
  TF.fullyConnected [10] TF.relu' >>
  TF.fullyConnected [genericLength actions, colOut] TF.tanh' >>
  -- TF.trainingByAdamWith TF.AdamConfig {TF.adamLearningRate = 0.001, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}
  -- TF.trainingByGradientDescent 0.01
  TF.trainingByRmsProp
  where inpLen = fromIntegral $ V.length (netInp initState)

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
netInp st = V.fromList [scaleNegPosOne (0, fromIntegral maxX) $ fromIntegral $ fst (getCurrentIdx st), scaleNegPosOne (0, fromIntegral maxY) $ fromIntegral $ snd (getCurrentIdx st)]

tblInp :: St -> V.Vector Float
tblInp st = V.fromList [fromIntegral $ fst (getCurrentIdx st), fromIntegral $ snd (getCurrentIdx st)]

names = ["random", "up   ", "down ", "left ", "right"]

initState :: St
initState = fromIdx (2,2)


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
  | st == fromIdx (0, 2) = True `V.cons` V.replicate (length actions - 1) False
actFilter _  = False `V.cons` V.replicate (length actions - 1) True


moveRand :: AgentType -> St -> IO (Reward St, St, EpisodeEnd)
moveRand = moveUp


goalState :: (AgentType -> St -> IO (Reward St, St, EpisodeEnd)) -> AgentType -> St -> IO (Reward St, St, EpisodeEnd)
goalState f agentType st = do
  x <- randomRIO (0, maxX :: Int)
  y <- randomRIO (0, maxY :: Int)
  r <- randomRIO (0, 8 :: Float)
  let stepRew (Reward re, s, e) = (Reward $ re + r, s, e)
  case getCurrentIdx st of
    (0, 2) -> return (Reward 10, fromIdx (x, y), True)
    _      -> stepRew <$> f agentType st


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


