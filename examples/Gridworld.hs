-- With the goal state being (0,2) an agent following the optimal policy needs on average 3.2 steps to the goal state.
-- Thus, it accumulates a reward of 3.2*4 + 10 = 22.8 every 4.2 steps. That is 5.428571429 as average reward in the optimal case.
--
-- rho^\pi^* = 5.428571429
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
{-# LANGUAGE TupleSections              #-}
{-# LANGUAGE TypeFamilies               #-}
module Main where

import           Control.Arrow            (first, second, (***))
import           Control.DeepSeq          (NFData)
import           Control.DeepSeq
import           Control.Lens
import           Control.Lens             (set, (^.))
import           Control.Monad            (foldM, liftM, unless, when)
import           Control.Monad.IO.Class   (liftIO)
import           Data.Default
import           Data.Function            (on)
import           Data.List                (elemIndex, genericLength, groupBy, sortBy)
import qualified Data.Map.Strict          as M
import           Data.Serialize
import           Data.Singletons.TypeLits hiding (natVal)
import qualified Data.Vector.Storable     as V
import           GHC.Generics
import           GHC.TypeLits
import           Grenade
import           Prelude                  hiding (Left, Right)
import           System.IO
import           System.Random

import           Experimenter
import           ML.BORL
import           SolveLp

import           Helper

import           Debug.Trace


expSetup :: BORL St Act -> ExperimentSetting
expSetup borl =
  ExperimentSetting
    { _experimentBaseName = "gridworld"
    , _experimentInfoParameters = [isNN]
    , _experimentRepetitions = 3
    , _preparationSteps = 300000
    , _evaluationWarmUpSteps = 0
    , _evaluationSteps = 10000
    , _evaluationReplications = 3
    , _evaluationMaxStepsBetweenSaves = Nothing
    }
  where
    isNN = ExperimentInfoParameter "Is neural network" (isNeuralNetwork (borl ^. proxies . v))

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

instance ExperimentDef (BORL St Act) where
  type ExpM (BORL St Act) = IO
  -- type ExpM (BORL St Act) = IO
  type InputValue (BORL St Act) = ()
  type InputState (BORL St Act) = ()
  type Serializable (BORL St Act) = BORLSerialisable St Act
  serialisable = toSerialisable
  deserialisable :: Serializable (BORL St Act) -> ExpM (BORL St Act) (BORL St Act)
  deserialisable = fromSerialisable actionFun actFilter netInp
  generateInput _ _ _ _ = return ((), ())
  runStep phase rl _ _ = do
      rl' <- stepM rl
      when (rl' ^. t `mod` 10000 == 0) $ liftIO $ prettyBORLHead True mInverseSt rl' >>= print
      let (eNr, eStart) = rl ^. episodeNrStart
          eLength = fromIntegral eStart / fromIntegral eNr
          val l = realToFrac $ head $ fromValue (rl' ^?! l)
          results =
            [ StepResult "avgRew" (Just $ fromIntegral $ rl' ^. t) (realToFrac $ V.head (rl' ^?! proxies . rho . proxyScalar))
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
    { _replayMemoryMaxSize             = 10000 -- 1000
    , _replayMemoryStrategy            = ReplayMemorySingle -- ReplayMemoryPerAction
    , _trainBatchSize                  = 8
    , _trainingIterations              = 1
    , _grenadeLearningParams           = OptAdam 0.001 0.9 0.999 1e-8 1e-3
    , _grenadeSmoothTargetUpdate       = 0.01
    , _grenadeSmoothTargetUpdatePeriod = 100
    , _learningParamsDecay             = ExponentialDecay (Just 1e-6) 0.75 10000
    , _prettyPrintElems                = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters                 = scalingByMaxAbsRewardAlg alg False 6
    , _scaleOutputAlgorithm            = ScaleMinMax
    , _cropTrainMaxValScaled           = Nothing -- Just 0.98. Implemented using LeakyTanh Layer
    , _grenadeDropoutFlipActivePeriod  = 10000
    , _grenadeDropoutOnlyInactiveAfter = 10^5
    , _clipGradients                   = NoClipping -- ClipByGlobalNorm 0.01
    }

borlSettings :: Settings
borlSettings =
  Settings
    { _useProcessForking             = True
    , _disableAllLearning            = False
    , _explorationStrategy           = EpsilonGreedy
    , _nStep                         = 5
    , _mainAgentSelectsGreedyActions = False -- True
    , _workersMinExploration         = take 5 [0.05, 0.10 .. 1.0] -- DIFFERENT ONES?
    , _overEstimateRho               = False -- True
    , _independentAgents             = 1
    , _independentAgentsSharedRho    = True -- False
    }


-- borlSettings :: Settings
-- borlSettings =
--   def
--     { _workersMinExploration = replicate 5 0.10
--     , _nStep = 5
--     , _mainAgentSelectsGreedyActions = False -- True
--     }


-- | BORL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha               = 0.01
    , _alphaRhoMin         = 2e-5
    , _beta                = 0.01
    , _delta               = 0.005
    , _gamma               = 0.01
    , _zeta                = 0.03
    , _xi                  = 0.005
    -- Exploration
    , _epsilon             = 0.25

    , _exploration         = 1.0
    , _learnRandomAbove    = 1.0 -- 0.8

    }

-- | Decay function of parameters.
decay :: ParameterDecaySetting
decay =
    Parameters
      { _alpha            = ExponentialDecay (Just 5e-5) 0.5 10000  -- 5e-4
      , _alphaRhoMin      = NoDecay
      , _beta             = ExponentialDecay (Just 1e-4) 0.5 25000
      , _delta            = ExponentialDecay (Just 5e-4) 0.5 25000
      , _gamma            = ExponentialDecay (Just 1e-3) 0.5 25000
      , _zeta             = ExponentialDecay (Just 0) 0.5 150000
      , _xi               = NoDecay
      -- Exploration
      , _epsilon          = [NoDecay] -- [ExponentialDecay (Just 0.050) 0.05 150000]
      , _exploration      = ExponentialDecay (Just 0.01) 0.50 25000
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
  rl <- mkUnichainTabular algBORL (liftInitSt initState) netInp actionFun actFilter params decay borlSettings (Just initVals)
  (changed, res) <- runExperiments liftIO databaseSetup expSetup () rl
  let runner = liftIO
  ---
  putStrLn $ "Any change: " ++ show changed
  evalRes <- genEvalsConcurrent 6 runner databaseSetup res evals
     -- print (view evalsResults evalRes)
  writeAndCompileLatex databaseSetup evalRes


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
-- mRefState = Just (fromIdx (0,2), 0)

alg :: Algorithm St
alg =
       -- AlgBORLVOnly ByStateValues Nothing
        -- AlgDQN 0.99 Exact            -- does not work
        -- AlgDQN 0.50  EpsilonSensitive            -- does work
        -- algDQNAvgRewardFree
  AlgDQNAvgRewAdjusted 0.8 1.0 ByStateValues
  -- AlgBORL 0.5 0.8 ByStateValues mRefState


usermode :: IO ()
usermode = do

  -- Approximate all fucntions using a single neural network
  rl <- mkUnichainGrenadeCombinedNet alg (liftInitSt initState) netInp actionFun actFilter params decay (modelBuilderGrenade actions initState) nnConfig borlSettings (Just initVals)


  -- Use an own neural network for every function to approximate
  -- rl <- mkUnichainGrenade alg (liftInitSt initState) netInp actionFun actFilter params decay (modelBuilderGrenade actions initState) nnConfig borlSettings (Just initVals)

  -- Use a table to approximate the function (tabular version)
  -- rl <- mkUnichainTabular alg (liftInitSt initState) tblInp actionFun actFilter params decay borlSettings (Just initVals)

  let invSt | isAnn rl = mInverseSt
            | otherwise = Nothing
  askUser invSt True usage cmds [] rl -- maybe increase learning by setting estimate of rho
  where
    cmds = map (\(s, a) -> (fst s, maybe [0] return (elemIndex a actions))) (zip usage [Up, Left, Down, Right])
    usage = [("i", "Move up"), ("j", "Move left"), ("k", "Move down"), ("l", "Move right")]

maxX,maxY, goalX, goalY :: Int
maxX = 4                        -- [0..maxX]
maxY = 4                        -- [0..maxY]
goalX = 0
goalY = 2


-- | The definition for a feed forward network using the dynamic module. Note the nested networks. This network clearly is over-engeneered for this example!
modelBuilderGrenade :: [Action a] -> St -> Integer -> IO SpecConcreteNetwork
modelBuilderGrenade actions initState cols =
  buildModelWith (def { cpuBackend = BLAS, gpuTriggerSize = Nothing } ) def $
  inputLayer1D lenIn >>
  -- fullyConnected 20 >> relu >> -- dropout 0.90 >>
  -- fullyConnected 10 >> relu >>
  -- fullyConnected 10 >> relu >>
  -- fullyConnected lenOut >> reshape (lenActs, cols, 1) >> tanhLayer
  -- buildModelWith (def { cpuBackend = BLAS, gpuTriggerSize = Nothing } ) def $
  -- inputLayer1D lenIn >>
  -- fullyConnected 20 >> relu >>
  -- fullyConnected 10 >> relu >>
  -- fullyConnected 10 >> relu >>
  -- fullyConnected lenOut >> tanhLayer
  fullyConnected (200) >> leakyRelu >>
  fullyConnected (round $ 1.75*fromIntegral lenIn) >> leakyRelu >>
  fullyConnected ((lenIn + lenOut) `div` 2) >> leakyRelu >>
  fullyConnected lenOut >> reshape (lenActs, cols, 1) -- >> tanhLayer -- leakyTanhLayer 0.98
  where
    lenOut = lenActs * cols
    lenIn = fromIntegral $ V.length (netInp initState)
    lenActs = genericLength actions


netInp :: St -> V.Vector Double
netInp st = V.fromList [scaleMinMax (0, fromIntegral maxX) $ fromIntegral $ fst (getCurrentIdx st), scaleMinMax (0, fromIntegral maxY) $ fromIntegral $ snd (getCurrentIdx st)]

tblInp :: St -> V.Vector Double
tblInp st = V.fromList [fromIntegral $ fst (getCurrentIdx st), fromIntegral $ snd (getCurrentIdx st)]

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


actionFun :: AgentType -> St -> [Act] -> IO (Reward St, St, EpisodeEnd)
actionFun tp s [Random] = goalState moveRand tp s
actionFun tp s [Up]     = goalState moveUp tp s
actionFun tp s [Down]   = goalState moveDown tp s
actionFun tp s [Left]   = goalState moveLeft tp s
actionFun tp s [Right]  = goalState moveRight tp s
actionFun _ _ xs        = error $ "Multiple actions received in actionFun: " ++ show xs

actFilter :: St -> [V.Vector Bool]
actFilter st
  | st == fromIdx (goalX, goalY) = [True `V.cons` V.replicate (length actions - 1) False]
actFilter _  = [False `V.cons` V.replicate (length actions - 1) True]


moveRand :: AgentType -> St -> IO (Reward St, St, EpisodeEnd)
moveRand = moveUp

goalState :: (AgentType -> St -> IO (Reward St, St, EpisodeEnd)) -> AgentType -> St -> IO (Reward St, St, EpisodeEnd)
goalState f agentType st = do
  x <- randomRIO (0, maxX :: Int)
  y <- randomRIO (0, maxY :: Int)
  r <- randomRIO (0, 8 :: Double)
  let stepRew (Reward re, s, e) = (Reward $ re + r, s, e)
  case getCurrentIdx st of
    (x', y')
      | x' == goalX && y' == goalY -> return (Reward 10, fromIdx (x, y), False)
    _ -> stepRew <$> f agentType st


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


-- LP instance

instance BorlLp St Act where
  lpActions _ = actions
  lpActionFilter _ = head . actFilter
  lpActionFunction = actionFun


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
