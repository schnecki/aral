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
import           Control.Lens
import           Control.Lens             (set, (^.))
import           Control.Monad            (foldM, liftM, unless, when)
import           Control.Monad.IO.Class   (liftIO)
import           Data.Function            (on)
import           Data.List                (genericLength, groupBy, sortBy)
import qualified Data.Map.Strict          as M
import           Data.Serialize
import           Data.Singletons.TypeLits hiding (natVal)
import           GHC.Generics
import           GHC.Int                  (Int32, Int64)
import           GHC.TypeLits
import           Grenade
import           System.IO
import           System.Random


import qualified TensorFlow.Build         as TF (addNewOp, evalBuildT, explicitName, opDef,
                                                 opDefWithName, opType, runBuildT,
                                                 summaries)
import qualified TensorFlow.Core          as TF hiding (value)
-- import qualified TensorFlow.GenOps.Core                         as TF (square)
import qualified TensorFlow.GenOps.Core   as TF (abs, add, approximateEqual,
                                                 approximateEqual, assign, cast,
                                                 getSessionHandle, getSessionTensor,
                                                 identity', lessEqual, matMul, mul,
                                                 readerSerializeState, relu, relu', shape,
                                                 square, sub, tanh, tanh', truncatedNormal)
import qualified TensorFlow.Minimize      as TF
-- import qualified TensorFlow.Ops                                 as TF (abs, add, assign,
--                                                                        cast, identity',
--                                                                        matMul, mul, relu,
--                                                                        sub,
--                                                                        truncatedNormal)
import qualified TensorFlow.Ops           as TF (initializedVariable, initializedVariable',
                                                 placeholder, placeholder', reduceMean,
                                                 reduceSum, restore, save, scalar, vector,
                                                 zeroInitializedVariable,
                                                 zeroInitializedVariable')
import qualified TensorFlow.Session       as TF
import qualified TensorFlow.Tensor        as TF (Ref (..), collectAllSummaries,
                                                 tensorNodeName, tensorRefFromName,
                                                 tensorValueFromName)


import           Debug.Trace

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
  | s == fromIdx (0, 2) && a == actRand = map ((, 1 / fromIntegral (length stateActions)) . first fromIdx) $ concatMap filterDistance $ groupBy ((==) `on` fst) $ sortBy (compare `on` fst) stateActions
  | s == fromIdx (0, 2) = []
  | a == actRand = []
  | otherwise =
    mkProbability $ -- map head $
    filterDistance $ filter filterActRand [(step sa', actUp), (step sa', actLeft), (step sa', actRight), (step sa', actRand)]
  where
    sa' = ((row, col), a)
    step ((row, col), a)
      | a == actUp = (max 0 $ row - 1, col)
      | a == actDown = (min 4 $ row + 1, col)
      | a == actLeft = (row, max 0 $ col - 1)
      | a == actRight = (row, min 4 $ col + 1)
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
  deserialisable = fromSerialisable actions actFilter decay netInp netInp modelBuilder
  generateInput _ _ _ _ = return ((), ())
  runStep rl _ _ =
    liftIO $ do
      rl' <- stepM rl
      when (rl' ^. t `mod` 10000 == 0) $ liftIO $ prettyBORLHead True mInverseSt rl' >>= print
      let (eNr, eStart) = rl ^. episodeNrStart
          eLength = fromIntegral eStart / fromIntegral eNr
          results =
            [ StepResult "avgRew" (Just $ fromIntegral $ rl' ^. t) (rl' ^?! proxies . rho . proxyScalar)
            , StepResult "psiRho" (Just $ fromIntegral $ rl' ^. t) (rl' ^?! psis . _1)
            , StepResult "psiV" (Just $ fromIntegral $ rl' ^. t) (rl' ^?! psis . _2)
            , StepResult "psiW" (Just $ fromIntegral $ rl' ^. t) (rl' ^?! psis . _3)
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
           [ AlgBORL defaultGamma0 defaultGamma1 (ByMovAvg 3000) False Nothing
           , AlgBORL defaultGamma0 defaultGamma1 (ByMovAvg 3000) True Nothing
           , AlgBORLVOnly (ByMovAvg 3000) Nothing
           ])
        Nothing
        Nothing
        Nothing
    ]

-- | BORL Parameters.
params :: Parameters
params =
  Parameters
    { _alpha              = 0.01
    , _alphaANN           = 0.5
    , _beta               = 0.01
    , _betaANN            = 1
    , _delta              = 0.005
    , _deltaANN           = 1
    , _gamma              = 0.01
    , _gammaANN           = 1
    , _epsilon            = 1.0
    , _exploration        = 1.0
    , _learnRandomAbove   = 0.0
    , _zeta               = 0.15
    , _xi                 = 0.0075
    , _disableAllLearning = False
    }

-- | Decay function of parameters.
decay :: Decay
decay t = exponentialDecayParameters (Just minValues) 0.05 300000 t
  where
    minValues =
      Parameters
        { _alpha              = 0.000
        , _alphaANN           = 0.5
        , _beta               = 0.0005
        , _betaANN            = 1.0
        , _delta              = 0.0005
        , _deltaANN           = 1.0
        , _gamma              = 0.0005
        , _gammaANN           = 1.0
        , _epsilon            = 0.05
        , _exploration        = 0.001
        , _learnRandomAbove   = 0.0
        , _zeta               = 0.0
        , _xi                 = 0.0075
        , _disableAllLearning = False
        }

initVals :: InitValues
initVals = InitValues 0 0 0 0 0

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
     -- let rl = mkUnichainTabular algBORL initState netInp actions actFilter params decay Nothing
     -- (changed, res) <- runExperiments runMonadBorlIO databaseSetup expSetup () rl
     -- let runner = runMonadBorlIO
  let mkInitSt = mkUnichainTensorflowM algBORL initState netInp actions actFilter params decay modelBuilder nnConfig (Just initVals)
  (changed, res) <- runExperimentsM runMonadBorlTF databaseSetup expSetup () mkInitSt
  let runner = runMonadBorlTF
  putStrLn $ "Any change: " ++ show changed
  evalRes <- genEvals runner databaseSetup res evals
     -- print (view evalsResults evalRes)
  writeAndCompileLatex evalRes


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
        -- AlgDQN 0.99             -- does not work
        -- AlgDQN 0.50             -- does work
        -- algDQNAvgRewardFree
  -- AlgDQNAvgRewardFree 0.8 0.995 ByStateValues

  AlgBORL 0.5 0.8 ByStateValues False mRefState

usermode :: IO ()
usermode = do

  -- Approximate all fucntions using a single neural network
  rl <-
    case alg of
      AlgBORL{} -> (randomNetworkInitWith UniformInit :: IO NNCombined) >>= \nn -> mkUnichainGrenadeCombinedNet alg initState netInp actions actFilter params decay nn nnConfig (Just initVals)
      AlgDQNAvgRewardFree{} -> (randomNetworkInitWith UniformInit :: IO NNCombinedAvgFree) >>= \nn -> mkUnichainGrenadeCombinedNet alg initState netInp actions actFilter params decay nn nnConfig (Just initVals)
      AlgDQN{} ->  (randomNetworkInitWith UniformInit :: IO NN) >>= \nn -> mkUnichainGrenadeCombinedNet alg initState netInp actions actFilter params decay nn nnConfig (Just initVals)

  -- Use an own neural network for every function to approximate
  -- rl <- (randomNetworkInitWith UniformInit :: IO NN) >>= \nn -> mkUnichainGrenade alg initState netInp actions actFilter params decay nn nnConfig (Just initVals)
  -- rl <- mkUnichainTensorflow alg initState netInp actions actFilter params decay modelBuilder nnConfig  (Just initVals)
  -- rl <- mkUnichainTensorflowCombinedNet alg initState netInp actions actFilter params decay modelBuilderCombined nnConfig (Just initVals)

  -- Use a table to approximate the function (tabular version)
  -- let rl = mkUnichainTabular alg initState tblInp actions actFilter params decay (Just initVals)

  askUser mInverseSt True usage cmds rl -- maybe increase learning by setting estimate of rho
  where
    cmds =
      zipWith3
        (\n (s, a) na -> (s, (n, Action a na)))
        [0 ..]
        [("i", goalState moveUp), ("j", goalState moveDown), ("k", goalState moveLeft), ("l", goalState moveRight)]
        (tail names)
    usage = [("i", "Move up"), ("j", "Move left"), ("k", "Move down"), ("l", "Move right")]

maxX,maxY :: Int
maxX = 4                        -- [0..maxX]
maxY = 4                        -- [0..maxY]


type NN = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 5, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 5, 'D1 5]
type NNCombined = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 40, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 40, 'D1 40]
type NNCombinedAvgFree = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 10, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 10]

modelBuilder :: (TF.MonadBuild m) => m TensorflowModel
modelBuilder = modelBuilderCombined 1

modelBuilderCombined :: (TF.MonadBuild m) => Int64 -> m TensorflowModel
modelBuilderCombined colOut =
  buildModel $
  inputLayer1D inpLen >> fullyConnected [5*inpLen] TF.relu' >> fullyConnected [3*inpLen] TF.relu' >> fullyConnected [2*inpLen] TF.relu' >> fullyConnected [genericLength actions, colOut] TF.tanh' >>
  trainingByAdamWith TF.AdamConfig {TF.adamLearningRate = 0.005, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}
  where inpLen = genericLength (netInp initState)


nnConfig :: NNConfig
nnConfig = NNConfig
  { _replayMemoryMaxSize  = 10000
  , _trainBatchSize       = 8
  , _grenadeLearningParams = LearningParameters 0.01 0.0 0.0001
  , _prettyPrintElems     = map netInp ([minBound .. maxBound] :: [St])
  , _scaleParameters      =
    -- ScalingNetOutParameters (-10) 10 (-25) 25 (-15) 15 (-25) 25
    -- scalingByMaxAbsReward False 6
    scalingByMaxAbsReward False 6
  , _updateTargetInterval = 100 -- 3000
  , _trainMSEMax          = Nothing -- Just 0.03
  }


netInp :: St -> [Double]
netInp st = [scaleNegPosOne (0, fromIntegral maxX) $ fromIntegral $ fst (getCurrentIdx st), scaleNegPosOne (0, fromIntegral maxY) $ fromIntegral $ snd (getCurrentIdx st)]

tblInp :: St -> [Double]
tblInp st = [fromIntegral $ fst (getCurrentIdx st), fromIntegral $ snd (getCurrentIdx st)]

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

actFilter :: St -> [Bool]
actFilter st
  | st == fromIdx (0, 2) = True : repeat False
actFilter _  = False : repeat True


moveRand :: St -> IO (Reward St, St, EpisodeEnd)
moveRand = moveUp


goalState :: (St -> IO (Reward St, St, EpisodeEnd)) -> St -> IO (Reward St, St, EpisodeEnd)
goalState f st = do
  x <- randomRIO (0, maxX :: Int)
  y <- randomRIO (0, maxY :: Int)
  r <- randomRIO (0, 8 :: Double)
  let stepRew (Reward re, s, e) = (Reward $ re + r, s, e)
  case getCurrentIdx st of
    (0, 2) -> return (Reward 10, fromIdx (x, y), True)
    _      -> stepRew <$> f st


moveUp :: St -> IO (Reward St,St, EpisodeEnd)
moveUp st
    | m == 0 = return (Reward (-1), st, False)
    | otherwise = return (Reward 0, fromIdx (m-1,n), False)
  where (m,n) = getCurrentIdx st

moveDown :: St -> IO (Reward St,St, EpisodeEnd)
moveDown st
    | m == maxX = return (Reward (-1), st, False)
    | otherwise = return (Reward 0, fromIdx (m+1,n), False)
  where (m,n) = getCurrentIdx st

moveLeft :: St -> IO (Reward St,St, EpisodeEnd)
moveLeft st
    | n == 0 = return (Reward (-1), st, False)
    | otherwise = return (Reward 0, fromIdx (m,n-1), False)
  where (m,n) = getCurrentIdx st

moveRight :: St -> IO (Reward St,St, EpisodeEnd)
moveRight st
    | n == maxY = return (Reward (-1), st, False)
    | otherwise = return (Reward 0, fromIdx (m,n+1), False)
  where (m,n) = getCurrentIdx st


-- Conversion from/to index for state

fromIdx :: (Int, Int) -> St
fromIdx (m,n) = St $ zipWith (\nr xs -> zipWith (\nr' ys -> if m == nr && n == nr' then 1 else 0) [0..] xs) [0..] base
  where base = replicate 5 [0,0,0,0,0]


allStateInputs :: M.Map [Double] St
allStateInputs = M.fromList $ zip (map netInp [minBound..maxBound]) [minBound..maxBound]

mInverseSt :: Maybe (NetInputWoAction -> Maybe St)
mInverseSt = Just $ \xs -> M.lookup xs allStateInputs

getCurrentIdx :: St -> (Int,Int)
getCurrentIdx (St st) =
  second (fst . head . filter ((==1) . snd)) $
  head $ filter ((1 `elem`) . map snd . snd) $
  zip [0..] $ map (zip [0..]) st


