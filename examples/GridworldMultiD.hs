-- This is a xD world, where each dimension is steered by one agent. All of them can go left, right, up down. The goal is at (0,0,...)
--
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

import           ML.BORL                  as B
import           SolveLp

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
import           Data.List                (elemIndex, genericLength, groupBy, sort, sortBy)
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
import           Prelude                  hiding (Left, Right)
import           System.IO
import           System.Random

import           Debug.Trace

dim, cubeSize, goal :: Int
dim = 3               -- dimensions
cubeSize = 4          -- size of each dim
goal = 0              -- goal is at [goal, goal, ...]

goalSt :: St
goalSt = St $ replicate dim goal

expSetup :: BORL St Act -> ExperimentSetting
expSetup borl =
  ExperimentSetting
    { _experimentBaseName         = "gridworld-mini 28.1."
    , _experimentInfoParameters   = [isNN]
    , _experimentRepetitions      = 40
    , _preparationSteps           = 500000
    , _evaluationWarmUpSteps      = 0
    , _evaluationSteps            = 10000
    , _evaluationReplications     = 1
    , _maximumParallelEvaluations = 1
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


nnConfig :: NNConfig
nnConfig =
  NNConfig
    { _replayMemoryMaxSize = 10000
    , _replayMemoryStrategy = ReplayMemorySingle
    , _trainBatchSize = 8
    , _trainingIterations = 1
    , _grenadeLearningParams = OptAdam 0.005 0.9 0.999 1e-8 1e-3
    , _grenadeSmoothTargetUpdate = 0.01
    , _learningParamsDecay = ExponentialDecay Nothing 0.05 100000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsRewardAlg alg False 6
    , _scaleOutputAlgorithm = ScaleMinMax
    , _cropTrainMaxValScaled = Just 0.98
    , _grenadeDropoutFlipActivePeriod = 10000
    , _grenadeDropoutOnlyInactiveAfter = 10^5
    , _updateTargetInterval = 1
    , _updateTargetIntervalDecay = NoDecay
    }

borlSettings :: Settings
borlSettings = def
  { _workersMinExploration = [0.3, 0.2, 0.1]
  , _nStep = 2
  , _independentAgents = 2
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
main = usermode


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
  rl <- mkUnichainGrenadeCombinedNet alg (liftInitSt initState) netInp actionFun actFilter params decay (modelBuilderGrenade actions initState) nnConfig borlSettings (Just initVals)

  -- Use a table to approximate the function (tabular version)
  rl <- mkUnichainTabular alg (liftInitSt initState) tblInp actionFun actFilter params decay borlSettings (Just initVals)
  let inverseSt | isAnn rl = Just mInverseSt
                | otherwise = Nothing

  askUser inverseSt True usage cmds [] rl -- maybe increase learning by setting estimate of rho
  where
    cmds = map (\(s, a) -> (fst s, maybe [0] return (elemIndex a actions))) (zip usage [Up, Left, Down, Right])
    usage = [("i", "Move up"), ("j", "Move left"), ("k", "Move down"), ("l", "Move right")]


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
    lenActs = genericLength actions * fromIntegral (borlSettings ^. independentAgents)


netInp :: St -> V.Vector Float
netInp st =
  V.fromList $ map (scaleMinMax (0, fromIntegral cubeSize) . fromIntegral) (getCurrentIdx st)

tblInp :: St -> V.Vector Float
tblInp (St st) = V.fromList (map fromIntegral st)

initState :: St
initState = St $ replicate dim (cubeSize^2 `div` 2)

-- State
newtype St =
  St [Int] -- ^ Number in cubeSize^2 for each dimension.
  deriving (Eq, Ord, Show, NFData, Generic, Serialize)


-- instance Enum St where
--   fromEnum (St st) = Prelude.sum $ zipWith (\i x -> i * dim + x) [0 ..] st
--   toEnum x =

--     fromIdx (x `div` (maxX + 1), x `mod` (maxX + 1))

-- instance Bounded St where
--   minBound = fromIdx (0,0)
--   maxBound = fromIdx (maxX, maxY)


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
actionFun tp s acts =
  let xs = zipWith (move tp s) acts [0..]
  in undefined

-- actionFun tp s [Up]     = goalState moveUp tp s
-- actionFun tp s [Down]   = goalState moveDown tp s
-- actionFun tp s [Left]   = goalState moveLeft tp s
-- actionFun tp s [Right]  = goalState moveRight tp s
-- -- actionFun tp s [Random, Random] = goalState moveRand tp s
-- actionFun tp s [x, y] = do
--   (r1, s1, e1) <- actionFun tp s [x]
--   (r2, s2, e2) <- actionFun tp s1 [y]
--   return ((r1 + r2) / 2, s2, e1 || e2)
actionFun _ _ xs        = error $ "Multiple/Unexpected actions received in actionFun: " ++ show xs

actFilter :: St -> [V.Vector Bool]
actFilter st
  | st == goalSt = replicate (borlSettings ^. independentAgents) (True `V.cons` V.replicate (length actions - 1) False)
actFilter _
  | borlSettings ^. independentAgents == 1 = [False `V.cons` V.replicate (length actions - 1) True]
actFilter _
  | borlSettings ^. independentAgents == 2 = [V.fromList [False, True, True, False, False], V.fromList [False, False, False, True, True]]
actFilter _ = error "Unexpected setup in actFilter in GridworldMini.hs"


moveRand :: AgentType -> St -> IO (Reward St, St, EpisodeEnd)
moveRand tp st = do
  when (st /= goalSt) $ error $ "moveRand in non-goal state: " ++ show st
  xs <- sequenceA (replicate dim $ randomRIO (0, cubeSize^2 :: Int))
  return (Reward 10, St xs, True)

goalState :: (AgentType -> St -> IO (Reward St, St, EpisodeEnd)) -> AgentType -> St -> IO (Reward St, St, EpisodeEnd)
goalState f tp st = do
  r <- randomRIO (0, 8 :: Float)
  let stepRew (Reward re, s, e) = (Reward $ re + r, s, e)
  stepRew <$> f tp st


move :: AgentType -> St -> [Act] -> IO (Reward St, St, EpisodeEnd)
move tp s acts
  | all (== Random) acts = moveRand tp s
  | otherwise = return $ zipWith moveX acts [0 ..]
  where
    moveX Random = moveRand tp s
    moveX Up     = moveUp tp s
    moveX Down   = moveUp tp s
    moveX Down   = moveUp tp s
    moveX Down   = moveUp tp s
    moveX Down   = moveUp tp s

moveUp :: AgentType -> St -> Int -> (Float, Int)
moveUp _ (St st) agNr
    | m == 0 = (-1, 0)
    | otherwise = (0, m-cubeSize)
  where m = st !! agNr

moveDown :: AgentType -> St -> Int -> (Float, Int)
moveDown _ (St st) agNr
    | m == cubeSize = (-1, cubeSize)
    | otherwise = (0, m+cubeSize)
  where m = st !! agNr

moveLeft :: AgentType -> St -> Int -> (Float, Int)
moveLeft _ (St st) agNr
    | nRest == 0 = (-1, m)
    | otherwise = (0, m-1)
  where m = st !! agNr
        nRest = m `mod` cubeSize
        n = m `div` cubeSize

moveRight :: AgentType -> St -> Act -> (Float, Int)
moveRight _ (St st) agNr
  | nRest == cubeSize = (-1, m)
  | otherwise = (0, m + 1)
  where
    m = st !! agNr
    nRest = m `mod` cubeSize
    n = m `div` cubeSize


-- Conversion from/to index for state

-- -- Convert to list of [x,y] values
-- toIndices :: St -> [[Int]]
-- toIndices (St nr) =
--   map (\x -> [x `div` cubeSize, x `mod` cubeSize]) nr


-- toIndices :: [[Int]] -> St
-- toIndices [x, y] =
--   map (\x -> [x `div` cubeSize, x `mod` cubeSize]) nr


allStateInputs :: M.Map NetInputWoAction St
allStateInputs = M.fromList $ zip (map netInp [minBound..maxBound]) [minBound..maxBound]

mInverseSt :: NetInputWoAction -> Maybe (Either String St)
mInverseSt xs = return <$> M.lookup xs allStateInputs

getCurrentIdx :: St -> [Int]
getCurrentIdx (St st) =
  second (fst . head . filter ((==1) . snd)) $
  head $ filter ((1 `elem`) . map snd . snd) $
  zip [0..] $ map (zip [0..]) st
