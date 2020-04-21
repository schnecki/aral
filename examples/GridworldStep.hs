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

-- | Gridworld example that simply returns 1 for every action, except when in the goal state where it returns 0. This is
-- simply used to test the minimisation objective.
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


maxX, maxY, goalX, goalY :: Int
maxX = 4                        -- [0..maxX]
maxY = 4                        -- [0..maxY]
goalX = 0
goalY = 0

instance RewardFuture St where
  type StoreType St = ()

nnConfig :: NNConfig
nnConfig =
  NNConfig
    { _replayMemoryMaxSize = 10000
    , _replayMemoryStrategy = ReplayMemorySingle
    , _trainBatchSize = 8
    , _grenadeLearningParams = OptAdam 0.001 0.9 0.999 1e-8 -- OptSGD 0.01 0.0 0.0001
    , _learningParamsDecay = ExponentialDecay Nothing 0.05 100000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsReward False 6
    , _stabilizationAdditionalRho = 0.5
    , _stabilizationAdditionalRhoDecay = ExponentialDecay Nothing 0.05 100000
    , _updateTargetInterval = 1
    , _updateTargetIntervalDecay = NoDecay


    , _workersMinExploration = []
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
      , _beta             = ExponentialDecay (Just 1e-4) 0.5 50000
      , _delta            = ExponentialDecay (Just 5e-4) 0.5 50000
      , _gamma            = ExponentialDecay (Just 1e-3) 0.5 50000 -- 1e-3
      , _zeta             = ExponentialDecay (Just 0) 0.5 50000
      , _xi               = NoDecay
      -- Exploration
      , _epsilon          = [NoDecay]
      , _exploration      = ExponentialDecay (Just 0.01) 0.50 30000
      , _learnRandomAbove = NoDecay
      }


initVals :: InitValues
initVals = InitValues 1 10 0 0 0 0

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
  rl <- mkUnichainGrenadeCombinedNet alg (liftInitSt initState) netInp actions actFilter params decay (modelBuilderGrenade actions initState) nnConfig (Just initVals)

  -- Use an own neural network for every function to approximate
  -- rl <- (randomNetworkInitWith UniformInit :: IO NN) >>= \nn -> mkUnichainGrenade alg (liftInitSt initState) netInp actions actFilter params decay nn nnConfig (Just initVals)
  -- rl <- mkUnichainTensorflow alg (liftInitSt initState) netInp actions actFilter params decay modelBuilder nnConfig  (Just initVals)
  -- rl <- mkUnichainTensorflowCombinedNet alg (liftInitSt initState) netInp actions actFilter params decay modelBuilder nnConfig (Just initVals)

  -- Use a table to approximate the function (tabular version)
  -- rl <- mkUnichainTabular alg (liftInitSt initState) tblInp actions actFilter params decay (Just initVals)

  askUser mInverseSt True usage cmds [] (flipObjective rl)
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

modelBuilder :: TF.ModelBuilderFunction
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
actFilter _  = False `V.cons` V.replicate (length actions - 1) True


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
      | x' == goalX && y' == goalY -> return (Reward 0, fromIdx (x, y), False)
    _ -> stepRew <$> f tp st

moveUp :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveUp _ st
    | m == 0 = return (Reward 1, st, False)
    | otherwise = return (Reward 1, fromIdx (m-1,n), False)
  where (m,n) = getCurrentIdx st

moveDown :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveDown _ st
    | m == maxX = return (Reward 1, st, False)
    | otherwise = return (Reward 1, fromIdx (m+1,n), False)
  where (m,n) = getCurrentIdx st

moveLeft :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveLeft _ st
    | n == 0 = return (Reward 1, st, False)
    | otherwise = return (Reward 1, fromIdx (m,n-1), False)
  where (m,n) = getCurrentIdx st

moveRight :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveRight _ st
    | n == maxY = return (Reward 1, st, False)
    | otherwise = return (Reward 1, fromIdx (m,n+1), False)
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


