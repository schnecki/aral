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

-- | Gridworld example that simply returns 1 for every action, except when in the goal state where it returns 0. This is
-- simply used to test the minimisation objective.
module Main where

import           ML.ARAL                  as B

import           Experimenter

import           Helper
import           SolveLp

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
import           Data.Maybe               (fromMaybe)
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
    { _replayMemoryMaxSize = 1000
    , _replayMemoryStrategy = ReplayMemoryPerAction -- ReplayMemorySingle
    , _trainBatchSize = 2
    , _trainingIterations = 1
    , _grenadeLearningParams = OptAdam 0.001 0.9 0.999 1e-8 1e-3
    , _grenadeSmoothTargetUpdate = 0.01
    , _grenadeSmoothTargetUpdatePeriod = 1
    , _learningParamsDecay = ExponentialDecay Nothing 0.05 100000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsRewardAlg alg False 6
    , _scaleOutputAlgorithm = ScaleMinMax -- ScaleLog 1000 -- ScaleMinMax
    , _cropTrainMaxValScaled = Just 0.98
    , _grenadeDropoutFlipActivePeriod = 10000
    , _grenadeDropoutOnlyInactiveAfter = 10^5
    , _clipGradients = ClipByGlobalNorm 0.01
    }

borlSettings :: Settings
borlSettings = def {_workersMinExploration = [] -- replicate 7 0.01
                   , _nStep = 1}


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
    , _epsilon             = 0.05

    , _exploration         = 1.0
    , _learnRandomAbove    = 0.99

    }


-- | Decay function of parameters.
decay :: ParameterDecaySetting
decay =
    Parameters
      { _alpha            = ExponentialDecay (Just 5e-5) 0.5 10000  -- 5e-4
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


-- -- | Decay function of parameters.
-- decay :: ParameterDecaySetting
-- decay =
--     Parameters
--       { _alpha            = ExponentialDecay (Just 1e-5) 0.5 50000  -- 5e-4
--       , _alphaRhoMin      = NoDecay
--       , _beta             = ExponentialDecay (Just 1e-4) 0.5 50000
--       , _delta            = ExponentialDecay (Just 5e-4) 0.5 50000
--       , _gamma            = ExponentialDecay (Just 1e-3) 0.5 50000 -- 1e-3
--       , _zeta             = ExponentialDecay (Just 0) 0.5 50000
--       , _xi               = NoDecay
--       -- Exploration
--       , _epsilon          = [NoDecay]
--       , _exploration      = ExponentialDecay (Just 0.01) 0.50 30000
--       , _learnRandomAbove = NoDecay
--       }


initVals :: InitValues
initVals = InitValues 0 10 0 0 0 0

main :: IO ()
main = do
  putStr "Experiment or user mode [User mode]? Enter l for lp mode, and u for user mode: " >> hFlush stdout
  l <- getLine
  case l of
    "l" -> lpMode
    _   -> usermode

instance BorlLp St Act where
  lpActions _ = actions
  lpActionFilter _ = head . actFilter
  lpActionFunction = actionFun

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

alg :: Algorithm St
alg =

  -- AlgBORLVOnly ByStateValues Nothing
        -- AlgDQN 0.99  EpsilonSensitive
        -- AlgDQN 0.50  EpsilonSensitive            -- does work
        -- algDQNAvgRewardFree
        AlgDQNAvgRewAdjusted 0.8 1.0 ByStateValues
  -- AlgBORL 0.5 0.8 ByStateValues mRefState

usermode :: IO ()
usermode = do

  -- Approximate all fucntions using a single neural network
  rl <- mkUnichainGrenadeCombinedNet alg (liftInitSt initState) netInp actionFun actFilter params decay modelBuilderGrenade nnConfig borlSettings (Just initVals)
  -- rl <- mkUnichainGrenade alg (liftInitSt initState) netInp actionFun actFilter params decay (modelBuilderGrenade actions initState) nnConfig borlSettings (Just initVals)

  -- Use a table to approximate the function (tabular version)
  -- rl <- mkUnichainTabular alg (liftInitSt initState) tblInp actionFun actFilter params decay borlSettings (Just initVals)

  askUser mInverseSt True usage cmds [] (flipObjective rl)
  where
    cmds = map (\(s, a) -> (fst s, maybe [0] return (elemIndex a actions))) (zip usage [Up, Left, Down, Right])
    usage = [("i", "Move up"), ("j", "Move left"), ("k", "Move down"), ("l", "Move right")]


-- | The definition for a feed forward network using the dynamic module. Note the nested networks. This network clearly is over-engeneered for this example!
modelBuilderGrenade :: Integer -> (Integer, Integer) -> IO SpecConcreteNetwork
modelBuilderGrenade lenIn (lenActs, cols)  =
  buildModelWith (NetworkInitSettings UniformInit HMatrix Nothing) def $
  inputLayer1D lenIn >>
  fullyConnected 20 >> leakyRelu >> dropout 0.90 >>
  fullyConnected 10 >> leakyRelu >>
  fullyConnected 10 >> leakyRelu >>
  fullyConnected lenOut >> reshape (lenActs, cols, 1) >> tanhLayer
  where
    lenOut = lenActs * cols


netInp :: St -> V.Vector Double
netInp st = V.fromList [scaleMinMax (0, fromIntegral maxX) $ fromIntegral $ fst (getCurrentIdx st), scaleMinMax (0, fromIntegral maxY) $ fromIntegral $ snd (getCurrentIdx st)]

tblInp :: St -> V.Vector Double
tblInp st = V.fromList [fromIntegral $ fst (getCurrentIdx st), fromIntegral $ snd (getCurrentIdx st)]

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
goalState f tp st = do
  x <- randomRIO (0, maxX :: Int)
  y <- randomRIO (0, maxY :: Int)
  r <- randomRIO (0, 8 :: Double)
  let stepRew (Reward re, s, e) = (Reward $ re + r, s, e)
  case getCurrentIdx st of
    (x', y')
      | x' == goalX && y' == goalY -> return (Reward 0, fromIdx (x, y), True)
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
