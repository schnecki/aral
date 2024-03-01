{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLists       #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}

module Main where


import           Control.DeepSeq
import           Control.Lens
import           Data.Default
import           Data.Int             (Int64)
import           Data.IORef
import           Data.List            (genericLength)
import qualified Data.Map.Strict      as M
import           Data.Maybe           (fromMaybe)
import           Data.Serialize
import           Data.Text            (Text)
import qualified Data.Vector.Storable as V
import           GHC.Exts             (fromList)
import           GHC.Generics
import           Prelude              hiding (Left, Right)
import           System.IO.Unsafe     (unsafePerformIO)
import           System.Random

import           ML.ARAL              hiding (actionFilter)
import           SolveLp

import           Helper

import           Debug.Trace

minDemand :: Int
minDemand = 0

maxDemand :: Int
maxDemand = 20  -- 10, 15

seasonDuration :: Int
seasonDuration = 365

epsilonDemand :: Int
epsilonDemand = 5  -- 1, 3, 5

short :: Double -- shortening of sin within min/max Demand
short = fromIntegral epsilonDemand

errorFun :: Int -> Int -> Double
errorFun prediction actual = fromIntegral (abs $ prediction - actual)

useSin :: Bool
useSin = False

seasonMedianDemandFun :: Double -> Int -> Double
seasonMedianDemandFun diam day
  | useSin = diam / 2 + diam * 0.5 * sin ((fromIntegral day * pi) / (fromIntegral seasonDuration / 2))
seasonMedianDemandFun diam day
  | x >= 0 = short + diam
  | x <= 0 = short
  where x = diam * 0.5 * sin ((fromIntegral day * pi) / (fromIntegral seasonDuration / 2))


nnConfig :: NNConfig
nnConfig =
  NNConfig
    { _replayMemoryMaxSize = 10000
    , _replayMemoryStrategy = ReplayMemorySingle
    , _trainBatchSize = 32
    , _trainingIterations = 1
    , _grenadeLearningParams = OptAdam 0.001 0.9 0.999 1e-8 1e-3
    , _grenadeSmoothTargetUpdate = 0.01
    , _grenadeSmoothTargetUpdatePeriod = 1
    , _learningParamsDecay = ExponentialDecay Nothing 0.5 100000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsReward False 2
    , _scaleOutputAlgorithm = ScaleMinMax
    , _cropTrainMaxValScaled = Just 0.98
    , _grenadeDropoutFlipActivePeriod = 0
    , _grenadeDropoutOnlyInactiveAfter = 0
    , _clipGradients = ClipByGlobalNorm 0.01
    , _autoNormaliseInput = True
    }


aralSettings :: Settings
aralSettings = def {_workersMinExploration = [], _nStep = 1}

netInp :: St -> V.Vector Double
netInp (St nr) = V.singleton (scaleMinMax (fromIntegral minDemand, fromIntegral maxDemand) (fromIntegral nr))

tblInp :: St -> V.Vector Double
tblInp (St nr) = V.singleton (fromIntegral nr)

numActions :: Int64
numActions = genericLength actions

numInputs :: Int64
numInputs = fromIntegral $ V.length $ netInp initState


instance RewardFuture St where
  type StoreType St = ()


-- instance AralLp St Act where
--   lpActions _ = actions
--   lpActionFilter _ = head . actionFilter
--   lpActionFunction = actionFun


-- policy :: Policy St Act
-- policy s a
--   | (s, a) == (A, Left)  = [((B, Right), 1.0)]
--   | (s, a) == (B, Right) = [((A, Left), 1.0)]
--   | (s, a) == (A, Right) = [((C, Left), 1.0)]
--   | (s, a) == (C, Left)  = [((A, Left), 1.0)]
--   | otherwise = []

mRefState :: Maybe (St, ActionIndex)
mRefState = Nothing

-- alg :: Algorithm St
-- alg = -- AlgRLearning
--         -- AlgARAL defaultGamma0 defaultGamma1 ByStateValues mRefState
--         -- algDQNAvgRewardFree
--       AlgARAL 0.8 0.999 ByStateValues
--         -- AlgARAL 0.8 0.999 (Fixed 1)
--         -- AlgARALVOnly (Fixed 1) Nothing
--         -- AlgDQN 0.99 EpsilonSensitive -- need to change epsilon accordingly to not have complete random!!!
--         -- AlgDQN 0.99 Exact

filename :: FilePath
filename = "demand_forecast"


main :: IO ()
main = do

  putStrLn $ unlines ["Choose Algorithm:",
                      "------------------------------",
                      "0: ARAL w/ gammas 0.8 0.99",
                      "1: Q-Learning w/ gamma 0.99",
                      "2: R-Learning"
                     ]
  nr <- getIOWithDefault (0 :: Int)
  let alg = case nr of
        1 -> AlgDQN 0.99 Exact
        2 -> AlgRLearning
        _ -> AlgARAL 0.8 0.99 ByStateValues


  -- runAralLpInferWithRewardRepetWMax 13 80000 policy mRefState >>= print
  -- runAralLp policy mRefState >>= print
  -- putStr "NOTE: Above you can see the solution generated using linear programming."

  rl <- mkUnichainTabular alg (liftInitSt initState) tblInp actionFun actionFilter params decay aralSettings (Just $ defInitValues { defaultRho = fromIntegral maxDemand, defaultRhoMinimum = fromIntegral maxDemand })
  let ql = flipObjective rl
  let inverseSt | isAnn ql = mInverseSt
                | otherwise = mInverseStTbl

  writeFile filename "Actual\tSin\tForecast\n"

  askUser (Just inverseSt) True usage cmds qlCmds ql   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []
        qlCmds = []


mInverseSt :: NetInputWoAction -> Maybe (Either String St)
mInverseSt xs = Nothing -- return <$> M.lookup xs allStateInputs

mInverseStTbl :: NetInputWoAction -> Maybe (Either String St)
mInverseStTbl xs = Nothing -- return <$> M.lookup xs allStateInputsTbl

allStateInputs :: IORef (M.Map NetInputWoAction St)
allStateInputs = unsafePerformIO $ newIORef $ M.fromList $ zip (map netInp [minBound..maxBound]) [minBound..maxBound]
{-# NOINLINE allStateInputs #-}

allStateInputsTbl :: IORef (M.Map NetInputWoAction St)
allStateInputsTbl = unsafePerformIO $ newIORef $ M.fromList $ zip (map tblInp [minBound..maxBound]) [minBound..maxBound]
{-# NOINLINE allStateInputsTbl #-}


initState :: St
initState = St 0

-- | ARAL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha               = 0.01
    , _alphaRhoMin = 2e-5
    , _beta                = 0.01
    , _delta               = 0.005
    , _gamma               = 0.01
    , _epsilon             = 0.05

    , _exploration         = 1.0
    , _learnRandomAbove    = 1.5
    , _zeta                = 0.03
    , _xi                  = 0.005

    }

-- | Decay function of parameters.
decay :: ParameterDecaySetting
decay =
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-5) 0.5 (steps `div` 2)  -- 5e-4
      , _alphaRhoMin      = NoDecay
      , _beta             = ExponentialDecay (Just 1e-4) 0.5 steps
      , _delta            = ExponentialDecay (Just 5e-4) 0.5 steps
      , _gamma            = ExponentialDecay (Just 1e-3) 0.5 steps -- 1e-3
      , _zeta             = ExponentialDecay (Just 0) 0.5 steps
      , _xi               = NoDecay
      -- Exploration
      , _epsilon          = [NoDecay]
      , _exploration      = ExponentialDecay (Just 0.01) 0.50 steps
      , _learnRandomAbove = NoDecay
      }
  where steps = 15000


-- State
data St =
  St
    { _seasonDay :: Int
    }
  deriving (Ord, Eq, Show, Bounded, NFData, Generic, Serialize)

instance Enum St where
  fromEnum (St nr) = nr
  toEnum = St


-- Actions
data Act = Act Int
  deriving (Eq, Ord, Generic, NFData, Serialize)

instance Enum Act where
  fromEnum (Act nr) = nr
  toEnum = Act

instance Bounded Act where
   minBound = Act minDemand
   maxBound = Act maxDemand

instance Show Act where
  show (Act nr)  = show nr

actions :: [Act]
actions = [minBound .. maxBound]

actionFun :: ARAL St Act -> AgentType -> St -> [Act] -> IO (Reward St, St, EpisodeEnd)
actionFun _ _ (St day) [Act nr]  = do
  let curveHigh = fromIntegral maxDemand - fromIntegral minDemand - 2 * short
  let seasonMedianDemand = seasonMedianDemandFun curveHigh day
  eps <- randomRIO (-epsilonDemand, epsilonDemand) :: IO Int
  let actual = max minDemand $ min maxDemand $ round $ seasonMedianDemand + fromIntegral eps
  let reward = Reward $ errorFun nr actual
  appendFile filename (show actual ++ "\t" ++ show seasonMedianDemand ++"\t" ++ show nr ++ "\n")
  return (reward, St $ (day + 1) `mod` seasonDuration, False)
actionFun _ _ _ xs       = error $ "Multiple actions received in actionFun: " ++ show xs

actionFilter :: St -> [V.Vector Bool]
actionFilter _ = [V.fromList (replicate (1 + maxDemand - minDemand) True)]
