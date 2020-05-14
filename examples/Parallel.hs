{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}

module Main where

import           ML.BORL              hiding (actionFilter)
import           SolveLp

import           Helper

import           Control.DeepSeq      (NFData)
import           Control.Lens
import           Data.Default
import           Data.Int             (Int64)
import           Data.List            (genericLength)
import           Data.Text            (Text)
import qualified Data.Vector.Storable as V
import           GHC.Exts             (fromList)
import           GHC.Generics
import           Grenade              hiding (train)

import qualified HighLevelTensorflow  as TF


-- State
data St = Start | Top Int | Bottom Int | End
  deriving (Ord, Eq, Show,NFData,Generic)

instance Bounded St where
  minBound = Start
  maxBound = End

maxSt :: Int
maxSt = 6

instance Enum St where
  toEnum 0 = Start
  toEnum nr | nr <= maxSt = Top nr
            | nr <= 2*maxSt = Bottom (nr-maxSt)
            | otherwise = End
  fromEnum Start       = 0
  fromEnum (Top nr)    = nr
  fromEnum (Bottom nr) = nr + maxSt
  fromEnum End         = 2*maxSt +1


type NN = Network '[ FullyConnected 1 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 2, Tanh] '[ 'D1 1, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 2, 'D1 2]

netInp :: St -> V.Vector Float
netInp st = V.singleton (scaleNegPosOne (minVal,maxVal) (fromIntegral $ fromEnum st))

maxVal :: Float
maxVal = fromIntegral $ fromEnum (maxBound :: St)

minVal :: Float
minVal = fromIntegral $ fromEnum (minBound :: St)

numActions :: Int64
numActions = genericLength actions


modelBuilder :: (TF.MonadBuild m) => Int64 -> m TF.TensorflowModel
modelBuilder cols =
  TF.buildModel $
  TF.inputLayer1D numInputs >>
  TF.fullyConnected [20] TF.relu' >>
  TF.fullyConnected [10] TF.relu' >>
  TF.fullyConnected [numActions, cols] TF.tanh' >>
  TF.trainingByAdamWith TF.AdamConfig {TF.adamLearningRate = 0.001, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}
  where
    numInputs = fromIntegral $ V.length (netInp initState)


instance RewardFuture St where
  type StoreType St = ()


instance BorlLp St where
  lpActions = actions
  lpActionFilter = actionFilter


policy :: Policy St
policy s a
  | s == End = [((Start, up), 1.0)]
  -- | s == End = [((Start, down), 1.0)]
  | (s, a) == (Start, up) = [((Top 1, up), 1.0)]
  | (s, a) == (Start, down) = [((Bottom 1, down), 1.0)]
  | otherwise =
    case s of
      Top nr
        | nr < maxSt -> [((Top (nr + 1), up), 1.0)]
      Top {} -> [((End, up), 1.0)]
      Bottom nr
        | nr < maxSt -> [((Bottom (nr + 1), down), 1.0)]
      Bottom {} -> [((End, up), 1.0)]
      x -> error (show s)

mRefState :: Maybe (St, ActionIndex)
-- mRefState = Just (initState, 0)
mRefState = Nothing


alg :: Algorithm St
alg =
        -- AlgBORL defaultGamma0 defaultGamma1 ByStateValues mRefState
        -- algDQNAvgRewardFree
        AlgDQNAvgRewAdjusted 0.84837 1 ByStateValues
        -- AlgBORLVOnly (Fixed 1) Nothing
        -- AlgDQN 0.99 EpsilonSensitive -- need to change epsilon accordingly to not have complete random!!!
        -- AlgDQN 0.99 Exact

main :: IO ()
main = do


  lpRes <- runBorlLpInferWithRewardRepetWMax 3 1 policy mRefState
  print lpRes
  mkStateFile 0.65 False True lpRes
  mkStateFile 0.65 False False lpRes
  putStr "NOTE: Above you can see the solution generated using linear programming."

  nn <- randomNetworkInitWith HeEtAl :: IO NN

  -- rl <- mkUnichainGrenade alg (liftInitSt initState) netInp actions actionFilter params decay (\_ -> return $ SpecConcreteNetwork1D1D nn) nnConfig borlSettings Nothing
  rl <- mkUnichainTensorflowCombinedNet alg (liftInitSt initState) netInp actions actionFilter params decay modelBuilder nnConfig borlSettings Nothing
  -- let rl = mkUnichainTabular alg (liftInitSt initState) (return . fromIntegral . fromEnum) actions actionFilter params decay borlSettings Nothing
  askUser Nothing True usage cmds [] rl   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []


initState :: St
initState = Start


nnConfig :: NNConfig
nnConfig =
  NNConfig
    { _replayMemoryMaxSize = 10000
    , _replayMemoryStrategy = ReplayMemorySingle
    , _trainBatchSize = 8
    , _trainingIterations = 1
    , _grenadeLearningParams = OptAdam 0.001 0.9 0.999 1e-8 1e-3
    , _grenadeSmoothTargetUpdate = 0.01
    , _learningParamsDecay = ExponentialDecay Nothing 0.05 100000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsReward False 6
    , _cropTrainMaxValScaled = Just 0.98
    , _grenadeDropoutFlipActivePeriod = 0
    , _grenadeDropoutOnlyInactiveAfter = 0
    , _updateTargetInterval = 1
    , _updateTargetIntervalDecay = NoDecay
    }

borlSettings :: Settings
borlSettings = def {_workersMinExploration = [], _nStep = 1}


-- | BORL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha = 0.005
    , _alphaRhoMin = 2e-5
    , _beta = 0.01
    , _delta = 0.01
    , _gamma = 0.01
    , _epsilon = 0.1

    , _exploration = 1.0
    , _learnRandomAbove = 0.5
    , _zeta = 0.15
    , _xi = 0.001

    }

-- | Decay function of parameters.
decay :: ParameterDecaySetting
decay =
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-3) 0.05 100000
      , _alphaRhoMin      = NoDecay
      , _beta             = ExponentialDecay (Just 1e-3) 0.05 100000
      , _delta            = ExponentialDecay (Just 1e-3) 0.05 100000
      , _gamma            = ExponentialDecay (Just 1e-2) 0.05 100000
      , _zeta             = ExponentialDecay (Just 1e-3) 0.5 150000
      , _xi               = ExponentialDecay (Just 1e-3) 0.5 150000
        -- Exploration
      , _epsilon          = [NoDecay]
      , _exploration      = ExponentialDecay (Just 10e-2) 0.01 10000
      , _learnRandomAbove = NoDecay
      }


-- Actions
actions :: [Action St]
actions = [up, down]

down,up :: Action St
up = Action moveUp "up  "
down = Action moveDown "down"

actionFilter :: St -> V.Vector Bool
actionFilter Start    = V.fromList [True, True]
actionFilter Top{}    = V.fromList [True, False]
actionFilter Bottom{} = V.fromList [False, True]
actionFilter End      = V.fromList [True, False]


moveUp :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveUp _ s =
  return $
  case s of
    Start               -> (Reward 0, Top 1, False)
    Top nr | nr == 1    -> (Reward 0, Top (nr+1), False)
    Top nr | nr == 3    -> (Reward 4, Top (nr+1), False)
    Top nr | nr < maxSt -> (Reward 0, Top (nr+1), False)
    Top{}               -> (Reward 0, End, False)
    End                 -> (Reward 0, Start, False)

moveDown :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveDown _ s =
  return $
  case s of
    Start                  -> (Reward (-2), Bottom 1, False)
    -- Bottom nr | nr == 1    -> (Reward 0.02, Bottom (nr+1), False)
    Bottom nr | nr == 3    -> (Reward 6, Bottom (nr+1), False)
    Bottom nr | nr < maxSt -> (Reward 0, Bottom (nr+1), False)
    Bottom{}               ->  (Reward 0, End, False)
    End                    -> (Reward 0, Start, False)


