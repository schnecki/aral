{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}

module Main where

import           ML.BORL                hiding (actionFilter)
import           SolveLp

import           Helper

import           Control.DeepSeq        (NFData)
import           Control.Lens
import           Data.Int               (Int64)
import           Data.List              (genericLength)
import           Data.Text              (Text)
import qualified Data.Vector            as V
import           GHC.Exts               (fromList)
import           GHC.Generics
import           Grenade                hiding (train)


import qualified TensorFlow.Build       as TF (addNewOp, evalBuildT, explicitName, opDef,
                                               opDefWithName, opType, runBuildT, summaries)
import qualified TensorFlow.Core        as TF hiding (value)
import qualified TensorFlow.GenOps.Core as TF (abs, add, approximateEqual,
                                               approximateEqual, assign, cast,
                                               getSessionHandle, getSessionTensor,
                                               identity', lessEqual, matMul, mul,
                                               readerSerializeState, relu', shape, square,
                                               sub, tanh, tanh', truncatedNormal)
import qualified TensorFlow.Minimize    as TF
import qualified TensorFlow.Ops         as TF (initializedVariable, initializedVariable',
                                               placeholder, placeholder', reduceMean,
                                               reduceSum, restore, save, scalar, vector,
                                               zeroInitializedVariable,
                                               zeroInitializedVariable')
import qualified TensorFlow.Tensor      as TF (Ref (..), collectAllSummaries,
                                               tensorNodeName, tensorRefFromName,
                                               tensorValueFromName)


-- State
data St = Start | LeftSide Int | RightSide Int | End
  deriving (Ord, Eq, Show,NFData,Generic)

instance Bounded St where
  minBound = Start
  maxBound = End

maxSt :: Int
maxSt = 6

instance Enum St where
  toEnum 0 = Start
  toEnum nr | nr <= maxSt = LeftSide nr
            | nr <= 2*maxSt = RightSide (nr-maxSt)
            | otherwise = End
  fromEnum Start          = 0
  fromEnum (LeftSide nr)  = nr
  fromEnum (RightSide nr) = nr + maxSt
  fromEnum End            = 2*maxSt +1

type R = Double
type P = Double


type NN = Network '[ FullyConnected 1 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 2, Tanh] '[ 'D1 1, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 2, 'D1 2]

netInp :: St -> [Double]
netInp st = [scaleNegPosOne (minVal,maxVal) (fromIntegral $ fromEnum st)]

maxVal :: Double
maxVal = fromIntegral $ fromEnum (maxBound :: St)

minVal :: Double
minVal = fromIntegral $ fromEnum (minBound :: St)

numActions :: Int64
numActions = genericLength actions

numInputs :: Int64
numInputs = genericLength (netInp initState)

modelBuilder :: (TF.MonadBuild m) => Int64 -> m TensorflowModel
modelBuilder cols =
  buildModel $
  inputLayer1D numInputs >> fullyConnected [20] TF.relu' >> fullyConnected [10] TF.relu' >> fullyConnected [numActions, cols] TF.tanh' >>
  trainingByAdamWith TF.AdamConfig {TF.adamLearningRate = 0.001, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}

instance RewardFuture St where
  type StoreType St = ()


instance BorlLp St where
  lpActions = actions
  lpActionFilter = actionFilter


policy :: Policy St
policy s a
  | s == End = [((Start, right), 1.0)]
  -- | s == End = [((Start, left), 1.0)]
  | (s, a) == (Start, left)  = [((LeftSide 1, left), 1.0)]
  | (s, a) == (Start, right) = [((RightSide 1, right), 1.0)]
  | otherwise = case s of
      LeftSide nr | nr < maxSt  -> [((LeftSide (nr+1), left), 1.0)]
      LeftSide{}                -> [((End, left), 1.0)]
      RightSide nr | nr < maxSt -> [((RightSide (nr+1), right), 1.0)]
      RightSide{}               -> [((End, left), 1.0)]
      x                         -> error (show s)

mRefState :: Maybe (St, ActionIndex)
-- mRefState = Just (initState, 0)
mRefState = Nothing


alg :: Algorithm St
alg =
        -- AlgBORL defaultGamma0 defaultGamma1 ByStateValues mRefState
        -- algDQNAvgRewardFree
        AlgDQNAvgRewAdjusted (Just 0.5) 0.65 1 ByStateValues
        -- AlgBORLVOnly (Fixed 1) Nothing
        -- AlgDQN 0.99 EpsilonSensitive -- need to change epsilon accordingly to not have complete random!!!
        -- AlgDQN 0.99 Exact

main :: IO ()
main = do


  runBorlLp policy mRefState >>= print
  putStr "NOTE: Above you can see the solution generated using linear programming."

  nn <- randomNetworkInitWith HeEtAl :: IO NN

  -- rl <- mkUnichainGrenade algorithm initState netInp actions actionFilter params decay nn nnConfig Nothing
  -- rl <- mkUnichainTensorflow algorithm initState netInp actions actionFilter params decay modelBuilder nnConfig Nothing
  let rl = mkUnichainTabular alg initState (return . fromIntegral . fromEnum) actions actionFilter params decay Nothing
  askUser Nothing True usage cmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = []
        usage = []


initState :: St
initState = Start


-- | BORL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha = 0.005
    , _alphaANN = 1
    , _beta = 0.01
    , _betaANN = 1
    , _delta = 0.01
    , _deltaANN = 1
    , _gamma = 0.01
    , _gammaANN = 1
    , _epsilon = 0.1
    , _explorationStrategy = EpsilonGreedy
    , _exploration = 1.0
    , _learnRandomAbove = 0.5
    , _zeta = 0.15
    , _xi = 0.001
    , _disableAllLearning = False
    }

-- | Decay function of parameters.
decay :: Decay
decay =
  decaySetupParameters
    Parameters
      { _alpha            = ExponentialDecay (Just 1e-3) 0.05 100000
      , _beta             = ExponentialDecay (Just 1e-3) 0.05 100000
      , _delta            = ExponentialDecay (Just 1e-3) 0.05 100000
      , _gamma            = ExponentialDecay (Just 1e-2) 0.05 100000
      , _zeta             = ExponentialDecay (Just 1e-3) 0.5 150000
      , _xi               = ExponentialDecay (Just 1e-3) 0.5 150000
        -- Exploration
      , _epsilon          = NoDecay
      , _exploration      = ExponentialDecay (Just 10e-2) 0.01 10000
      , _learnRandomAbove = NoDecay
      -- ANN
      , _alphaANN         = ExponentialDecay (Just 0.3) 0.75 150000
      , _betaANN          = ExponentialDecay (Just 0.3) 0.75 150000
      , _deltaANN         = ExponentialDecay (Just 0.3) 0.75 150000
      , _gammaANN         = ExponentialDecay (Just 0.3) 0.75 150000
      }


-- decay :: Decay
-- decay = -- exponentialDecayParameters (Just minValues) 0.05 100000
--   exponentialDecayParameters Nothing 0.05 100000
--   where
--     minValues =
--       Parameters
--         { _alpha = 0
--         , _alphaANN = 0
--         , _beta = 0
--         , _betaANN = 0
--         , _delta = 0
--         , _deltaANN = 0
--         , _gamma = 0
--         , _gammaANN = 0
--         , _epsilon = 0.1
--         , _exploration = 0.01
--         , _learnRandomAbove = 0
--         , _zeta = 0
--         , _xi = 0
--         , _disableAllLearning = False
--         }


-- Actions
actions :: [Action St]
actions = [left, right]

left,right :: Action St
left = Action moveLeft "left "
right = Action moveRight "right"

actionFilter :: St -> [Bool]
actionFilter Start       = [True, True]
actionFilter LeftSide{}  = [True, False]
actionFilter RightSide{} = [False, True]
actionFilter End         = [True, False]


moveLeft :: St -> IO (Reward St,St, EpisodeEnd)
moveLeft s =
  return $
  case s of
    Start                    -> (Reward 0, LeftSide 1, False)
    LeftSide nr | nr == 1    -> (Reward 1, LeftSide (nr+1), False)
    LeftSide nr | nr == 3    -> (Reward 4, LeftSide (nr+1), False)
    LeftSide nr | nr < maxSt -> (Reward 0, LeftSide (nr+1), False)
    LeftSide{}               ->  (Reward 1, End, False)
    End                      -> (Reward 0, Start, False)

moveRight :: St -> IO (Reward St,St, EpisodeEnd)
moveRight s =
  return $
  case s of
    Start                     -> (Reward 0, RightSide 1, False)
    -- RightSide nr | nr == 1    -> (Reward 0.02, RightSide (nr+1), False)
    RightSide nr | nr == 3    -> (Reward 6, RightSide (nr+1), False)
    RightSide nr | nr < maxSt -> (Reward 0, RightSide (nr+1), False)
    RightSide{}               ->  (Reward 0, End, False)
    End                       -> (Reward 0, Start, False)


encodeImageBatch :: TF.TensorDataType V.Vector a => [[a]] -> TF.TensorData a
encodeImageBatch xs = TF.encodeTensorData [genericLength xs, 2] (V.fromList $ mconcat xs)
-- encodeLabelBatch xs = TF.encodeTensorData [genericLength xs] (V.fromList xs)

setCheckFile :: FilePath -> TensorflowModel' -> TensorflowModel'
setCheckFile tempDir model = model { checkpointBaseFileName = Just tempDir }

prependName :: Text -> TensorflowModel' -> TensorflowModel'
prependName txt model = model { tensorflowModel = (tensorflowModel model)
        { inputLayerName = txt <> "/" <> (inputLayerName $ tensorflowModel model)
        , outputLayerName = txt <> "/" <> (outputLayerName $ tensorflowModel model)
        , labelLayerName = txt <> "/" <> (labelLayerName $ tensorflowModel model)
        }}


