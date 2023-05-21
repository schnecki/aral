{-# LANGUAGE DataKinds             #-}

{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLists       #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}

module Main where

import           Control.Monad
import           Control.Monad.IO.Class
import qualified Data.Map.Strict        as M
import           Experimenter
import           ML.ARAL                as B hiding (actionFilter)
import           SolveLp
import           System.IO

import           Helper

import           Control.DeepSeq        (NFData)
import           Control.Lens
import           Data.Default
import           Data.Int               (Int64)
import           Data.List              (genericLength)
import           Data.Serialize
import           Data.Text              (Text)
import qualified Data.Vector.Storable   as V
import           GHC.Exts               (fromList)
import           GHC.Generics
import           Grenade                hiding (train)
import           Prelude                hiding (Left, Right)

expSetup :: ARAL St Act -> ExperimentSetting
expSetup borl =
  ExperimentSetting
    { _experimentBaseName = "parallel"
    , _experimentInfoParameters = [isNN]
    , _experimentRepetitions = 30
    , _preparationSteps = 500000
    , _evaluationWarmUpSteps = 0
    , _evaluationSteps = 10000
    , _evaluationReplications = 1
    , _evaluationMaxStepsBetweenSaves = Just 20000
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
  -- , Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Last (Of "avgEpisodeLength")
  -- , Name "Exp Mean of Repl. Mean Steps to Goal" $ Mean OverExperimentRepetitions $ Stats $ Mean OverReplications $ Last (Of "avgEpisodeLength")
  -- , Name "Repl. Mean Steps to Goal" $ Mean OverReplications $ Last (Of "avgEpisodeLength")
  -- , Name "Exp StdDev of Repl. Mean Steps to Goal" $ StdDev OverExperimentRepetitions $ Stats $ Mean OverReplications $ Last (Of "avgEpisodeLength")
  -- , Mean OverExperimentRepetitions $ Stats $ StdDev OverReplications $ Last (Of "avgEpisodeLength")
  ]

mInverseSt :: NetInputWoAction -> Maybe (Either String St)
mInverseSt xs = return <$> M.lookup xs allStateInputs

allStateInputs :: M.Map NetInputWoAction St
allStateInputs = M.fromList $ zip (map netInp [minBound..maxBound]) [minBound..maxBound]


instance ExperimentDef (ARAL St Act)
  -- type ExpM (ARAL St Act) = TF.SessionT IO
                                          where
  type ExpM (ARAL St Act) = IO
  type InputValue (ARAL St Act) = ()
  type InputState (ARAL St Act) = ()
  type Serializable (ARAL St Act) = ARALSerialisable St Act
  serialisable = toSerialisable
  -- deserialisable :: Serializable (ARAL St Act) -> ExpM (ARAL St Act) (ARAL St Act)
  deserialisable = fromSerialisable actionFun actionFilter tblInp
  generateInput _ _ _ _ = return ((), ())
  runStep phase rl _ _ = do
    rl' <- stepM rl
    let inverseSt | isAnn rl = Just mInverseSt
                  | otherwise = Nothing
    when (rl' ^. t `mod` 10000 == 0) $ liftIO $ prettyARALHead True inverseSt rl' >>= print
    let (eNr, eSteps) = rl ^. episodeNrStart
        eLength = fromIntegral eSteps / max 1 (fromIntegral eNr)
        p = Just $ fromIntegral $ rl' ^. t
        val l = realToFrac $ head $ fromValue (rl' ^?! l)
        results | phase /= EvaluationPhase =
                  [ StepResult "reward" p (realToFrac (rl' ^?! lastRewards._head))
                  -- , StepResult "avgEpisodeLength" p eLength
                  ]
                | otherwise =
                  [ StepResult "reward" p (realToFrac $ rl' ^?! lastRewards._head)
                  , StepResult "avgRew" p (realToFrac $ V.head (rl' ^?! proxies . rho . proxyScalar))
                  , StepResult "psiRho" p (val $ psis . _1)
                  , StepResult "psiV" p (val $ psis . _2)
                  , StepResult "psiW" p (val $ psis . _3)
                  -- , StepResult "avgEpisodeLength" p eLength
                  -- , StepResult "avgEpisodeLengthNr" (Just $ fromIntegral eNr) eLength
                  ] -- ++
                  -- concatMap
                  --   (\s ->
                  --      map (\a -> StepResult (T.pack $ show (s, a)) p (M.findWithDefault 0 (tblInp s, a) (rl' ^?! proxies . r1 . proxyTable))) (filteredActionIndexes actions actFilter s))
                  --   (sort [(minBound :: St) .. maxBound])
    return (results, rl')
  parameters _ =
    [ParameterSetup "algorithm" (set algorithm) (view algorithm) (Just $ const $ return
                                                                  [ AlgARAL 0.8 1.0 ByStateValues
                                                                  , AlgARAL 0.8 0.999 ByStateValues
                                                                  , AlgARAL 0.8 0.99 ByStateValues
                                                                  -- , AlgDQN 0.99 EpsilonSensitive
                                                                  -- , AlgDQN 0.5 EpsilonSensitive
                                                                  , AlgDQN 0.999 Exact
                                                                  , AlgDQN 0.99 Exact
                                                                  , AlgDQN 0.50 Exact
                                                                  ]) Nothing Nothing Nothing]
  beforeEvaluationHook _ _ _ _ rl = return $ set episodeNrStart (0, 0) $ set (B.parameters . exploration) 0.00 $ set (B.settings . disableAllLearning) True rl


-- State
data St
  = Start
  | Top Int
  | Bottom Int
  | End
  deriving (Ord, Eq, Show, NFData, Generic, Serialize)

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

netInp :: St -> V.Vector Double
netInp st = V.singleton (scaleMinMax (minVal,maxVal) (fromIntegral $ fromEnum st))

maxVal :: Double
maxVal = fromIntegral $ fromEnum (maxBound :: St)

minVal :: Double
minVal = fromIntegral $ fromEnum (minBound :: St)

numActions :: Int64
numActions = genericLength actions


instance RewardFuture St where
  type StoreType St = ()


instance BorlLp St Act where
  lpActions _ = actions
  lpActionFilter _ = head . actionFilter
  lpActionFunction = actionFun


policy :: Policy St Act
policy s a
  -- | s == End = [((Start, Up), 1.0)]
  | s == End = [((Start, Down), 1.0)]
  | (s, a) == (Start, Up) = [((Top 1, Up), 1.0)]
  | (s, a) == (Start, Down) = [((Bottom 1, Down), 1.0)]
  | otherwise =
    case s of
      Top nr
        | nr < maxSt -> [((Top (nr + 1), Up), 1.0)]
      Top {} -> [((End, Up), 1.0)]
      Bottom nr
        | nr < maxSt -> [((Bottom (nr + 1), Down), 1.0)]
      Bottom {} -> [((End, Up), 1.0)]
      x -> error (show s)

mRefState :: Maybe (St, ActionIndex)
-- mRefState = Just (initState, 0)
mRefState = Nothing


-- alg :: Algorithm St
-- alg =
--         -- AlgARAL defaultGamma0 defaultGamma1 ByStateValues mRefState
--         -- algDQNAvgRewardFree
--         AlgARAL 0.9 0.99 ByStateValues
--         -- AlgARAL 0.84837 0.99 ByStateValues

--         -- AlgARALVOnly (Fixed 1) Nothing
--         -- AlgDQN 0.99 EpsilonSensitive -- need to change epsilon accordingly to not have complete random!!!
--         -- AlgDQN 0.99 Exact

main :: IO ()
main = do
  putStr "Experiment or user mode [User mode]? Enter e for experiment mode, l for lp mode, and u for user mode: " >> hFlush stdout
  l <- getLine
  case l of
    "l"   -> do
      putStrLn "I am solving the system using linear programming to provide the optimal solution...\n"
      lpRes <- runBorlLpInferWithRewardRepet 100000 policy mRefState
      print lpRes
      mkStateFile 0.65 False True lpRes
      mkStateFile 0.65 False False lpRes
      putStrLn "NOTE: Above you can see the solution generated using linear programming. Bye!"
    "e"   -> experimentMode
    "exp" -> experimentMode
    _     -> usermode

tblInp = fromIntegral . fromEnum

experimentMode :: IO ()
experimentMode = do
  let databaseSetup = DatabaseSetting "host=localhost dbname=experimenter user=experimenter password= port=5432" 10
  ---
  rl <- mkUnichainTabular (AlgARAL 0.8 1.0 ByStateValues) (liftInitSt initState) tblInp actionFun actionFilter params decay borlSettings Nothing
  (changed, res) <- runExperiments liftIO databaseSetup expSetup () rl
  let runner = liftIO
  ---
  putStrLn $ "Any change: " ++ show changed
  evalRes <- genEvalsConcurrent 6 runner databaseSetup res evals
     -- print (view evalsResults evalRes)
  writeAndCompileLatex databaseSetup evalRes
  writeCsvMeasure databaseSetup res NoSmoothing ["reward"] -- , "avgEpisodeLength"]


usermode :: IO ()
usermode = do


  lpRes <- runBorlLpInferWithRewardRepetWMax 3 1 policy mRefState
  print lpRes
  mkStateFile 0.65 False True lpRes
  mkStateFile 0.65 False False lpRes
  putStr "NOTE: Above you can see the solution generated using linear programming."

  nn <- randomNetworkInitWith (NetworkInitSettings HeEtAl HMatrix Nothing) :: IO NN

  alg <- chooseAlg mRefState
  -- rl <- mkUnichainGrenade alg (liftInitSt initState) netInp actionFun actionFilter params decay (\_ _ -> return $ SpecConcreteNetwork1D1D nn) nnConfig borlSettings Nothing
  rl <- mkUnichainTabular alg (liftInitSt initState) (fromIntegral . fromEnum) actionFun actionFilter params decay borlSettings Nothing
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
    , _grenadeSmoothTargetUpdatePeriod = 1
    , _learningParamsDecay = ExponentialDecay Nothing 0.05 100000
    , _prettyPrintElems = map netInp ([minBound .. maxBound] :: [St])
    , _scaleParameters = scalingByMaxAbsReward False 6
    , _scaleOutputAlgorithm = ScaleMinMax
    , _cropTrainMaxValScaled = Just 0.98
    , _grenadeDropoutFlipActivePeriod = 0
    , _grenadeDropoutOnlyInactiveAfter = 0
    , _clipGradients = ClipByGlobalNorm 0.01
    , _autoNormaliseInput = True
    }

borlSettings :: Settings
borlSettings = def {_workersMinExploration = [], _nStep = 1}


-- | ARAL Parameters.
params :: ParameterInitValues
params =
  Parameters
    { _alpha = 0.005
    , _alphaRhoMin = 2e-5
    , _beta = 0.01
    , _delta = 0.01
    , _gamma = 0.01
    , _epsilon = 0.015

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
data Act
  = Up
  | Down
  deriving (Eq, Ord, Enum, Bounded, Generic, NFData, Serialize)

instance Show Act where
  show Up   = "up  "
  show Down = "down"

actions :: [Act]
actions = [Up, Down]


actionFun :: ARAL St Act -> AgentType -> St -> [Act] -> IO (Reward St, St, EpisodeEnd)
actionFun _ tp s [Up]   = moveUp tp s
actionFun _ tp s [Down] = moveDown tp s
actionFun _ _ _ xs      = error $ "Multiple actions received in actionFun: " ++ show xs

actionFilter :: St -> [V.Vector Bool]
actionFilter Start    = [V.fromList [True, True]]
actionFilter Top{}    = [V.fromList [True, False]]
actionFilter Bottom{} = [V.fromList [False, True]]
actionFilter End      = [V.fromList [True, False]]


moveUp :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveUp _ s =
  return $
  case s of
    Start               -> (Reward 0, Top 1, False)
    Top nr | nr == 1    -> (Reward 1, Top (nr+1), False)
    Top nr | nr == 3    -> (Reward 4, Top (nr+1), False)
    Top nr | nr == 6    -> (Reward 1, if maxSt == nr then End else Top (nr+1), False)
    Top nr | nr < maxSt -> (Reward 0, Top (nr+1), False)
    Top{}               -> (Reward 0, End, False)
    End                 -> (Reward 0, Start, False)

moveDown :: AgentType -> St -> IO (Reward St,St, EpisodeEnd)
moveDown _ s =
  return $
  case s of
    Start                  -> (Reward 0, Bottom 1, False)
    Bottom nr | nr == 3    -> (Reward 6, Bottom (nr+1), False)
    Bottom nr | nr < maxSt -> (Reward 0, Bottom (nr+1), False)
    Bottom{}               -> (Reward 0, End, False)
    End                    -> (Reward 0, Start, False)
