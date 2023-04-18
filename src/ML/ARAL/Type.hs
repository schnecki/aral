{-# LANGUAGE CPP                  #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE DeriveAnyClass       #-}
{-# LANGUAGE DeriveGeneric        #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE OverloadedStrings    #-}
{-# LANGUAGE PolyKinds            #-}
{-# LANGUAGE Rank2Types           #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TemplateHaskell      #-}
{-# LANGUAGE TupleSections        #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE UndecidableInstances #-}

module ML.ARAL.Type
  ( -- objective
    Objective (..)
  , setObjective
  , flipObjective
    -- common
  , RewardFutureData (..)
  , futurePeriod
  , futureState
  , futureActionNr
  , futureReward
  , futureStateNext
  , futureEpisodeEnd
  , mapRewardFutureData
  , idxStart
  -- * ARAL
  , ARAL (..)
  , ActionFunction
  , NrFeatures
  , NrRows
  , NrCols
  , ModelBuilderFun
  , ModelBuilderFunHT
  , actionList
  , actionFilter
  , actionFunction
  , s
  , workers
  , featureExtractor
  , t
  , episodeNrStart
  , parameters
  , settings
  , decaySetting
  , decayedParameters
  , futureRewards
  , algorithm
  , expSmoothedReward
  , objective
  , lastVValues
  , lastRewards
  , psis
  , proxies
  -- * actions
  , actionIndicesFiltered
  , actionIndicesDisallowed
  , actionsFiltered
    -- initial values
  , InitValues (..)
  , defInitValues
  -- * Constructors
  -- ** Tabular
  , mkUnichainTabular
  , mkUnichainTabularAs
  , mkMultichainTabular
  -- ** Regression
  , mkUnichainRegressionAs
  -- ** Hasktorch
  , mkUnichainHasktorch
  , mkUnichainHasktorchAs
  , mkUnichainHasktorchAsSAM
  , mkUnichainHasktorchAsSAMAC
  -- ** Grenade
  , mkUnichainGrenade
  , mkUnichainGrenadeAs
  , mkUnichainGrenadeCombinedNet
  , mkMultichainGrenade
  -- ** Scaling
  , scalingByMaxAbsReward
  , scalingByMaxAbsRewardAlg
  -- * Proxy/Proxies helpers
  , overAllProxies
  , setAllProxies
  , allProxies
  ) where

import           Control.DeepSeq
import           Control.Lens
import           Control.Monad                               (join, replicateM)
import           Control.Monad.IO.Class                      (liftIO)
import           Data.Default                                (def)
import           Data.List                                   (foldl', genericLength, zipWith3)
import           Data.List.Singletons
import qualified Data.Map.Strict                             as M
import           Data.Maybe                                  (catMaybes, fromMaybe)
import qualified Data.Proxy                                  as Type
import           Data.Serialize
import           Data.Singletons                             (Sing, SingI, SomeSing (..), sing)
import qualified Data.Text                                   as T
import           Data.Typeable                               (Typeable)
import qualified Data.Vector                                 as VB
import qualified Data.Vector.Mutable                         as VM
import qualified Data.Vector.Storable                        as V
import           EasyLogger
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.KnownNat
import           GHC.TypeLits.Singletons
import           Grenade
import           Statistics.Sample.WelfordOnlineMeanVariance
import           System.IO.Unsafe                            (unsafePerformIO)
import qualified Torch
import qualified Torch.NN                                    as Torch

import           RegNet

import           ML.ARAL.Algorithm
import           ML.ARAL.Decay
import           ML.ARAL.NeuralNetwork
import           ML.ARAL.Parameters
import           ML.ARAL.Proxy.Proxies
import           ML.ARAL.Proxy.Type
import           ML.ARAL.Reward
import           ML.ARAL.RewardFuture
import           ML.ARAL.Settings
import           ML.ARAL.Types
import           ML.ARAL.Workers.Type


import           Debug.Trace


-- Type

type NrFeatures = Integer
type NrRows = Integer
type NrCols = Integer
type ModelBuilderFun = NrFeatures -> (NrRows, NrCols) -> IO SpecConcreteNetwork
type ModelBuilderFunHT = NrFeatures -> (NrRows, NrCols) -> MLPSpec
type SingleNetPerOutputAction = Bool

type ActionFunction s as = ARAL s as -> AgentType -> s -> [as] -> IO (Reward s, s, EpisodeEnd)

-------------------- Main RL Datatype --------------------


data ARAL s as = ARAL
  { _actionList        :: !(VB.Vector (Action as)) -- ^ List of possible actions each agent. All agents can do the same actions!
  , _actionFunction    :: !(ActionFunction s as)   -- ^ Action function that traverses the system from s to s' using a list of actions, one for each agent.
  , _actionFilter      :: !(ActionFilter s)        -- ^ Function to filter actions in state s.
  , _s                 :: !s                       -- ^ Current state.
  , _workers           :: !(Workers s)             -- ^ Additional workers. (Workers s = [WorkerState s])

  , _featureExtractor  :: !(FeatureExtractor s)    -- ^ Function that extracts the features of a state. For ANNs the resulting values are expected to be in the range (-1,1).
  , _t                 :: !Int                     -- ^ Current time t.
  , _episodeNrStart    :: !(Int, Int)              -- ^ Nr of Episode and start period.
  , _parameters        :: !ParameterInitValues     -- ^ Parameter setup.
  , _decaySetting      :: !ParameterDecaySetting   -- ^ Decay Setup
  , _settings          :: !Settings                -- ^ Parameter setup.
  , _futureRewards     :: !(VB.Vector (RewardFutureData s)) -- ^ List of future reward.

  -- define algorithm to use
  , _algorithm         :: !(Algorithm StateFeatures) -- ^ What algorithm to use.
  , _objective         :: !Objective                 -- ^ Objective to minimise or maximise.

  -- Values:
  , _expSmoothedReward :: !Double                 -- ^ Exponentially smoothed reward value (with rate 0.0001).
  , _lastVValues       :: !(VB.Vector Value)      -- ^ List of X last V values (head is last seen value)
  , _lastRewards       :: !(V.Vector Double)      -- ^ List of X last rewards (head is last received reward)
  , _psis              :: !(Value, Value, Value)  -- ^ Exponentially smoothed psi values.
  , _proxies           :: !Proxies                -- ^ Scalar, Tables and Neural Networks
  }
makeLenses ''ARAL

instance (NFData as, NFData s) => NFData (ARAL s as) where
  rnf (ARAL as _ af st ws ftExt time epNr par setts dec fut alg ph expSmth lastVs lastRews psis' proxies') =
    rnf as `seq` rnf af `seq` rnf st `seq` rnf ws `seq` rnf ftExt `seq` rnf time `seq`
    rnf epNr `seq` rnf par `seq` rnf dec `seq` rnf setts `seq` rnf1 fut `seq` rnf alg `seq` rnf ph `seq` rnf expSmth `seq`
    rnf lastVs `seq` rnf lastRews  `seq` rnf psis' `seq` rnf proxies'


decayedParameters :: ARAL s as -> ParameterDecayedValues
decayedParameters borl
  | borl ^. t < repMemSize = mkStaticDecayedParams decayedParams
  | otherwise = decayedParams
  where
    decayedParams = decaySettingParameters (borl ^. decaySetting) (borl ^. t - repMemSize) (borl ^. parameters)
    repMemSize = maybe 0 replayMemoriesSize (borl ^. proxies . replayMemory)

------------------------------ Indexed Action ------------------------------

-- | Get the filtered actions of the current given state and with the ActionFilter set in ARAL.
actionIndicesFiltered :: ARAL s as -> s -> FilteredActionIndices
actionIndicesFiltered borl state = VB.map (\fil -> V.ifilter (\idx _ -> fil V.! idx) actionIndices) (VB.fromList filterVals)
  where
    filterVals :: [V.Vector Bool]
    filterVals = (borl ^. actionFilter) state
    -- actionIndices = V.fromList [0 .. length (actionLengthCheck filterVals $ borl ^. actionList) - 1]
    actionIndices = V.generate (length (actionLengthCheck filterVals $ borl ^. actionList)) id

-- | Get the filtered actions of the current given state and with the ActionFilter set in ARAL.
actionIndicesDisallowed :: ARAL s as -> s -> DisallowedActionIndicies
actionIndicesDisallowed borl state = DisallowedActionIndicies $ VB.map (\fil -> V.ifilter (\idx _ -> not (fil V.! idx)) actionIndices) (VB.fromList filterVals)
  where
    filterVals :: [V.Vector Bool]
    filterVals = (borl ^. actionFilter) state
    -- actionIndices = V.fromList [0 .. length (actionLengthCheck filterVals $ borl ^. actionList) - 1]
    actionIndices = V.generate (length (actionLengthCheck filterVals $ borl ^. actionList)) id


-- | Get the filtered actions of the current given state and with the ActionFilter set in ARAL.
actionsFiltered :: ARAL s as -> s -> FilteredActions as
actionsFiltered borl state = VB.map (\fil -> VB.ifilter (\idx _ -> fil V.! idx) (actionLengthCheck filterVals $ borl ^. actionList)) (VB.fromList filterVals)
  where
    filterVals :: [V.Vector Bool]
    filterVals = (borl ^. actionFilter) state


actionLengthCheck :: [V.Vector Bool] -> VB.Vector (Action as) -> VB.Vector (Action as)
#ifdef DEBUG
actionLengthCheck fils as
  | all ((== VB.length as) . V.length) fils = as
  | otherwise = error $ "Action and ActionFilter length do not fit. Action Length: " ++ show (VB.length as) ++ " ++ show filters: " ++ show fils
#endif
actionLengthCheck _ xs = xs
{-# INLINE actionLengthCheck #-}

-- -- | Get a list of filtered actions with the actions list and the filter function for the current given state.
-- filteredActions :: [Action a] -> (s -> V.Vector Bool) -> s -> [Action a]
-- filteredActions actions actFilter state = map snd $ filter (\(idx, _) -> actFilter state V.! idx) $ zip [(0 :: Int) ..] actions

-- -- | Get a list of filtered action indices with the actions list and the filter function for the current given state.
-- filteredActionIndexes :: [Action a] -> (s -> V.Vector Bool) -> s -> [ActionIndex]
-- filteredActionIndexes actions actFilter state = map fst $ filter (\(idx,_) -> actFilter state V.! idx) $ zip [(0::Int)..] actions


------------------------------ Initial Values ------------------------------

idxStart :: Int
idxStart = 0

data InitValues = InitValues
  { defaultRho        :: !Double -- ^ Starting rho value [Default: 0]
  , defaultRhoMinimum :: !Double -- ^ Starting minimum value (when objective is Maximise, otherwise if the objective is Minimise it's the Maximum rho value) [Default: 0]
  , defaultV          :: !Double -- ^ Starting V value [Default: 0]
  , defaultW          :: !Double -- ^ Starting W value [Default: 0]
  , defaultR0         :: !Double -- ^ Starting R0 value [Default: 0]
  , defaultR1         :: !Double -- ^ starting R1 value [Default: 0]
  } deriving (Show, Eq, Ord, Serialize, Generic)


defInitValues :: InitValues
defInitValues = InitValues 0 0 0 0 0 0


-------------------- Objective --------------------

setObjective :: Objective -> ARAL s as -> ARAL s as
setObjective obj = objective .~ obj

-- | Default objective is Maximise.
flipObjective :: ARAL s as -> ARAL s as
flipObjective borl = case borl ^. objective of
  Minimise -> objective .~ Maximise $ borl
  Maximise -> objective .~ Minimise $ borl


-------------------- Constructors --------------------

-- Tabular representations

convertAlgorithm :: FeatureExtractor s -> Algorithm s -> Algorithm StateFeatures
convertAlgorithm ftExt (AlgNBORL g0 g1 avgRew (Just (s, a))) = AlgNBORL g0 g1 avgRew (Just (ftExt s, a))
convertAlgorithm ftExt (AlgARALVOnly avgRew (Just (s, a)))   = AlgARALVOnly avgRew (Just (ftExt s, a))
convertAlgorithm _ (AlgNBORL g0 g1 avgRew Nothing)           = AlgNBORL g0 g1 avgRew Nothing
convertAlgorithm _ (AlgARALVOnly avgRew Nothing)             = AlgARALVOnly avgRew Nothing
convertAlgorithm _ (AlgDQN ga cmp)                           = AlgDQN ga cmp
convertAlgorithm _ AlgRLearning                              = AlgRLearning
convertAlgorithm _ (AlgARAL ga1 ga2 avgRew)                  = AlgARAL ga1 ga2 avgRew

mkUnichainTabular ::
     forall s as. (Enum as, Bounded as, Eq as, Ord as, NFData as)
  => Algorithm s
  -> InitialStateFun s
  -> FeatureExtractor s
  -> ActionFunction s as
  -> ActionFilter s
  -> ParameterInitValues
  -> ParameterDecaySetting
  -> Settings
  -> Maybe InitValues
  -> IO (ARAL s as)
mkUnichainTabular = mkUnichainTabularAs [minBound .. maxBound]

mkUnichainTabularAs ::
     forall s as. (Eq as, Ord as, NFData as)
  => [Action as]
  -> Algorithm s
  -> InitialStateFun s
  -> FeatureExtractor s
  -> ActionFunction s as
  -> ActionFilter s
  -> ParameterInitValues
  -> ParameterDecaySetting
  -> Settings
  -> Maybe InitValues
  -> IO (ARAL s as)
mkUnichainTabularAs as alg initialStateFun ftExt asFun asFilter params decayFun settings initVals = do
  $(logPrintDebugText) "Creating tabular unichain ARAL"
  st <- initialStateFun MainAgent
  let proxies' =
        Proxies
          (Scalar (V.replicate agents defRhoMin) (length as))
          (Scalar (V.replicate agents defRho) (length as))
          (tabSA 0)
          (tabSA defV)
          (tabSA 0)
          (tabSA defW)
          (tabSA defR0)
          (tabSA defR1)
          Nothing
  workers' <- liftIO $ mkWorkers initialStateFun as Nothing settings
  return $
    ARAL
      (VB.fromList as)
      asFun
      asFilter
      st
      workers'
      ftExt
      0
      (0, 0)
      params
      decayFun
      settings
      mempty
      (convertAlgorithm ftExt alg)
      Maximise
      defRhoMin
      mempty
      mempty
      (toValue agents 0, toValue agents 0, toValue agents 0)
      proxies'
  where
    tabSA def = Table mempty def (length as)
    defRhoMin = defaultRhoMinimum (fromMaybe defInitValues initVals)
    defRho = defaultRho (fromMaybe defInitValues initVals)
    defV = V.replicate agents $ defaultV (fromMaybe defInitValues initVals)
    defW = V.replicate agents $ defaultW (fromMaybe defInitValues initVals)
    defR0 = V.replicate agents $ defaultR0 (fromMaybe defInitValues initVals)
    defR1 = V.replicate agents $ defaultR1 (fromMaybe defInitValues initVals)
    agents = settings ^. independentAgents


mkUnichainRegressionAs ::
  forall s as . (Eq as, NFData as, Ord as, Enum as, NFData s) =>
     [Action as]
  -> Algorithm s
  -> InitialStateFun s
  -> FeatureExtractor s
  -> ActionFunction s as
  -> ActionFilter s
  -> ParameterInitValues
  -> ParameterDecaySetting
  -> (s -> RegressionConfig)
  -> NNConfig
  -> Settings
  -> Maybe InitValues
  -> IO (ARAL s as)
mkUnichainRegressionAs as alg initialStateFun ftExt asFun asFilter params decayFun mkRegConfig nnConfig settings initValues = do
  $(logPrintDebugText) "Creating unichain ARAL with Hasktorch"
  initialState <- initialStateFun MainAgent
  let regConf = mkRegConfig initialState
  let nnConfig' = nnConfig { _trainBatchSize = regConfigBatchSize regConf }
  let mkRegressionProxy xs = RegressionProxy xs (length as) nnConfig'
  let inp = ftExt initialState
  tabSA <- mkRegressionProxy <$> randRegressionLayer (Just regConf) (V.length inp) (length as)
  repMem <- mkReplayMemories as settings nnConfig'
  let proxies' =
            Proxies
              (Scalar (V.replicate agents defRhoMin) (length as))
              (Scalar (V.replicate agents defRho) (length as))
              tabSA
              tabSA
              tabSA
              tabSA
              tabSA
              tabSA
              repMem
  -- workers' <- liftIO $ mkWorkers initialStateFun as Nothing settings
  workers' <- liftIO $ mkWorkers initialStateFun as (Just nnConfig') settings
  return $!
    force $
    ARAL
      (VB.fromList as)
      asFun
      asFilter
      initialState
      workers'
      ftExt
      0
      (0, 0)
      params
      decayFun
      settings
      VB.empty
      (convertAlgorithm ftExt alg)
      Maximise
      defRhoMin
      mempty
      mempty
      (toValue agents 0, toValue agents 0, toValue agents 0)
      proxies'
  where
    defRho = defaultRho (fromMaybe defInitValues initValues)
    defRhoMin = defaultRhoMinimum (fromMaybe defInitValues initValues)
    agents = settings ^. independentAgents


mkMultichainTabular ::
     forall s as. (Bounded as, Enum as, Eq as, Ord as, NFData as) =>
     Algorithm s
  -> InitialStateFun s
  -> FeatureExtractor s
  -> ActionFunction s as
  -> ActionFilter s
  -> ParameterInitValues
  -> ParameterDecaySetting
  -> Settings
  -> Maybe InitValues
  -> IO (ARAL s as)
mkMultichainTabular alg initialStateFun ftExt asFun asFilter params decayFun settings initValues = do
  initialState <- initialStateFun MainAgent
  return $
    ARAL
      (VB.fromList as)
      asFun
      asFilter
      initialState
      []
      ftExt
      0
      (0, 0)
      params
      decayFun
      settings
      VB.empty
      (convertAlgorithm ftExt alg)
      Maximise
      0
      mempty
      mempty
      (toValue agents 0, toValue agents 0, toValue agents 0)
      (Proxies (tabSA defRhoMin) (tabSA defRho) (tabSA 0) (tabSA defV) (tabSA 0) (tabSA defW) (tabSA defR0) (tabSA defR1) Nothing)
  where
    as = [minBound .. maxBound] :: [Action as]
    tabSA def = Table mempty def (length as)
    defRhoMin = V.replicate agents $ defaultRhoMinimum (fromMaybe defInitValues initValues)
    defRho = V.replicate agents $ defaultRho (fromMaybe defInitValues initValues)
    defV = V.replicate agents $ defaultV (fromMaybe defInitValues initValues)
    defW = V.replicate agents $ defaultW (fromMaybe defInitValues initValues)
    defR0 = V.replicate agents $ defaultR0 (fromMaybe defInitValues initValues)
    defR1 = V.replicate agents $ defaultR1 (fromMaybe defInitValues initValues)
    agents = settings ^. independentAgents

-- Neural network approximations

mkUnichainHasktorch ::
  forall s as . (Eq as, NFData as, Ord as, Bounded as, Enum as, NFData s) =>
     Algorithm s
  -> InitialStateFun s
  -> FeatureExtractor s
  -> ActionFunction s as
  -> ActionFilter s
  -> ParameterInitValues
  -> ParameterDecaySetting
  -> ModelBuilderFunHT
  -> NNConfig
  -> Settings
  -> Maybe InitValues
  -> IO (ARAL s as)
mkUnichainHasktorch = mkUnichainHasktorchAs [minBound .. maxBound]


mkUnichainHasktorchAs ::
  forall s as . (Eq as, NFData as, Ord as, Enum as, NFData s) =>
     [Action as]
  -> Algorithm s
  -> InitialStateFun s
  -> FeatureExtractor s
  -> ActionFunction s as
  -> ActionFilter s
  -> ParameterInitValues
  -> ParameterDecaySetting
  -> ModelBuilderFunHT
  -> NNConfig
  -> Settings
  -> Maybe InitValues
  -> IO (ARAL s as)
mkUnichainHasktorchAs = mkUnichainHasktorchAsSAM Nothing

mkUnichainHasktorchAsSAM ::
  forall s as . (Eq as, NFData as, Ord as, Enum as, NFData s) =>
     Maybe (Int, Double) -- ^ SAM config
  -> [Action as]
  -> Algorithm s
  -> InitialStateFun s
  -> FeatureExtractor s
  -> ActionFunction s as
  -> ActionFilter s
  -> ParameterInitValues
  -> ParameterDecaySetting
  -> ModelBuilderFunHT
  -> NNConfig
  -> Settings
  -> Maybe InitValues
  -> IO (ARAL s as)
mkUnichainHasktorchAsSAM = mkUnichainHasktorchAsSAMAC False

mkUnichainHasktorchAsSAMAC ::
  forall s as . (Eq as, NFData as, Ord as, Enum as, NFData s) =>
     Bool
  -> Maybe (Int, Double) -- ^ SAM config
  -> [Action as]
  -> Algorithm s
  -> InitialStateFun s
  -> FeatureExtractor s
  -> ActionFunction s as
  -> ActionFilter s
  -> ParameterInitValues
  -> ParameterDecaySetting
  -> ModelBuilderFunHT
  -> NNConfig
  -> Settings
  -> Maybe InitValues
  -> IO (ARAL s as)
mkUnichainHasktorchAsSAMAC actorCritic mSAM as alg initialStateFun ftExt asFun asFilter params decayFun modelBuilderHT nnConfig settings initValues = do
  $(logPrintDebugText) "Creating unichain ARAL with Hasktorch"
  initialState <- initialStateFun MainAgent
  let feats = fromIntegral $ V.length (ftExt initialState)
      rows = genericLength as * fromIntegral (settings ^. independentAgents)
      netFun cols = modelBuilderHT feats (rows, cols)
  let model = netFun 1
  putStrLn "Net: "
  print model
  repMem <- mkReplayMemories as settings nnConfig
  let nnConfig' = set replayMemoryMaxSize (maybe 1 replayMemoriesSize repMem) nnConfig
  let opt w = mkAdamW 0.9 0.999 (Torch.flattenParameters w) 1e-2 -- 1e-4 -- 1e-3
  let nnSA tp =
        case alg of
          AlgARAL {}
            | tp /= R0Table && tp /= R1Table -> nnEmpty tp
          _ -> do
            modelT <- Torch.sample model
            modelW <- Torch.sample model
            let tp'
                  | actorCritic = NoScaling tp Nothing
                  | otherwise = tp
                modelT' = modelT {mlpIsPolicy = actorCritic}
            return $ Hasktorch modelT' modelW tp' nnConfig' (length as) (settings ^. independentAgents) (opt modelT') (opt modelW) model WelfordExistingAggregateEmpty mSAM
      nnEmpty tp =
        return $
        Hasktorch
          (MLP [] Torch.relu [] Nothing Nothing Nothing Nothing HasktorchHuber False)
          (MLP [] Torch.relu [] Nothing Nothing Nothing Nothing HasktorchHuber False)
          tp
          nnConfig'
          (length as)
          (settings ^. independentAgents)
          (mkAdamW 0.9 0.999 [] 1e-4)
          (mkAdamW 0.9 0.999 [] 1e-4)
          model
          WelfordExistingAggregateEmpty
          mSAM
  nnSAVTable <- nnSA VTable
  nnSAWTable <- nnSA WTable
  nnSAR0Table <- nnSA R0Table
  nnSAR1Table <- nnSA R1Table
  nnPsiV <- nnSA PsiVTable
  nnPsiW <- nnSA PsiWTable
  let nnType
        | isAlgDqnAvgRewardAdjusted alg = CombinedUnichain -- ScaleAs VTable
        | otherwise = CombinedUnichain
  let nnComb = nnSA nnType
  let proxies' =
        Proxies
          (Scalar (V.replicate agents defRhoMin) (length as))
          (Scalar (V.replicate agents defRho) (length as))
          nnPsiV
          nnSAVTable
          nnPsiW
          nnSAWTable
          nnSAR0Table
          nnSAR1Table
          repMem
          -- D2Sing SNat SNat -> ProxiesCombinedUnichain (Scalar (V.replicate agents defRhoMin) (length as)) (Scalar (V.replicate agents defRho) (length as)) nnComb repMem
          -- _ -> error "3D output is not supported by ARAL!"
  workers' <- liftIO $ mkWorkers initialStateFun as (Just nnConfig) settings
  return $!
    force $
    ARAL
      (VB.fromList as)
      asFun
      asFilter
      initialState
      workers'
      ftExt
      0
      (0, 0)
      params
      decayFun
      settings
      VB.empty
      (convertAlgorithm ftExt alg)
      Maximise
      defRhoMin
      mempty
      mempty
      (toValue agents 0, toValue agents 0, toValue agents 0)
      proxies'
  where
    defRho = defaultRho (fromMaybe defInitValues initValues)
    defRhoMin = defaultRhoMinimum (fromMaybe defInitValues initValues)
    agents = settings ^. independentAgents


mkUnichainGrenade ::
  forall s as . (Eq as, NFData as, Ord as, Bounded as, Enum as, NFData s) =>
     Algorithm s
  -> InitialStateFun s
  -> FeatureExtractor s
  -> ActionFunction s as
  -> ActionFilter s
  -> ParameterInitValues
  -> ParameterDecaySetting
  -> ModelBuilderFun
  -> NNConfig
  -> Settings
  -> Maybe InitValues
  -> IO (ARAL s as)
mkUnichainGrenade = mkUnichainGrenadeAs ([minBound .. maxBound] :: [Action as])

mkUnichainGrenadeAs ::
  forall s as . (Eq as, NFData as, Ord as, Bounded as, Enum as, NFData s) =>
     [as]
  -> Algorithm s
  -> InitialStateFun s
  -> FeatureExtractor s
  -> ActionFunction s as
  -> ActionFilter s
  -> ParameterInitValues
  -> ParameterDecaySetting
  -> ModelBuilderFun
  -> NNConfig
  -> Settings
  -> Maybe InitValues
  -> IO (ARAL s as)
mkUnichainGrenadeAs as alg initialStateFun ftExt asFun asFilter params decayFun modelBuilder nnConfig settings initValues = do
  $(logPrintDebugText) "Creating unichain ARAL with Grenade"
  initialState <- initialStateFun MainAgent
  let feats = fromIntegral $ V.length (ftExt initialState)
      rows = genericLength as * fromIntegral (settings ^. independentAgents)
      netFun cols = modelBuilder feats (rows, cols)
  specNet <- netFun 1
  fmap (checkNetworkOutput False) $ case specNet of
    SpecConcreteNetwork1D1D{} -> netFun 1 >>= (\(SpecConcreteNetwork1D1D net) -> mkUnichainGrenadeHelper as alg initialState initialStateFun ftExt asFun asFilter params decayFun nnConfig settings initValues net)
    SpecConcreteNetwork1D2D{} -> netFun 1 >>= (\(SpecConcreteNetwork1D2D net) -> mkUnichainGrenadeHelper as alg initialState initialStateFun ftExt asFun asFilter params decayFun nnConfig settings initValues net)
    SpecConcreteNetwork1D3D{} -> netFun 1 >>= (\(SpecConcreteNetwork1D3D net) -> mkUnichainGrenadeHelper as alg initialState initialStateFun ftExt asFun asFilter params decayFun nnConfig settings initValues net)
    _ -> error "ARAL currently requieres a 1D input and either 1D, 2D or 3D output"
    -- also fix in Serialisable if enabled!!!
    -- SpecConcreteNetwork2D1D{} -> netFun 1 >>= (\(SpecConcreteNetwork2D1D net) -> mkUnichainGrenadeHelper as alg initialState ftExt as asFilter params decayFun nnConfig initValues net)
    -- SpecConcreteNetwork2D2D{} -> netFun 1 >>= (\(SpecConcreteNetwork2D2D net) -> mkUnichainGrenadeHelper as alg initialState ftExt as asFilter params decayFun nnConfig initValues net)
    -- SpecConcreteNetwork2D3D{} -> netFun 1 >>= (\(SpecConcreteNetwork2D3D net) -> mkUnichainGrenadeHelper as alg initialState ftExt as asFilter params decayFun nnConfig initValues net)
    -- SpecConcreteNetwork3D1D{} -> netFun 1 >>= (\(SpecConcreteNetwork3D1D net) -> mkUnichainGrenadeHelper as alg initialState ftExt as asFilter params decayFun nnConfig initValues net)
    -- SpecConcreteNetwork3D2D{} -> netFun 1 >>= (\(SpecConcreteNetwork3D2D net) -> mkUnichainGrenadeHelper as alg initialState ftExt as asFilter params decayFun nnConfig initValues net)
    -- SpecConcreteNetwork3D3D{} -> netFun 1 >>= (\(SpecConcreteNetwork3D3D net) -> mkUnichainGrenadeHelper as alg initialState ftExt as asFilter params decayFun nnConfig initValues net)


-- | Modelbuilder takes the number of output columns, which determins if the ANN is 1D or 2D! (#actions, #columns, 1)
mkUnichainGrenadeCombinedNet ::
     forall as s . (Eq as, Ord as, NFData as, Enum as, Bounded as, NFData s) =>
     Algorithm s
  -> InitialStateFun s
  -> FeatureExtractor s
  -> ActionFunction s as
  -> ActionFilter s
  -> ParameterInitValues
  -> ParameterDecaySetting
  -> ModelBuilderFun
  -> NNConfig
  -> Settings
  -> Maybe InitValues
  -> IO (ARAL s as)
mkUnichainGrenadeCombinedNet alg initialStateFun ftExt asFun asFilter params decayFun modelBuilder nnConfig settings initValues = do
  let nrNets | isAlgDqn alg = 1
             | isAlgDqnAvgRewardAdjusted alg = 2
             | otherwise = 6
  initialState <- initialStateFun MainAgent
  let feats = fromIntegral $ V.length (ftExt initialState)
      rows = genericLength as * fromIntegral (settings ^. independentAgents)
      netFun cols = modelBuilder feats (rows, cols)
  specNet <- netFun nrNets
  fmap (checkNetworkOutput True) $ case specNet of
    SpecConcreteNetwork1D1D{} -> netFun nrNets >>= (\(SpecConcreteNetwork1D1D net) -> mkUnichainGrenadeHelper as alg initialState initialStateFun ftExt asFun asFilter params decayFun nnConfig settings initValues net)
    SpecConcreteNetwork1D2D{} -> netFun nrNets >>= (\(SpecConcreteNetwork1D2D net) -> mkUnichainGrenadeHelper as alg initialState initialStateFun ftExt asFun asFilter params decayFun nnConfig settings initValues net)
    SpecConcreteNetwork1D3D{} -> netFun nrNets >>= (\(SpecConcreteNetwork1D3D net) -> mkUnichainGrenadeHelper as alg initialState initialStateFun ftExt asFun asFilter params decayFun nnConfig settings initValues net)
    _ -> error "ARAL currently requieres a 1D input and either 1D, 2D or 3D output"
    -- SpecConcreteNetwork2D1D{} -> netFun nrNets >>= (\(SpecConcreteNetwork2D1D net) -> mkUnichainGrenadeHelper as alg initialState initialStateFun ftExt asFun asFilter params decayFun nnConfig initValues net)
    -- SpecConcreteNetwork2D2D{} -> netFun nrNets >>= (\(SpecConcreteNetwork2D2D net) -> mkUnichainGrenadeHelper as alg initialState initialStateFun ftExt asFun asFilter params decayFun nnConfig initValues net)
    -- SpecConcreteNetwork2D3D{} -> netFun nrNets >>= (\(SpecConcreteNetwork2D3D net) -> mkUnichainGrenadeHelper as alg initialState initialStateFun ftExt asFun asFilter params decayFun nnConfig initValues net)
    -- SpecConcreteNetwork3D1D{} -> netFun nrNets >>= (\(SpecConcreteNetwork3D1D net) -> mkUnichainGrenadeHelper as alg initialState initialStateFun ftExt asFun asFilter params decayFun nnConfig initValues net)
    -- SpecConcreteNetwork3D2D{} -> netFun nrNets >>= (\(SpecConcreteNetwork3D2D net) -> mkUnichainGrenadeHelper as alg initialState initialStateFun ftExt asFun asFilter params decayFun nnConfig initValues net)
    -- SpecConcreteNetwork3D3D{} -> netFun nrNets >>= (\(SpecConcreteNetwork3D3D net) -> mkUnichainGrenadeHelper as alg initialState initialStateFun ftExt asFun asFilter params decayFun nnConfig initValues net)

  where as = [minBound .. maxBound] :: [as]

mkUnichainGrenadeHelper ::
     forall as s layers shapes nrH.
     ( Enum as, Bounded as, Eq as, Ord as, NFData as
     , GNum (Gradients layers)
     , FoldableGradient (Gradients layers)
     , KnownNat nrH
     , Typeable layers
     , Typeable shapes
     , Head shapes ~ 'D1 nrH
     , SingI (Last shapes)
     , NFData (Tapes layers shapes)
     , NFData (Gradients layers)
     , NFData (Network layers shapes)
     , Serialize (Gradients layers)
     , Serialize (Network layers shapes)
     , Show (Network layers shapes)
     , FromDynamicLayer (Network layers shapes)
     , GNum (Network layers shapes)
     , NFData s
     )
  => [as]
  -> Algorithm s
  -> s
  -> InitialStateFun s
  -> FeatureExtractor s
  -> ActionFunction s as
  -> ActionFilter s
  -> ParameterInitValues
  -> ParameterDecaySetting
  -> NNConfig
  -> Settings
  -> Maybe InitValues
  -> Network layers shapes
  -> IO (ARAL s as)
mkUnichainGrenadeHelper as alg initialState initialStateFun ftExt asFun asFilter params decayFun nnConfig settings initValues net = do
  putStrLn "Using following Greande Specification: "
  print $ networkToSpecification net
  putStrLn "Net: "
  print net
  repMem <- mkReplayMemories as settings nnConfig
  let nnConfig' = set replayMemoryMaxSize (maybe 1 replayMemoriesSize repMem) nnConfig
  let nnSA tp = Grenade net net tp nnConfig' (length as) (settings ^. independentAgents) WelfordExistingAggregateEmpty
  let nnSAVTable = nnSA VTable
  let nnSAWTable = nnSA WTable
  let nnSAR0Table = nnSA R0Table
  let nnSAR1Table = nnSA R1Table
  let nnPsiV = nnSA PsiVTable
  let nnPsiW = nnSA PsiWTable
  let nnType
        | isAlgDqnAvgRewardAdjusted alg = CombinedUnichain -- ScaleAs VTable
        | otherwise = CombinedUnichain
  let nnComb = nnSA nnType
  let proxies' =
        case (sing :: Sing (Last shapes)) of
          D1Sing SNat ->
            Proxies
              (Scalar (V.replicate agents defRhoMin) (length as))
              (Scalar (V.replicate agents defRho) (length as))
              nnPsiV
              nnSAVTable
              nnPsiW
              nnSAWTable
              nnSAR0Table
              nnSAR1Table
              repMem
          D2Sing SNat SNat -> ProxiesCombinedUnichain (Scalar (V.replicate agents defRhoMin) (length as)) (Scalar (V.replicate agents defRho) (length as)) nnComb repMem
          _ -> error "3D output is not supported by ARAL!"
  workers' <- liftIO $ mkWorkers initialStateFun as (Just nnConfig) settings
  return $! force $
    ARAL
      (VB.fromList as)
      asFun
      asFilter
      initialState
      workers'
      ftExt
      0
      (0, 0)
      params
      decayFun
      settings
      VB.empty
      (convertAlgorithm ftExt alg)
      Maximise
      defRhoMin
      mempty
      mempty
      (toValue agents 0, toValue agents 0, toValue agents 0)
      proxies'
  where
    defRho = defaultRho (fromMaybe defInitValues initValues)
    defRhoMin = defaultRhoMinimum (fromMaybe defInitValues initValues)
    agents = settings ^. independentAgents

mkMultichainGrenade ::
     forall as nrH nrL s layers shapes.
     ( Enum as, Bounded as, Eq as, Ord as, NFData as
     , GNum (Gradients layers)
     , GNum (Network layers shapes)
     , FoldableGradient (Gradients layers)
     , Typeable layers
     , Typeable shapes
     , KnownNat nrH
     , Head shapes ~ 'D1 nrH
     , KnownNat nrL
     , Last shapes ~ 'D1 nrL
     , NFData (Tapes layers shapes)
     , NFData (Gradients layers)
     , NFData (Network layers shapes)
     , Serialize (Network layers shapes)
     , Serialize (Gradients layers)
     , FromDynamicLayer (Network layers shapes)
     , NFData s
     )
  => Algorithm s
  -> InitialStateFun s
  -> FeatureExtractor s
  -> ActionFunction s as
  -> ActionFilter s
  -> ParameterInitValues
  -> ParameterDecaySetting
  -> Network layers shapes
  -> NNConfig
  -> Settings
  -> Maybe InitValues
  -> IO (ARAL s as)
mkMultichainGrenade alg initialStateFun ftExt asFun asFilter params decayFun net nnConfig settings initVals = do
  let as = [minBound .. maxBound] :: [as]
  repMem <- mkReplayMemories as settings nnConfig
  let nnConfig' = set replayMemoryMaxSize (maybe 1 replayMemoriesSize repMem) nnConfig
  let nnSA tp = Grenade net net tp nnConfig' (length as) (settings ^. independentAgents) WelfordExistingAggregateEmpty
  let nnSAMinRhoTable = nnSA VTable
  let nnSARhoTable = nnSA VTable
  let nnSAVTable = nnSA VTable
  let nnSAWTable = nnSA WTable
  let nnSAR0Table = nnSA R0Table
  let nnSAR1Table = nnSA R1Table
  let nnPsiV = nnSA PsiVTable
  let nnPsiW = nnSA PsiWTable
  initialState <- initialStateFun MainAgent
  let proxies' = Proxies nnSAMinRhoTable nnSARhoTable nnPsiV nnSAVTable nnPsiW nnSAWTable nnSAR0Table nnSAR1Table repMem
  workers <- liftIO $ mkWorkers initialStateFun as (Just nnConfig) settings
  return $! force $ checkNetworkOutput False $
    ARAL
      (VB.fromList as)
      asFun
      asFilter
      initialState
      workers
      ftExt
      0
      (0, 0)
      params
      decayFun
      settings
      VB.empty
      (convertAlgorithm ftExt alg)
      Maximise
      defRhoMin
      mempty
      mempty
      (toValue agents 0, toValue agents 0, toValue agents 0)
      proxies'
  where
    agents = settings ^. independentAgents
    defRhoMin = defaultRhoMinimum (fromMaybe defInitValues initVals)

------------------------------ Replay Memory/Memories ------------------------------

mkReplayMemories :: [Action as] -> Settings -> NNConfig -> IO (Maybe ReplayMemories)
mkReplayMemories = mkReplayMemories' False

mkReplayMemories' :: Bool -> [Action as] -> Settings -> NNConfig -> IO (Maybe ReplayMemories)
mkReplayMemories' allowSz1 as setts nnConfig =
  case nnConfig ^. replayMemoryStrategy of
    ReplayMemorySingle -> fmap (ReplayMemoriesUnified (length as)) <$> mkReplayMemory allowSz1 repMemSizeSingle
    ReplayMemoryPerAction -> do
      tmpRepMem <- mkReplayMemory allowSz1 (setts ^. nStep)
      fmap (ReplayMemoriesPerActions (length as) tmpRepMem) . sequence . VB.fromList <$> replicateM (length as * agents) (mkReplayMemory allowSz1 repMemSizePerAction)
  where
    agents = setts ^. independentAgents
    repMemSizeSingle = max (nnConfig ^. replayMemoryMaxSize) (setts ^. nStep * nnConfig ^. trainBatchSize)
    repMemSizePerAction = (size `div` (setts ^. nStep)) * (setts ^. nStep)
      where
        size = max (ceiling $ fromIntegral (nnConfig ^. replayMemoryMaxSize) / fromIntegral (length as * agents)) (setts ^. nStep)


mkReplayMemory :: Bool -> Int -> IO (Maybe ReplayMemory)
mkReplayMemory allowSz1 sz | sz <= 1 && not allowSz1 = return Nothing
mkReplayMemory _ sz = do
  vec <- VM.new sz
  return $ Just $ ReplayMemory vec sz 0 (-1)


-------------------- Other Constructors --------------------

-- | Infer scaling by maximum reward.
scalingByMaxAbsReward :: Bool -> Double -> ScalingNetOutParameters
scalingByMaxAbsReward onlyPositive maxR = ScalingNetOutParameters (-maxV) maxV (-maxW) maxW (if onlyPositive then 0 else -maxR0) maxR0 (if onlyPositive then 0 else -maxR1) maxR1
  where maxDiscount g = sum $ take 10000 $ map (\p -> (g^p) * maxR) [(0::Int)..]
        maxV = 1.0 * maxR
        maxW = 50 * maxR
        maxR0 = 2 * maxDiscount defaultGamma0
        maxR1 = 1.0 * maxDiscount defaultGamma1

scalingByMaxAbsRewardAlg :: Algorithm s -> Bool -> Double -> ScalingNetOutParameters
scalingByMaxAbsRewardAlg alg onlyPositive maxR =
  case alg of
    AlgARAL{} -> ScalingNetOutParameters (-maxR1) maxR1 (-maxW) maxW (-maxR1) maxR1 (-maxR1) maxR1
    _         -> scalingByMaxAbsReward onlyPositive maxR
  where
    maxW = 50 * maxR
    maxR1 = 1.0 * maxR

-- | Creates the workers data structure if applicable (i.e. there is a replay memory of size >1 AND the minimum
-- expoloration rates are configured in NNConfig).
mkWorkers :: InitialStateFun s -> [Action as] -> Maybe NNConfig -> Settings -> IO (Workers s)
mkWorkers state as mNNConfig setts = do
  let nr = length $ setts ^. workersMinExploration
      workerTypes = map WorkerAgent [1 .. nr]
      nStepSett = setts ^. nStep
  if nr <= 0
    then return []
    else do
      repMems <- replicateM nr (maybe (fmap (ReplayMemoriesUnified (length as)) <$> mkReplayMemory True (max 1 $ setts ^. nStep)) (mkReplayMemories' True as setts) mNNConfig)
      states <- mapM state workerTypes
      return $ zipWith3 (\wNr st rep -> WorkerState wNr st (fromMaybe err rep) VB.empty 0) [1 ..] states repMems
  where
    err = error $ "Could not create replay memory for workers with nStep=" ++ show (setts ^. nStep) ++ " and memMaxSize=" ++ show (view replayMemoryMaxSize <$> mNNConfig)

-------------------- Helpers --------------------

checkNetworkOutput :: Bool -> ARAL s as -> ARAL s as
checkNetworkOutput combined borl
  | not isAnn = borl
  | (reqX, reqY, reqZ) /= (x, y, z) = error $ "Expected ANN output: " ++ show (reqX, reqY, reqZ) ++ ". Actual ANN output: " ++ show (x, y, z)
  | otherwise = borl
  where
    isAnn :: Bool
    isAnn = any isNeuralNetwork (allProxies $ borl ^. proxies)
    reqZ = 1 :: Integer
    reqX = fromIntegral $ length (borl ^. actionList) * (borl ^. settings . independentAgents) :: Integer
    (x, y, z) = case px of
      Grenade t _ _ _ _ _ _                     -> mkDims t
      CombinedProxy (Grenade t _ _ _ _ _ _) _ _ -> mkDims t
      px'                                       -> error $ "Error in checkNetworkOutput. This should not have happend. Proxy is: "++ show px'
    mkDims :: forall layers shapes . (SingI (Last shapes)) => Network layers shapes -> (Integer,Integer,Integer)
    mkDims _ = tripleFromSomeShape (SomeSing (sing :: Sing (Last shapes)))
    px =
      case borl ^. algorithm of
        AlgARAL {}      -> borl ^. proxies . r1
        AlgNBORL {}     -> borl ^. proxies . v
        AlgARALVOnly {} -> borl ^. proxies . v
        AlgRLearning    -> borl ^. proxies . v
        AlgDQN {}       -> borl ^. proxies . r1
    reqY :: Integer
    reqY
      | not combined = 1
      | otherwise =
        case borl ^. algorithm of
          AlgARAL {}      -> 2
          AlgDQN {}       -> 1
          AlgRLearning {} -> 1
          _               -> 6


-- | Perform an action over all proxies (combined proxies are seen once only).
overAllProxies :: ((a -> Identity b) -> Proxy -> Identity Proxy) -> (a -> b) -> ARAL s as -> ARAL s as
overAllProxies l f borl = foldl' (\b p -> over (proxies . p . l) f b) borl (allProxiesLenses (borl ^. proxies))

setAllProxies :: ((a -> Identity b) -> Proxy -> Identity Proxy) -> b -> ARAL s as -> ARAL s as
setAllProxies l = overAllProxies l . const

allProxies :: Proxies -> [Proxy]
allProxies pxs = map (pxs ^. ) (allProxiesLenses pxs)
