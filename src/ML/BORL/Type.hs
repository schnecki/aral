{-# LANGUAGE BangPatterns         #-}
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

module ML.BORL.Type
  ( -- objective
    Objective (..)
  , setObjective
  , flipObjective
    -- common
  , ActionIndexed
  , RewardFutureData (..)
  , futurePeriod
  , futureState
  , futureActionNr
  , futureRandomAction
  , futureReward
  , futureStateNext
  , futureEpisodeEnd
  , mapRewardFutureData
  , idxStart
    -- BORL
  , BORL (..)
  , actionList
  , actionFilter
  , s
  , featureExtractor
  , t
  , episodeNrStart
  , parameters
  , decayFunction
  , futureRewards
  , algorithm
  , objective
  , lastVValues
  , lastRewards
  , psis
  , proxies
    -- actions
  , actionsIndexed
  , filteredActions
  , filteredActionIndexes
    -- initial values
  , InitValues (..)
  , defInitValues
    -- constructors
  , mkUnichainTabular
  , mkMultichainTabular

  , mkTensorflowModel
  , mkUnichainTensorflowM
  , mkUnichainTensorflowCombinedNetM
  , mkUnichainTensorflow
  , mkUnichainTensorflowCombinedNet

  , mkUnichainGrenade
  , mkUnichainGrenadeCombinedNet
  , mkMultichainGrenade

    -- scaling
  , scalingByMaxAbsReward
  , scalingByMaxAbsRewardAlg
    -- proxy/proxies helpers
  , overAllProxies
  , setAllProxies
  , allProxies
  ) where

import           ML.BORL.Action.Type
import           ML.BORL.Algorithm
import           ML.BORL.Decay
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import           ML.BORL.Proxy.Proxies
import           ML.BORL.Proxy.Type
import           ML.BORL.Reward.Type
import           ML.BORL.Types

import           Control.DeepSeq
import           Control.Lens
import           Control.Monad                (zipWithM)
import           Control.Monad.IO.Class       (MonadIO, liftIO)
import           Data.List                    (foldl')
import qualified Data.Map.Strict              as M
import           Data.Maybe                   (fromMaybe)
import qualified Data.Proxy                   as Type
import           Data.Serialize
import           Data.Singletons.Prelude.List
import qualified Data.Text                    as T
import qualified Data.Vector.Mutable          as V
import           GHC.Generics
import           GHC.TypeLits
import           Grenade
import           System.IO
import qualified TensorFlow.Core              as TF
import qualified TensorFlow.Session           as TF

import           Debug.Trace


-------------------- Main RL Datatype --------------------

type ActionIndexed s = (ActionIndex, Action s) -- ^ An action with index.

data Objective
  = Minimise
  | Maximise
  deriving (Eq, Ord, NFData, Generic, Show, Serialize)

data RewardFutureData s = RewardFutureData
                { _futurePeriod       :: Period
                , _futureState        :: State s
                , _futureActionNr     :: ActionIndex
                , _futureRandomAction :: IsRandomAction
                , _futureReward       :: Reward s
                , _futureStateNext    :: StateNext s
                , _futureEpisodeEnd   :: Bool
                } deriving (Generic, NFData, Serialize)
makeLenses ''RewardFutureData

mapRewardFutureData :: (RewardFuture s') => (s -> s') -> (StoreType s -> StoreType s') -> RewardFutureData s -> RewardFutureData s'
mapRewardFutureData f g (RewardFutureData p s aNr rand rew stateNext epEnd) = RewardFutureData p (f s) aNr rand (mapReward g rew) (f stateNext) epEnd


data BORL s = BORL
  { _actionList       :: ![ActionIndexed s]    -- ^ List of possible actions in state s.
  , _actionFilter     :: !(s -> [Bool])        -- ^ Function to filter actions in state s.
  , _s                :: !s                    -- ^ Current state.
  , _featureExtractor :: !(s -> [Double])      -- ^ Function that extracts the features of a state.
  , _t                :: !Int                  -- ^ Current time t.
  , _episodeNrStart   :: !(Int, Int)           -- ^ Nr of Episode and start period.
  , _parameters       :: !ParameterInitValues  -- ^ Parameter setup.
  , _decayFunction    :: !Decay                -- ^ Decay function at period t.
  , _futureRewards    :: ![RewardFutureData s] -- ^ List of future reward.

  -- define algorithm to use
  , _algorithm        :: !(Algorithm [Double]) -- ^ What algorithm to use.
  , _objective        :: !Objective            -- ^ Objective to minimise or maximise.

  -- Values:
  , _lastVValues      :: ![Double]                 -- ^ List of X last V values (head is last seen value)
  , _lastRewards      :: ![Double]                 -- ^ List of X last rewards (head is last received reward)
  , _psis             :: !(Double, Double, Double) -- ^ Exponentially smoothed psi values.
  , _proxies          :: Proxies                   -- ^ Scalar, Tables and Neural Networks
  }
makeLenses ''BORL

instance (NFData s) => NFData (BORL s) where
  rnf (BORL as af s ftExt t epNr par dec fut alg ph lastVs lastRews psis proxies) =
    rnf as `seq` rnf af `seq` rnf s `seq` rnf ftExt `seq`
    rnf t `seq` rnf epNr `seq` rnf par `seq` rnf dec `seq` rnf fut `seq` rnf alg `seq` rnf ph `seq` rnf lastVs `seq` rnf lastRews `seq` rnf proxies `seq` rnf psis `seq` rnf s

------------------------------ Indexed Action ------------------------------


actionsIndexed :: BORL s -> s -> [ActionIndexed s]
actionsIndexed borl state = map snd $ filter fst $ zip ((borl ^. actionFilter) state) (borl ^. actionList)


filteredActions :: [Action a] -> (s -> [Bool]) -> s -> [Action a]
filteredActions actions actFilter state = map (snd.snd) $ filter fst $ zip (actFilter state) (zip [(0::Int)..] actions)

filteredActionIndexes :: [Action a] -> (s -> [Bool]) -> s -> [ActionIndex]
filteredActionIndexes actions actFilter state = map (fst.snd) $ filter fst $ zip (actFilter state) (zip [(0::Int)..] actions)


------------------------------ Initial Values ------------------------------


idxStart :: Int
idxStart = 0


data InitValues = InitValues
  { defaultRho        :: Double
  , defaultRhoMinimum :: Double
  , defaultV          :: Double
  , defaultW          :: Double
  , defaultR0         :: Double
  , defaultR1         :: Double
  }


defInitValues :: InitValues
defInitValues = InitValues 0 0 0 0 0 0


-------------------- Objective --------------------

setObjective :: Objective -> BORL s -> BORL s
setObjective obj = objective .~ obj

-- | Default objective is Maximise.
flipObjective :: BORL s -> BORL s
flipObjective borl = case borl ^. objective of
  Minimise -> objective .~ Maximise $ borl
  Maximise -> objective .~ Minimise $ borl


-------------------- Constructors --------------------

-- Tabular representations

convertAlgorithm :: FeatureExtractor s -> Algorithm s -> Algorithm [Double]
convertAlgorithm ftExt (AlgBORL g0 g1 avgRew (Just (s,a))) = AlgBORL g0 g1 avgRew (Just (ftExt s,a))
convertAlgorithm ftExt (AlgBORLVOnly avgRew (Just (s, a))) = AlgBORLVOnly avgRew (Just (ftExt s, a))
convertAlgorithm _ (AlgBORL g0 g1 avgRew Nothing) = AlgBORL g0 g1 avgRew Nothing
convertAlgorithm _ (AlgBORLVOnly avgRew Nothing) = AlgBORLVOnly avgRew Nothing
convertAlgorithm _ (AlgDQN ga cmp) = AlgDQN ga cmp
convertAlgorithm _ (AlgDQNAvgRewAdjusted mEpsGa1 ga1 ga2 avgRew) = AlgDQNAvgRewAdjusted mEpsGa1 ga1 ga2 avgRew


mkUnichainTabular :: Algorithm s -> InitialState s -> FeatureExtractor s -> [Action s] -> (s -> [Bool]) -> ParameterInitValues -> Decay -> Maybe InitValues -> BORL s
mkUnichainTabular alg initialState ftExt as asFilter params decayFun initVals =
  BORL
    (zip [idxStart ..] as)
    asFilter
    initialState
    ftExt
    0
    (0, 0)
    params
    decayFun
    mempty
    (convertAlgorithm ftExt alg)
    Maximise
    mempty
    mempty
    (0, 0, 0)
    (Proxies (Scalar defRhoMin) (Scalar defRho) (tabSA 0) (tabSA defV) (tabSA 0) (tabSA defW) (tabSA defR0) (tabSA defR1) Nothing)
  where
    tabSA def = Table mempty def
    defRhoMin = defaultRhoMinimum (fromMaybe defInitValues initVals)
    defRho = defaultRho (fromMaybe defInitValues initVals)
    defV = defaultV (fromMaybe defInitValues initVals)
    defW = defaultW (fromMaybe defInitValues initVals)
    defR0 = defaultR0 (fromMaybe defInitValues initVals)
    defR1 = defaultR1 (fromMaybe defInitValues initVals)

mkTensorflowModel :: (MonadBorl' m) => [a2] -> ProxyType -> T.Text -> [Double] -> TF.SessionT IO TensorflowModel -> m TensorflowModel'
mkTensorflowModel as tp scope netInpInitState modelBuilderFun = do
      !model <- prependName (proxyTypeName tp <> scope) <$> liftTensorflow modelBuilderFun
      saveModel
        (TensorflowModel' model Nothing (Just (map realToFrac netInpInitState, replicate (length as) 0)) modelBuilderFun)
        [map realToFrac netInpInitState]
        [replicate (length as) 0]
  where
    prependName txt model =
      model {inputLayerName = txt <> "/" <> inputLayerName model, outputLayerName = txt <> "/" <> outputLayerName model, labelLayerName = txt <> "/" <> labelLayerName model}


mkUnichainTensorflowM ::
     forall s m. (NFData s, MonadBorl' m)
  => Algorithm s
  -> InitialState s
  -> FeatureExtractor s
  -> [Action s]
  -> (s -> [Bool])
  -> ParameterInitValues
  -> Decay
  -> ModelBuilderFunction
  -> NNConfig
  -> Maybe InitValues
  -> m (BORL s)
mkUnichainTensorflowM alg initialState ftExt as asFilter params decayFun modelBuilder nnConfig initValues = do
  let nnTypes = [VTable, VTable, WTable, WTable, R0Table, R0Table, R1Table, R1Table, PsiVTable, PsiVTable, PsiWTable, PsiWTable]
      scopes = concat $ repeat ["_target", "_worker"]
  let fullModelInit = sequenceA (zipWith3 (\tp sc fun -> TF.withNameScope (proxyTypeName tp <> sc) fun) nnTypes scopes (repeat $ modelBuilder 1))
  let netInpInitState = ftExt initialState
      nnSA :: ProxyType -> Int -> IO Proxy
      nnSA tp idx = do
        nnT <- runMonadBorlTF $ mkTensorflowModel as tp "_target" netInpInitState ((!! idx) <$> fullModelInit)
        nnW <- runMonadBorlTF $ mkTensorflowModel as tp "_worker" netInpInitState ((!! (idx + 1)) <$> fullModelInit)
        return $ TensorflowProxy nnT nnW mempty tp nnConfig (length as)
  if isAlgDqnAvgRewardFree alg
    then do
      r0 <- liftIO $ nnSA R0Table 4
      r1 <- liftIO $ nnSA VTable 0
      repMem <- liftIO $ mkReplayMemory (nnConfig ^. replayMemoryMaxSize)
      buildTensorflowModel (r0 ^?! proxyTFTarget)
      return $
        force $
        BORL
          (zip [idxStart ..] as)
          asFilter
          initialState
          ftExt
          0
          (0, 0)
          params
          decayFun
          mempty
          (convertAlgorithm ftExt alg)
          Maximise
          mempty
          mempty
          (0, 0, 0)
          (Proxies (Scalar defRhoMin) (Scalar defRho) (Scalar 0) r1 (Scalar 0) (Scalar 0) r0 r1 repMem)
    else do
      v <- liftIO $ nnSA VTable 0
      w <- liftIO $ nnSA WTable 2
      r0 <- liftIO $ nnSA R0Table 4
      r1 <- liftIO $ nnSA R1Table 6
      psiV <- liftIO $ nnSA PsiVTable 8
      psiW <- liftIO $ nnSA PsiWTable 10
      repMem <- liftIO $ mkReplayMemory (nnConfig ^. replayMemoryMaxSize)
      buildTensorflowModel (v ^?! proxyTFTarget)
      return $
        force $
        BORL
          (zip [idxStart ..] as)
          asFilter
          initialState
          ftExt
          0
          (0, 0)
          params
          decayFun
          mempty
          (convertAlgorithm ftExt alg)
          Maximise
          mempty
          mempty
          (0, 0, 0)
          (Proxies (Scalar defRhoMin) (Scalar defRho) psiV v psiW w r0 r1 repMem)
  where
    defRho = defaultRho (fromMaybe defInitValues initValues)
    defRhoMin = defaultRhoMinimum (fromMaybe defInitValues initValues)

-- ^ The output tensor must be 2D with the number of rows corresponding to the number of actions and the columns being
-- variable.
mkUnichainTensorflowCombinedNetM ::
     forall s m. (NFData s, MonadBorl' m)
  => Algorithm s
  -> InitialState s
  -> FeatureExtractor s
  -> [Action s]
  -> (s -> [Bool])
  -> ParameterInitValues
  -> Decay
  -> ModelBuilderFunction
  -> NNConfig
  -> Maybe InitValues
  -> m (BORL s)
mkUnichainTensorflowCombinedNetM alg initialState ftExt as asFilter params decayFun modelBuilder nnConfig initValues = do
  let nrNets | isAlgDqn alg = 1
             | isAlgDqnAvgRewardFree alg = 2
             | otherwise = 6
  let nnType | isAlgDqnAvgRewardFree alg = CombinedUnichain -- ScaleAs VTable
             | otherwise = CombinedUnichain
      scopes = ["_target", "_worker"]
  let fullModelInit = sequenceA (zipWith3 (\tp sc fun -> TF.withNameScope (proxyTypeName tp <> sc) fun) (repeat nnType) scopes (repeat (modelBuilder nrNets)))
  let netInpInitState = ftExt initialState
      nnSA :: ProxyType -> Int -> IO Proxy
      nnSA tp idx = do
        nnT <- runMonadBorlTF $ mkTensorflowModel (concat $ replicate (fromIntegral nrNets) as) tp "_target" netInpInitState ((!! idx) <$> fullModelInit)
        nnW <- runMonadBorlTF $ mkTensorflowModel (concat $ replicate (fromIntegral nrNets) as) tp "_worker" netInpInitState ((!! (idx + 1)) <$> fullModelInit)
        return $ TensorflowProxy nnT nnW mempty tp nnConfig (length as)
  proxy <- liftIO $ nnSA nnType 0
  repMem <- liftIO $ mkReplayMemory (nnConfig ^. replayMemoryMaxSize)
  buildTensorflowModel (proxy ^?! proxyTFTarget)
  return $
    force $
    BORL
      (zip [idxStart ..] as)
      asFilter
      initialState
      ftExt
      0
      (0, 0)
      params
      decayFun
      mempty
      (convertAlgorithm ftExt alg)
      Maximise
      mempty
      mempty
      (0, 0, 0)
      (ProxiesCombinedUnichain (Scalar defRhoMin) (Scalar defRho) proxy repMem)
  where
    defRho = defaultRho (fromMaybe defInitValues initValues)
    defRhoMin = defaultRhoMinimum (fromMaybe defInitValues initValues)


-- ^ Uses a single network for each value to learn. Thus the output tensor must be 1D with the number of rows
-- corresponding to the number of actions.
mkUnichainTensorflow ::
     forall s . (NFData s)
  => Algorithm s
  -> InitialState s
  -> FeatureExtractor s
  -> [Action s]
  -> (s -> [Bool])
  -> ParameterInitValues
  -> Decay
  -> ModelBuilderFunction
  -> NNConfig
  -> Maybe InitValues
  -> IO (BORL s)
mkUnichainTensorflow alg initialState ftExt as asFilter params decayFun modelBuilder nnConfig initValues =
  runMonadBorlTF (mkUnichainTensorflowM alg initialState ftExt as asFilter params decayFun modelBuilder nnConfig initValues)

-- ^ Use a single network for all function approximations. Thus, the output tensor must be 2D with the number of rows
-- corresponding to the number of actions and the columns being variable.
mkUnichainTensorflowCombinedNet ::
     forall s . (NFData s)
  => Algorithm s
  -> InitialState s
  -> FeatureExtractor s
  -> [Action s]
  -> (s -> [Bool])
  -> ParameterInitValues
  -> Decay
  -> ModelBuilderFunction
  -> NNConfig
  -> Maybe InitValues
  -> IO (BORL s)
mkUnichainTensorflowCombinedNet alg initialState ftExt as asFilter params decayFun modelBuilder nnConfig initValues =
  runMonadBorlTF (mkUnichainTensorflowCombinedNetM alg initialState ftExt as asFilter params decayFun modelBuilder nnConfig initValues)


mkMultichainTabular :: Algorithm s -> InitialState s -> FeatureExtractor s -> [Action s] -> (s -> [Bool]) -> ParameterInitValues -> Decay -> Maybe InitValues -> BORL s
mkMultichainTabular alg initialState ftExt as asFilter params decayFun initValues =
  BORL
    (zip [0 ..] as)
    asFilter
    initialState
    ftExt
    0
    (0, 0)
    params
    decayFun
    mempty
    (convertAlgorithm ftExt alg)
    Maximise
    mempty
    mempty
    (0, 0, 0)
    (Proxies (tabSA defRhoMin) (tabSA defRho) (tabSA 0) (tabSA defV) (tabSA 0) (tabSA defW) (tabSA defR0) (tabSA defR1) Nothing)
  where
    tabSA def = Table mempty def
    defRhoMin = defaultRhoMinimum (fromMaybe defInitValues initValues)
    defRho = defaultRho (fromMaybe defInitValues initValues)
    defV = defaultV (fromMaybe defInitValues initValues)
    defW = defaultW (fromMaybe defInitValues initValues)
    defR0 = defaultR0 (fromMaybe defInitValues initValues)
    defR1 = defaultR1 (fromMaybe defInitValues initValues)

-- Neural network approximations

mkUnichainGrenade ::
     forall nrH nrL s layers shapes. (GNum (Gradients layers), KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, Ord s, NFData (Tapes layers shapes), NFData (Network layers shapes), Serialize (Network layers shapes))
  => Algorithm s
  -> InitialState s
  -> FeatureExtractor s
  -> [Action s]
  -> (s -> [Bool])
  -> ParameterInitValues
  -> Decay
  -> Network layers shapes
  -> NNConfig
  -> Maybe InitValues
  -> IO (BORL s)
mkUnichainGrenade alg initialState ftExt as asFilter params decayFun net nnConfig initValues = do
  let nnSA tp = Grenade net net mempty tp nnConfig (length as)
  let nnSAVTable = nnSA VTable
  let nnSAWTable = nnSA WTable
  let nnSAR0Table = nnSA R0Table
  let nnSAR1Table = nnSA R1Table
  let nnPsiV = nnSA PsiVTable
  let nnPsiW = nnSA PsiWTable
  repMem <- mkReplayMemory (nnConfig ^. replayMemoryMaxSize)
  return $
    checkGrenade net 1 nnConfig $
    BORL
      (zip [idxStart ..] as)
      asFilter
      initialState
      ftExt
      0
      (0, 0)
      params
      decayFun
      []
      (convertAlgorithm ftExt alg)
      Maximise
      mempty
      mempty
      (0, 0, 0)
      (Proxies (Scalar defRhoMin) (Scalar defRho) nnPsiV nnSAVTable nnPsiW nnSAWTable nnSAR0Table nnSAR1Table repMem)
  where
    defRho = defaultRho (fromMaybe defInitValues initValues)
    defRhoMin = defaultRhoMinimum (fromMaybe defInitValues initValues)

mkUnichainGrenadeCombinedNet ::
     forall nrH nrL s layers shapes. (GNum (Gradients layers), KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, Ord s, NFData (Tapes layers shapes), NFData (Network layers shapes), Serialize (Network layers shapes))
  => Algorithm s
  -> InitialState s
  -> FeatureExtractor s
  -> [Action s]
  -> (s -> [Bool])
  -> ParameterInitValues
  -> Decay
  -> Network layers shapes
  -> NNConfig
  -> Maybe InitValues
  -> IO (BORL s)
mkUnichainGrenadeCombinedNet alg initialState ftExt as asFilter params decayFun net nnConfig initValues = do
  let nrNets | isAlgDqn alg = 1
             | isAlgDqnAvgRewardFree alg = 2
             | otherwise = 6
  let nnSA tp = Grenade net net mempty tp nnConfig (length as)
  let nnType | isAlgDqnAvgRewardFree alg = CombinedUnichain -- ScaleAs VTable
             | otherwise = CombinedUnichain
  let nn = nnSA nnType
  repMem <- mkReplayMemory (nnConfig ^. replayMemoryMaxSize)
  return $
    checkGrenade net nrNets nnConfig $
    BORL
      (zip [idxStart ..] as)
      asFilter
      initialState
      ftExt
      0
      (0, 0)
      params
      decayFun
      []
      (convertAlgorithm ftExt alg)
      Maximise
      mempty
      mempty
      (0, 0, 0)
      (ProxiesCombinedUnichain (Scalar defRhoMin) (Scalar defRho) nn repMem)
  where
    defRhoMin = defaultRhoMinimum (fromMaybe defInitValues initValues)
    defRho = defaultRho (fromMaybe defInitValues initValues)


mkMultichainGrenade ::
     forall nrH nrL s layers shapes.
     ( GNum (Gradients layers)
     , KnownNat nrH
     , Head shapes ~ 'D1 nrH
     , KnownNat nrL
     , Last shapes ~ 'D1 nrL
     , Ord s
     , NFData (Tapes layers shapes)
     , NFData (Network layers shapes)
     , Serialize (Network layers shapes)
     )
  => Algorithm s
  -> InitialState s
  -> FeatureExtractor s
  -> [Action s]
  -> (s -> [Bool])
  -> ParameterInitValues
  -> Decay
  -> Network layers shapes
  -> NNConfig
  -> IO (BORL s)
mkMultichainGrenade alg initialState ftExt as asFilter params decayFun net nnConfig = do
  let nnSA tp = Grenade net net mempty tp nnConfig (length as)
  let nnSAMinRhoTable = nnSA VTable
  let nnSARhoTable = nnSA VTable
  let nnSAVTable = nnSA VTable
  let nnSAWTable = nnSA WTable
  let nnSAR0Table = nnSA R0Table
  let nnSAR1Table = nnSA R1Table
  let nnPsiV = nnSA PsiVTable
  let nnPsiW = nnSA PsiWTable
  repMem <- mkReplayMemory (nnConfig ^. replayMemoryMaxSize)
  return $
    checkGrenade net 1 nnConfig $
    BORL
      (zip [0 ..] as)
      asFilter
      initialState
      ftExt
      0
      (0, 0)
      params
      decayFun
      []
      (convertAlgorithm ftExt alg)
      Maximise
      mempty
      mempty
      (0, 0, 0)
      (Proxies nnSAMinRhoTable nnSARhoTable nnPsiV nnSAVTable nnPsiW nnSAWTable nnSAR0Table nnSAR1Table repMem)


mkReplayMemory :: Int -> IO (Maybe ReplayMemory)
mkReplayMemory sz | sz <= 1 = return Nothing
mkReplayMemory sz = do
  vec <- V.new sz
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
    AlgDQNAvgRewAdjusted _ ga0 _ _ -> ScalingNetOutParameters (-maxR1) maxR1 (-maxW) maxW (-maxR0) maxR0 (-maxR1) maxR1
      where maxR0 = ga0 * maxR
    _ -> scalingByMaxAbsReward onlyPositive maxR
  where
    maxW = 50 * maxR
    maxR1 = 1.0 * maxR


-------------------- Helpers --------------------

-- | Checks the neural network setup and throws an error in case of a faulty number of input or output nodes.
checkGrenade ::
     forall layers shapes nrH nrL s. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, Ord s)
  => Network layers shapes
  -> Integer
  -> NNConfig
  -> BORL s
  -> BORL s
checkGrenade _ mult nnConfig borl
  | nnInpNodes /= stInp = error $ "Number of input nodes for neural network is " ++ show nnInpNodes ++ " but should be " ++ show stInp
  | nnOutNodes /= mult * fromIntegral nrActs = error $ "Number of output nodes for neural network is " ++ show nnOutNodes ++ " but should be " ++ show (fromIntegral mult * nrActs)
  | otherwise = borl
  where
    nnInpNodes = fromIntegral $ natVal (Type.Proxy :: Type.Proxy nrH)
    nnOutNodes = natVal (Type.Proxy :: Type.Proxy nrL)
    stInp = length ((borl ^. featureExtractor) (borl ^. s))
    nrActs = length (borl ^. actionList)

-- | Perform an action over all proxies (combined proxies are seen once only).
overAllProxies :: ((a -> Identity b) -> Proxy -> Identity Proxy) -> (a -> b) -> BORL s -> BORL s
overAllProxies l f borl = foldl' (\b p -> over (proxies . p . l) f b) borl (allProxiesLenses (borl ^. proxies))

setAllProxies :: ((a -> Identity b) -> Proxy -> Identity Proxy) -> b -> BORL s -> BORL s
setAllProxies l = overAllProxies l . const

allProxies :: Proxies -> [Proxy]
allProxies pxs = map (pxs ^. ) (allProxiesLenses pxs)


