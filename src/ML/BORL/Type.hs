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

module ML.BORL.Type where

import           ML.BORL.Action
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

data Phase = IncreasingStateValues
           --  | DecreasingStateValues
           | SteadyStateValues
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
  , _parameters       :: !Parameters           -- ^ Parameter setup.
  , _decayFunction    :: !Decay                -- ^ Decay function at period t.
  , _futureRewards    :: ![RewardFutureData s] -- ^ List of future reward.

  -- define algorithm to use
  , _algorithm        :: !(Algorithm s) -- ^ What algorithm to use.
  , _phase            :: !Phase         -- ^ Current phase for scaling by `StateValueHandling`.

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


------------------------------ Initial Values ------------------------------


idxStart :: Int
idxStart = 0


data InitValues = InitValues
  { defaultRho :: Double
  , defaultV   :: Double
  , defaultW   :: Double
  , defaultR0  :: Double
  , defaultR1  :: Double
  }


defInitValues :: InitValues
defInitValues = InitValues 0 0 0 0 0

-------------------- Constructors --------------------

-- Tabular representations

mkUnichainTabular :: Algorithm s -> InitialState s -> FeatureExtractor s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> Maybe InitValues -> BORL s
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
    alg
    SteadyStateValues
    mempty
    mempty
    (0, 0, 0)
    (Proxies (Scalar 0) (Scalar defRho) (tabSA 0) (tabSA defV) (tabSA 0) (tabSA defW) (tabSA 0) (tabSA defW) (tabSA defR0) (tabSA defR1) Nothing)
  where
    tabSA def = Table mempty def
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
  -> Parameters
  -> Decay
  -> TF.Session TensorflowModel
  -> NNConfig
  -> Maybe InitValues
  -> m (BORL s)
mkUnichainTensorflowM alg initialState ftExt as asFilter params decayFun modelBuilder nnConfig initValues = do
  let nnTypes = [VTable, VTable, WTable, WTable, W2Table, W2Table, R0Table, R0Table, R1Table, R1Table, PsiVTable, PsiVTable, PsiWTable, PsiWTable, PsiW2Table, PsiW2Table]
      scopes = concat $ repeat ["_target", "_worker"]
  let fullModelInit = sequenceA (zipWith3 (\tp sc fun -> TF.withNameScope (proxyTypeName tp <> sc) fun) nnTypes scopes (repeat modelBuilder))
  let netInpInitState = ftExt initialState
      nnSA :: ProxyType -> Int -> IO Proxy
      nnSA tp idx = do
        nnT <- runMonadBorlTF $ mkTensorflowModel as tp "_target" netInpInitState ((!! idx) <$> fullModelInit)
        nnW <- runMonadBorlTF $ mkTensorflowModel as tp "_worker" netInpInitState ((!! (idx + 1)) <$> fullModelInit)
        return $ TensorflowProxy nnT nnW mempty tp nnConfig (length as)
  v <- liftIO $ nnSA VTable 0
  w <- liftIO $ nnSA WTable 2
  w2 <- liftIO $ nnSA W2Table 4
  r0 <- liftIO $ nnSA R0Table 6
  r1 <- liftIO $ nnSA R1Table 8
  psiV <- liftIO $ nnSA PsiVTable 10
  psiW <- liftIO $ nnSA PsiWTable 12
  psiW2 <- liftIO $ nnSA PsiW2Table 14
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
      alg
      SteadyStateValues
      mempty
      mempty
      (0, 0, 0)
      (Proxies (Scalar 0) (Scalar defRho) psiV v psiW w psiW2 w2 r0 r1 repMem)
  where
    defRho = defaultRho (fromMaybe defInitValues initValues)

-- ^ The output tensor must be 2D with the number of rows corresponding to the number of actions and there must be 8
-- columns
mkUnichainTensorflowCombinedNetM ::
     forall s m. (NFData s, MonadBorl' m)
  => Algorithm s
  -> InitialState s
  -> FeatureExtractor s
  -> [Action s]
  -> (s -> [Bool])
  -> Parameters
  -> Decay
  -> ModelBuilderFunction
  -> NNConfig
  -> Maybe InitValues
  -> m (BORL s)
mkUnichainTensorflowCombinedNetM alg initialState ftExt as asFilter params decayFun modelBuilder nnConfig initValues = do
  let nrNets | isAlgDqn alg = 1
             | isAlgDqnAvgRewardFree alg = 2
             | otherwise = 8
  let nnTypes = [CombinedUnichain, CombinedUnichain]
      scopes = concat $ repeat ["_target", "_worker"]
  let fullModelInit = sequenceA (zipWith3 (\tp sc fun -> TF.withNameScope (proxyTypeName tp <> sc) fun) nnTypes scopes (repeat (modelBuilder nrNets)))
  let netInpInitState = ftExt initialState
      nnSA :: ProxyType -> Int -> IO Proxy
      nnSA tp idx = do
        nnT <- runMonadBorlTF $ mkTensorflowModel (concat $ replicate (fromIntegral nrNets) as) tp "_target" netInpInitState ((!! idx) <$> fullModelInit)
        nnW <- runMonadBorlTF $ mkTensorflowModel (concat $ replicate (fromIntegral nrNets) as) tp "_worker" netInpInitState ((!! (idx + 1)) <$> fullModelInit)
        return $ TensorflowProxy nnT nnW mempty tp nnConfig (length as)
  proxy <- liftIO $ nnSA CombinedUnichain 0
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
      alg
      SteadyStateValues
      mempty
      mempty
      (0, 0, 0)
      (ProxiesCombinedUnichain (Scalar 0) (Scalar defRho) proxy repMem)
  where
    defRho = defaultRho (fromMaybe defInitValues initValues)


-- ^ Uses a single network for each value to learn. Thus the output tensor must be 1D with the number of rows
-- corresponding to the number of actions.
mkUnichainTensorflow ::
     forall s . (NFData s)
  => Algorithm s
  -> InitialState s
  -> FeatureExtractor s
  -> [Action s]
  -> (s -> [Bool])
  -> Parameters
  -> Decay
  -> TF.Session TensorflowModel
  -> NNConfig
  -> Maybe InitValues
  -> IO (BORL s)
mkUnichainTensorflow alg initialState ftExt as asFilter params decayFun modelBuilder nnConfig initValues =
  runMonadBorlTF (mkUnichainTensorflowM alg initialState ftExt as asFilter params decayFun modelBuilder nnConfig initValues)

-- ^ Use a single network for all function approximations. Thus, the output tensor must be 2D with the number of rows
-- corresponding to the number of actions and there must be 8 columns.
mkUnichainTensorflowCombinedNet ::
     forall s . (NFData s)
  => Algorithm s
  -> InitialState s
  -> FeatureExtractor s
  -> [Action s]
  -> (s -> [Bool])
  -> Parameters
  -> Decay
  -> ModelBuilderFunction
  -> NNConfig
  -> Maybe InitValues
  -> IO (BORL s)
mkUnichainTensorflowCombinedNet alg initialState ftExt as asFilter params decayFun modelBuilder nnConfig initValues =
  runMonadBorlTF (mkUnichainTensorflowCombinedNetM alg initialState ftExt as asFilter params decayFun modelBuilder nnConfig initValues)


mkMultichainTabular :: Algorithm s -> InitialState s -> FeatureExtractor s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> Maybe InitValues -> BORL s
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
    alg
    SteadyStateValues
    mempty
    mempty
    (0, 0, 0)
    (Proxies (tabSA 0) (tabSA defRho) (tabSA 0) (tabSA defV) (tabSA 0) (tabSA defW) (tabSA 0) (tabSA defW) (tabSA defR0) (tabSA defR1) Nothing)
  where
    tabSA def = Table mempty def
    defRho = defaultRho (fromMaybe defInitValues initValues)
    defV = defaultV (fromMaybe defInitValues initValues)
    defW = defaultW (fromMaybe defInitValues initValues)
    defW2 = defaultW (fromMaybe defInitValues initValues)
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
  -> Parameters
  -> Decay
  -> Network layers shapes
  -> NNConfig
  -> Maybe InitValues
  -> IO (BORL s)
mkUnichainGrenade alg initialState ftExt as asFilter params decayFun net nnConfig initValues = do
  let nnSA tp = Grenade net net mempty tp nnConfig (length as)
  let nnSAVTable = nnSA VTable
  let nnSAWTable = nnSA WTable
  let nnSAW2Table = nnSA W2Table
  let nnSAR0Table = nnSA R0Table
  let nnSAR1Table = nnSA R1Table
  let nnPsiV = nnSA PsiVTable
  let nnPsiW = nnSA PsiWTable
  let nnPsiW2 = nnSA PsiW2Table
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
      alg
      SteadyStateValues
      mempty
      mempty
      (0, 0, 0)
      (Proxies (Scalar 0) (Scalar defRho) nnPsiV nnSAVTable nnPsiW nnSAWTable nnPsiW2 nnSAW2Table nnSAR0Table nnSAR1Table repMem)
  where
    defRho = defaultRho (fromMaybe defInitValues initValues)

mkUnichainGrenadeCombinedNet ::
     forall nrH nrL s layers shapes. (GNum (Gradients layers), KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, Ord s, NFData (Tapes layers shapes), NFData (Network layers shapes), Serialize (Network layers shapes))
  => Algorithm s
  -> InitialState s
  -> FeatureExtractor s
  -> [Action s]
  -> (s -> [Bool])
  -> Parameters
  -> Decay
  -> Network layers shapes
  -> NNConfig
  -> Maybe InitValues
  -> IO (BORL s)
mkUnichainGrenadeCombinedNet alg initialState ftExt as asFilter params decayFun net nnConfig initValues = do
  let nrNets | isAlgDqn alg = 1
             | isAlgDqnAvgRewardFree alg = 2
             | otherwise = 8
  let nnSA tp = Grenade net net mempty tp nnConfig (length as)
  let nn = nnSA CombinedUnichain
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
      alg
      SteadyStateValues
      mempty
      mempty
      (0, 0, 0)
      (ProxiesCombinedUnichain (Scalar 0) (Scalar defRho) nn repMem)
  where
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
  -> Parameters
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
  let nnSAW2Table = nnSA W2Table
  let nnSAR0Table = nnSA R0Table
  let nnSAR1Table = nnSA R1Table
  let nnPsiV = nnSA PsiVTable
  let nnPsiW = nnSA PsiWTable
  let nnPsiW2 = nnSA PsiW2Table
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
      alg
      SteadyStateValues
      mempty
      mempty
      (0, 0, 0)
      (Proxies nnSAMinRhoTable nnSARhoTable nnPsiV nnSAVTable nnPsiW nnSAWTable nnPsiW2 nnSAW2Table nnSAR0Table nnSAR1Table repMem)


mkReplayMemory :: Int -> IO (Maybe ReplayMemory)
mkReplayMemory sz | sz <= 0 = return Nothing
mkReplayMemory sz = do
  vec <- V.new sz
  return $ Just $ ReplayMemory vec sz (-1)


-------------------- Other Constructors --------------------

-- noScaling :: ScalingNetOutParameters
-- noScaling = ScalingNetOutParameters

-- | Infer scaling by maximum reward.
scalingByMaxAbsReward :: Bool -> Double -> ScalingNetOutParameters
scalingByMaxAbsReward onlyPositive maxR = ScalingNetOutParameters (-maxV) maxV (-maxW) maxW (if onlyPositive then 0 else -maxR0) maxR0 (if onlyPositive then 0 else -maxR1) maxR1
  where maxDiscount g = sum $ take 10000 $ map (\p -> (g^p) * maxR) [(0::Int)..]
        maxV = 1.0 * maxR
        maxW = 150 * maxR
        maxR0 = 2 * maxDiscount defaultGamma0
        maxR1 = 1.0 * maxDiscount defaultGamma1


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


overAllProxies :: ((a -> Identity b) -> Proxy -> Identity Proxy) -> (a -> b) -> BORL s -> BORL s
overAllProxies l f borl
  -- | isCombinedProxy (borl ^. proxies . r0) = foldl' (\b p -> over (proxies . p . l) f b) borl [rhoMinimum, rho, r0]
  | otherwise = foldl' (\b p -> over (proxies . p . l) f b) borl [rhoMinimum, rho, psiV, v, psiW, w, psiW2, w, r0, r1]

setAllProxies :: ((a -> Identity b) -> Proxy -> Identity Proxy) -> b -> BORL s -> BORL s
setAllProxies l = overAllProxies l . const

allProxies :: Proxies -> [Proxy]
allProxies pxs@Proxies{} = [pxs ^. rhoMinimum, pxs ^. rho, pxs ^?! psiV, pxs ^?! v, pxs ^?! psiW , pxs ^?! w, pxs^?!psiW2, pxs ^?! w2, pxs ^?! r0, pxs ^?! r1]
allProxies pxs@ProxiesCombinedUnichain{} = [pxs ^. rhoMinimum, pxs ^. rho, _proxy pxs]


