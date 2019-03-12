{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TupleSections       #-}
{-# LANGUAGE TypeFamilies        #-}

module ML.BORL.Type where

import           ML.BORL.Action
import           ML.BORL.Algorithm
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import           ML.BORL.Proxy.Type
import           ML.BORL.Types

import           Control.DeepSeq
import           Control.Lens
import           Control.Monad                (zipWithM)
import           Control.Monad.IO.Class       (MonadIO, liftIO)
import qualified Data.Map.Strict              as M
import           Data.Maybe                   (fromMaybe)
import qualified Data.Proxy                   as Type
import           Data.Singletons.Prelude.List
import qualified Data.Vector.Mutable          as V
import           GHC.Generics
import           GHC.TypeLits
import           Grenade
import qualified TensorFlow.Core              as TF
import qualified TensorFlow.Session           as TF

type ActionIndexed s = (ActionIndex, Action s)                        -- ^ An action with index.
type Decay = Period -> Parameters -> Parameters -- ^ Function specifying the decay of the
                                                                      -- parameters at time t.


-------------------- Main RL Datatype --------------------


data BORL s = BORL
  { _actionList     :: ![ActionIndexed s]    -- ^ List of possible actions in state s.
  , _actionFilter   :: !(s -> [Bool])        -- ^ Function to filter actions in state s.
  , _s              :: !s                    -- ^ Current state.
  , _t              :: !Integer              -- ^ Current time t.
  , _episodeNrStart :: !(Integer, Integer)   -- ^ Nr of Episode and start period.
  , _parameters     :: !Parameters           -- ^ Parameter setup.
  , _decayFunction  :: !Decay                -- ^ Decay function at period t.

  -- define algorithm to use
  , _algorithm      :: !Algorithm

  -- Values:
  , _lastVValues    :: ![Double] -- ^ List of X last V values
  , _lastRewards    :: ![Double] -- ^ List of X last rewards
  , _psis           :: !(Double, Double, Double)  -- ^ Exponentially smoothed psi values.
  , _proxies        :: Proxies s                  -- ^ Scalar, Tables and Neural Networks

#ifdef DEBUG
  -- Stats:
  , _visits         :: !(M.Map s Integer) -- ^ Counts the visits of the states
#endif
  }
makeLenses ''BORL

instance NFData s => NFData (BORL s) where
  rnf (BORL as af s t epNr par dec alg lastVs lastRews psis proxies
#ifdef DEBUG
       vis
#endif
      ) =
    rnf as `seq` rnf af `seq` rnf s `seq` rnf t `seq` rnf epNr `seq` rnf par `seq` rnf dec `seq` rnf alg `seq` rnf lastVs `seq` rnf lastRews `seq` rnf proxies `seq` rnf psis `seq` rnf s
#ifdef DEBUG
       `seq` rnf vis
#endif


idxStart :: Int
idxStart = 0


-------------------- Constructors --------------------

-- Tabular representations

mkUnichainTabular :: (Ord s) => Algorithm -> InitialState s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> Maybe Double -> BORL s
mkUnichainTabular alg initialState as asFilter params decayFun mRhoInit =
  BORL
    (zip [idxStart ..] as)
    asFilter
    initialState
    0
    (0, 0)
    params
    decayFun
    alg
    mempty
    mempty
    (0, 0, 0)
    (Proxies (Scalar $ fromMaybe 0 mRhoInit) (Scalar $ fromMaybe 0 mRhoInit) tabSA tabSA tabSA tabSA tabSA Nothing)
#ifdef DEBUG
    mempty
#endif
  where
    tabSA = Table mempty 0

mkUnichainTensorflow :: forall s m . (NFData s, Ord s) => Algorithm -> InitialState s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> TF.Session TensorflowModel -> NNConfig s -> Maybe Double -> IO (BORL s)
mkUnichainTensorflow alg initialState as asFilter params decayFun modelBuilder nnConfig mInitRho
  -- Initialization for all NNs
 = do
  let nnTypes = [VTable, VTable, WTable, WTable, R0Table, R0Table, R1Table, R1Table, PsiVTable, PsiVTable]
      scopes = concat $ repeat ["_target", "_worker"]
  let fullModelInit = sequenceA (zipWith3 (\tp sc fun -> TF.withNameScope (name tp <> sc) fun) nnTypes scopes (repeat modelBuilder))
  let netInpInitState = (nnConfig ^. toNetInp) initialState
      nnSA :: ProxyType -> Int -> IO (Proxy s)
      nnSA tp idx = do
        nnT <- runMonadBorl $ mkModel tp "_target" netInpInitState ((!! idx) <$> fullModelInit)
        nnW <- runMonadBorl $ mkModel tp "_worker" netInpInitState ((!! (idx + 1)) <$> fullModelInit)
        return $ TensorflowProxy nnT nnW mempty tp nnConfig (length as)
  v <- nnSA VTable 0
  w <- nnSA WTable 2
  r0 <- nnSA R0Table 4
  r1 <- nnSA R1Table 6
  psiV <- nnSA PsiVTable 8
  repMem <- mkReplayMemory (nnConfig ^. replayMemoryMaxSize)
  return $
    force $
    BORL
      (zip [idxStart ..] as)
      asFilter
      initialState
      0
      (0, 0)
      params
      decayFun
      alg
      mempty
      mempty
      (0, 0, 0)
      (Proxies (Scalar $ fromMaybe 0 mInitRho) (Scalar $ fromMaybe 0 mInitRho) psiV v w r0 r1 (Just repMem))
#ifdef DEBUG
    mempty
#endif
  where
    mkModel tp scope netInpInitState modelBuilderFun = do
      !model <- prependName (name tp <> scope) <$> Tensorflow modelBuilderFun
      saveModel
        (TensorflowModel' model Nothing (Just (map realToFrac netInpInitState, replicate (length as) 0)) modelBuilderFun)
        [map realToFrac netInpInitState]
        [replicate (length as) 0]
    prependName txt model =
      model {inputLayerName = txt <> "/" <> inputLayerName model, outputLayerName = txt <> "/" <> outputLayerName model, labelLayerName = txt <> "/" <> labelLayerName model}
    name VTable    = "v"
    name WTable    = "w"
    name R0Table   = "r0"
    name R1Table   = "r1"
    name PsiVTable = "psiV"


mkMultichainTabular :: (Ord s) => Algorithm -> InitialState s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> Maybe Double -> BORL s
mkMultichainTabular alg initialState as asFilter params decayFun mRhoInit =
  BORL
    (zip [0 ..] as)
    asFilter
    initialState
    0
    (0, 0)
    params
    decayFun
    alg
    mempty
    mempty
    (0, 0, 0)
    (Proxies tabSARho tabSARho tabSA tabSA tabSA tabSA tabSA Nothing)
#ifdef DEBUG
    mempty
#endif
  where
    tabSA = Table mempty 0
    tabSARho = Table mempty (fromMaybe 0 mRhoInit)

-- Neural network approximations

mkUnichainGrenade ::
     forall nrH nrL s layers shapes. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, Ord s, NFData (Tapes layers shapes), NFData (Network layers shapes))
  => Algorithm
  -> InitialState s
  -> [Action s]
  -> (s -> [Bool])
  -> Parameters
  -> Decay
  -> Network layers shapes
  -> NNConfig s
  -> IO (BORL s)
mkUnichainGrenade alg initialState as asFilter params decayFun net nnConfig = do
  let nnSA tp = Grenade net net mempty tp nnConfig (length as)
  let nnSAVTable = nnSA VTable
  let nnSAWTable = nnSA WTable
  let nnSAR0Table = nnSA R0Table
  let nnSAR1Table = nnSA R1Table
  let nnPsiV = nnSA PsiVTable
  repMem <- mkReplayMemory (nnConfig ^. replayMemoryMaxSize)
  return $
    checkGrenade net nnConfig $
    BORL
      (zip [idxStart ..] as)
      asFilter
      initialState
      0
      (0, 0)
      params
      decayFun
      alg
      mempty
      mempty
      (0, 0, 0)
      (Proxies (Scalar 0) (Scalar 0) nnPsiV nnSAVTable nnSAWTable nnSAR0Table nnSAR1Table (Just repMem))
#ifdef DEBUG
    mempty
#endif


mkMultichainGrenade ::
     forall nrH nrL s layers shapes. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, Ord s, NFData (Tapes layers shapes), NFData (Network layers shapes))
  => Algorithm
  -> InitialState s
  -> [Action s]
  -> (s -> [Bool])
  -> Parameters
  -> Decay
  -> Network layers shapes
  -> NNConfig s
  -> IO (BORL s)
mkMultichainGrenade alg initialState as asFilter params decayFun net nnConfig = do
  let nnSA tp = Grenade net net mempty tp nnConfig (length as)
  let nnSAMinRhoTable = nnSA VTable
  let nnSARhoTable = nnSA VTable
  let nnSAVTable = nnSA VTable
  let nnSAWTable = nnSA WTable
  let nnSAR0Table = nnSA R0Table
  let nnSAR1Table = nnSA R1Table
  let nnPsiV = nnSA PsiVTable
  repMem <- mkReplayMemory (nnConfig ^. replayMemoryMaxSize)
  return $
    checkGrenade net nnConfig $
    BORL
      (zip [0 ..] as)
      asFilter
      initialState
      0
      (0, 0)
      params
      decayFun
      alg
      mempty
      mempty
      (0, 0, 0)
      (Proxies nnSAMinRhoTable nnSARhoTable nnPsiV nnSAVTable nnSAWTable nnSAR0Table nnSAR1Table (Just repMem))
#ifdef DEBUG
    mempty
#endif


mkReplayMemory :: Int -> IO (ReplayMemory s)
mkReplayMemory sz = do
  vec <- V.new sz
  return $ ReplayMemory vec sz


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
        maxR1 = 0.8 * maxDiscount defaultGamma1


-------------------- Helpers --------------------

-- | Checks the neural network setup and throws an error in case of a faulty number of input or output nodes.
checkGrenade ::
     forall layers shapes nrH nrL s. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, Ord s)
  => Network layers shapes
  -> NNConfig s
  -> BORL s
  -> BORL s
checkGrenade _ nnConfig borl
  | nnInpNodes /= stInp = error $ "Number of input nodes for neural network is " ++ show nnInpNodes ++ " but should be " ++ show stInp
  | nnOutNodes /= fromIntegral nrActs = error $ "Number of output nodes for neural network is " ++ show nnOutNodes ++ " but should be " ++ show nrActs
  | otherwise = borl
  where
    nnInpNodes = fromIntegral $ natVal (Type.Proxy :: Type.Proxy nrH)
    nnOutNodes = natVal (Type.Proxy :: Type.Proxy nrL)
    stInp = length ((nnConfig ^. toNetInp) (borl ^. s))
    nrActs = length (borl ^. actionList)

