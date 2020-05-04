{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE Rank2Types          #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Unsafe              #-}


module ML.BORL.Serialisable where

import           Control.Arrow                (first)
import           Control.DeepSeq
import           Control.Lens
import           Control.Monad                (zipWithM_)
import           Data.Constraint              (Dict (..))
import           Data.List                    (foldl')
import qualified Data.Map                     as M
import           Data.Serialize
import           Data.Singletons.Prelude.List
import           Data.Typeable                (Typeable)
import qualified Data.Vector                  as VB
import qualified Data.Vector.Mutable          as VM
import qualified Data.Vector.Storable         as V
import           GHC.Generics
import           GHC.TypeLits
import           Grenade
import qualified HighLevelTensorflow          as TF
import           System.IO.Unsafe
import           Unsafe.Coerce                (unsafeCoerce)

import           ML.BORL.Action.Type
import           ML.BORL.Algorithm
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import           ML.BORL.Proxy.Proxies
import           ML.BORL.Proxy.Type
import           ML.BORL.Reward.Type
import           ML.BORL.SaveRestore
import           ML.BORL.Settings
import           ML.BORL.Type
import           ML.BORL.Types
import           ML.BORL.Workers.Type


data BORLSerialisable s = BORLSerialisable
  { serS                 :: !s                    -- ^ Current state.
  , serWorkers           :: !(Workers s)          -- ^ Workers
  , serT                 :: !Int                  -- ^ Current time t.
  , serEpisodeNrStart    :: !(Int, Int)           -- ^ Nr of Episode and start period.
  , serParameters        :: !ParameterInitValues  -- ^ Parameter setup.
  , serParameterSetting  :: !ParameterDecaySetting
  , serSettings          :: !Settings  -- ^ Parameter setup.
  , serRewardFutures     :: [RewardFutureData s]

  -- define algorithm to use
  , serAlgorithm         :: !(Algorithm [Float])
  , serObjective         :: !Objective

  -- Values:
  , serExpSmoothedReward :: Float                  -- ^ Exponentially smoothed reward
  , serLastVValues       :: ![Float]               -- ^ List of X last V values
  , serLastRewards       :: ![Float]               -- ^ List of X last rewards
  , serPsis              :: !(Float, Float, Float) -- ^ Exponentially smoothed psi values.
  , serProxies           :: Proxies                -- ^ Scalar, Tables and Neural Networks
  } deriving (Generic, Serialize)

toSerialisable :: (MonadBorl' m, RewardFuture s) => BORL s -> m (BORLSerialisable s)
toSerialisable = toSerialisableWith id id


toSerialisableWith :: (MonadBorl' m, RewardFuture s') => (s -> s') -> (StoreType s -> StoreType s') -> BORL s -> m (BORLSerialisable s')
toSerialisableWith f g borl = do
  BORL _ _ state workers' _ time eNr par dec setts future alg obj expSmthRew v rew psis prS <- saveTensorflowModels borl
  return $ BORLSerialisable (f state) (mapWorkers f g workers') time eNr par dec setts (map (mapRewardFutureData f g) future) (mapAlgorithmState V.toList alg) obj expSmthRew v rew psis prS

fromSerialisable :: (MonadBorl' m, RewardFuture s) => [Action s] -> ActionFilter s -> FeatureExtractor s -> TF.ModelBuilderFunction -> BORLSerialisable s -> m (BORL s)
fromSerialisable = fromSerialisableWith id id

fromSerialisableWith ::
     (MonadBorl' m, RewardFuture s)
  => (s' -> s)
  -> (StoreType s' -> StoreType s)
  -> [Action s]
  -> ActionFilter s
  -> FeatureExtractor s
  -> TF.ModelBuilderFunction
  -> BORLSerialisable s'
  -> m (BORL s)
fromSerialisableWith f g as aF ftExt builder (BORLSerialisable st workers' t e par dec setts future alg obj expSmthRew lastV rew psis prS) = do
  let aL = zip [idxStart ..] as
      borl = BORL (VB.fromList aL) aF (f st) (mapWorkers f g workers') ftExt t e par dec setts (map (mapRewardFutureData f g) future) (mapAlgorithmState V.fromList alg) obj expSmthRew lastV rew psis prS
      pxs = borl ^. proxies
      nrOutCols | isCombinedProxies pxs && isAlgDqn alg = 1
                | isCombinedProxies pxs && isAlgDqnAvgRewardAdjusted alg = 2
                | isCombinedProxies pxs = 6
                | otherwise = 1
      borl' =
        flip (foldl' (\b p -> over (proxies . p . proxyTFWorker) (\x -> x {tensorflowModelBuilder = builder nrOutCols}) b)) (allProxiesLenses pxs) $
        flip (foldl' (\b p -> over (proxies . p . proxyTFTarget) (\x -> x {tensorflowModelBuilder = builder nrOutCols}) b)) (allProxiesLenses pxs) borl
  liftTf $ restoreTensorflowModels False borl'
  return borl'


instance Serialize Proxies
instance Serialize ReplayMemories where
  put (ReplayMemoriesUnified r)         = put (0 :: Int) >> put r
  put (ReplayMemoriesPerActions tmp xs) = put (1 :: Int) >> put tmp >> put (VB.toList xs)
  get = do
    nr <- get
    case (nr :: Int) of
      0 -> ReplayMemoriesUnified <$> get
      1 -> ReplayMemoriesPerActions <$> get <*> fmap VB.fromList get
      _ -> error "index error"

instance (Serialize s, RewardFuture s) => Serialize (WorkerState s)

instance Serialize NNConfig where
  put (NNConfig memSz memStrat batchSz trainIter opt smooth decaySetup prS scale stab stabDec upInt upIntDec) =
    case opt of
      o@OptSGD{} -> put memSz >> put memStrat >> put batchSz >> put trainIter >> put o >> put smooth >> put decaySetup >> put (map V.toList prS) >> put scale >> put stab >> put stabDec >> put upInt >> put upIntDec
      o@OptAdam{} -> put memSz >> put memStrat >> put batchSz >> put trainIter >> put o >> put smooth >> put decaySetup >> put (map V.toList prS) >> put scale >> put stab >> put stabDec >> put upInt >> put upIntDec
    -- put memSz >> put memStrat >> put batchSz >> put opt >> put decaySetup >> put (map V.toList prS) >> put scale >> put stab >> put stabDec >> put upInt >> put upIntDec
  get = do
    memSz <- get
    memStrat <- get
    batchSz <- get
    trainIter <- get
    opt <- get
    smooth <- get
    decaySetup <- get
    prS <- map V.fromList <$> get
    scale <- get
    stab <- get
    stabDec <- get
    upInt <- get
    upIntDec <- get
    return $ NNConfig memSz memStrat batchSz trainIter opt smooth decaySetup prS scale stab stabDec upInt upIntDec


instance Serialize Proxy where
  put (Scalar x) = put (0 :: Int) >> put x
  put (Table m d) = put (1 :: Int) >> put (M.mapKeys (first V.toList) m) >> put d
  put (Grenade t w tp conf nr) = put (2 :: Int) >> put (networkToSpecification t) >> put t >> put w >> put tp >> put conf >> put nr
  put (TensorflowProxy t w tp conf nr) = put (3 :: Int) >> put t >> put w >> put tp >> put conf >> put nr
  get = do
    (c :: Int) <- get
    case c of
      0 -> get >>= return . Scalar
      1 -> do
        m <- M.mapKeys (first V.fromList) <$> get
        d <- get
        return $ Table m d
      2 -> do
        (specT :: SpecNet) <- get
        case unsafePerformIO (networkFromSpecificationGenericWith UniformInit specT) of
          SpecNetwork (netT :: Network tLayers tShapes) -> do
            case (unsafeCoerce (Dict :: Dict ()) :: Dict (Head tShapes ~ 'D1 th, KnownNat th, Typeable tLayers, Typeable tShapes)) of
              Dict -> do
                (t :: Network tLayers tShapes) <- get
                (w :: Network tLayers tShapes) <- get
                tp <- get
                conf <- get
                nr <- get
                return $ Grenade t w tp conf nr
      3 -> do
        t <- get
        w <- get
        tp <- get
        conf <- get
        nr <- get
        return $ TensorflowProxy t w tp conf nr
      _ -> error "Unknown constructor for proxy"

-- ^ Replay Memory
instance Serialize ReplayMemory where
  put (ReplayMemory vec sz idx maxIdx) = do
    let xs = unsafePerformIO $ mapM (VM.read vec) [0 .. maxIdx]
    put sz
    put idx
    put $ map (\((st,as), a, rand, rew, (st',as'), epsEnd) -> ((V.toList st, V.toList as), a, rand, rew, (V.toList st', V.toList as'), epsEnd)) xs
    put maxIdx
  get = do
    sz <- get
    idx <- get
    (xs :: [Experience]) <- map (\((st,as), a, rand, rew, (st',as'), epsEnd) -> ((V.fromList st, V.fromList as), a, rand, rew, (V.fromList st', V.fromList as'), epsEnd)) <$> get
    maxIdx <- get
    return $
      unsafePerformIO $ do
        vec <- VM.new sz
        vec `seq` zipWithM_ (VM.write vec) [0 .. maxIdx] xs
        return (ReplayMemory vec sz idx maxIdx)
