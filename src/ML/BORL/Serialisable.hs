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
import           Control.Concurrent.MVar
import           Control.DeepSeq
import           Control.Lens
import           Control.Monad                (zipWithM_)
import           Control.Monad.IO.Class
import           Data.Constraint              (Dict (..))
import           Data.List                    (foldl')
import qualified Data.Map                     as M
import           Data.Serialize
import           Data.Singletons              (sing, withSingI)
import           Data.Singletons.Prelude.List
import           Data.Typeable                (Typeable)
import qualified Data.Vector                  as VB
import qualified Data.Vector.Mutable          as VM
import qualified Data.Vector.Storable         as V
import           GHC.Generics
import           GHC.TypeLits
import           Grenade
import           System.IO.Unsafe
import           Unsafe.Coerce                (unsafeCoerce)

import           ML.BORL.Action.Type
import           ML.BORL.Algorithm
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import           ML.BORL.Proxy.Proxies
import           ML.BORL.Proxy.Type
import           ML.BORL.Reward.Type
import           ML.BORL.Settings
import           ML.BORL.Type
import           ML.BORL.Types
import           ML.BORL.Workers.Type

import           Debug.Trace


data BORLSerialisable s as = BORLSerialisable
  { serActionList        :: ![as]
  , serS                 :: !s                    -- ^ Current state.
  , serWorkers           :: !(Workers s)          -- ^ Workers
  , serT                 :: !Int                  -- ^ Current time t.
  , serEpisodeNrStart    :: !(Int, Int)           -- ^ Nr of Episode and start period.
  , serParameters        :: !ParameterInitValues  -- ^ Parameter setup.
  , serParameterSetting  :: !ParameterDecaySetting
  , serSettings          :: !Settings  -- ^ Parameter setup.
  , serRewardFutures     :: [RewardFutureData s]

  -- define algorithm to use
  , serAlgorithm         :: !(Algorithm [Double])
  , serObjective         :: !Objective

  -- Values:
  , serExpSmoothedReward :: Double                  -- ^ Exponentially smoothed reward
  , serLastVValues       :: ![Value]               -- ^ List of X last V values
  , serLastRewards       :: ![Double]               -- ^ List of X last rewards
  , serPsis              :: !(Value, Value, Value) -- ^ Exponentially smoothed psi values.
  , serProxies           :: Proxies                -- ^ Scalar, Tables and Neural Networks
  } deriving (Generic, Serialize)

toSerialisable :: (MonadIO m, RewardFuture s) => BORL s as -> m (BORLSerialisable s as)
toSerialisable = toSerialisableWith id id


toSerialisableWith :: (MonadIO m, RewardFuture s') => (s -> s') -> (StoreType s -> StoreType s') -> BORL s as -> m (BORLSerialisable s' as)
toSerialisableWith f g (BORL as _ _ state workers' _ time eNr par dec setts future alg obj expSmthRew v rew psis prS) =
  return $
  BORLSerialisable
    (VB.toList as)
    (f state)
    (mapWorkers f g workers')
    time
    eNr
    par
    dec
    setts
    (map (mapRewardFutureData f g) future)
    (mapAlgorithmState V.toList alg)
    obj
    expSmthRew
    (VB.toList v)
    (V.toList rew)
    psis
    prS

fromSerialisable :: (MonadIO m, RewardFuture s) => ActionFunction s as -> ActionFilter s -> FeatureExtractor s -> BORLSerialisable s as -> m (BORL s as)
fromSerialisable = fromSerialisableWith id id

fromSerialisableWith ::
     (MonadIO m, RewardFuture s)
  => (s' -> s)
  -> (StoreType s' -> StoreType s)
  -> ActionFunction s as
  -> ActionFilter s
  -> FeatureExtractor s
  -> BORLSerialisable s' as
  -> m (BORL s as)
fromSerialisableWith f g asFun aF ftExt (BORLSerialisable as st workers' t e par dec setts future alg obj expSmthRew lastV rew psis prS) =
  return $
  BORL
    (VB.fromList as)
    asFun
    aF
    (f st)
    (mapWorkers f g workers')
    ftExt
    t
    e
    par
    dec
    setts
    (map (mapRewardFutureData f g) future)
    (mapAlgorithmState V.fromList alg)
    obj
    expSmthRew
    (VB.fromList lastV)
    (V.fromList rew)
    psis
    prS


instance Serialize Proxies
instance Serialize ReplayMemories where
  put (ReplayMemoriesUnified nr r)         = put (0 :: Int) >> put nr >> put r
  put (ReplayMemoriesPerActions nrAs tmp xs) = put (1 :: Int) >> put nrAs >> put tmp >> put (VB.toList xs)
  get = do
    nr <- get
    case (nr :: Int) of
      0 -> ReplayMemoriesUnified <$> get <*> get
      1 -> ReplayMemoriesPerActions <$> get <*> get <*> fmap VB.fromList get
      _ -> error "index error"

instance (Serialize s, RewardFuture s) => Serialize (WorkerState s)

instance Serialize NNConfig where
  put (NNConfig memSz memStrat batchSz trainIter opt smooth smoothPer decaySetup prS scale scaleOutAlg crop stab stabDec clip) =
    case opt of
      o@OptSGD{} -> put memSz >> put memStrat >> put batchSz >> put trainIter >> put o >> put smooth >> put smoothPer >> put decaySetup >> put (map V.toList prS) >> put scale >>  put scaleOutAlg >> put crop >> put stab >> put stabDec >> put clip
      o@OptAdam{} -> put memSz >> put memStrat >> put batchSz >> put trainIter >> put o >> put smooth >> put smoothPer >> put decaySetup >> put (map V.toList prS) >> put scale >> put scaleOutAlg >> put crop >>  put stab >> put stabDec >> put clip
  get = do
    memSz <- get
    memStrat <- get
    batchSz <- get
    trainIter <- get
    opt <- get
    smooth <- get
    smoothPer <- get
    decaySetup <- get
    prS <- map V.fromList <$> get
    scale <- get
    scaleOutAlg <- get
    crop <- get
    stab <- get
    stabDec <- get
    clip <- get
    return $ NNConfig memSz memStrat batchSz trainIter opt smooth smoothPer decaySetup prS scale scaleOutAlg crop stab stabDec clip


instance Serialize Proxy where
  put (Scalar x nrAs) = put (0 :: Int) >> put (V.toList x) >> put nrAs
  put (Table m d acts) = put (1 :: Int) >> put (M.mapKeys (first V.toList) . M.map V.toList $ m) >> put (V.toList d) >> put acts
  put (Grenade t w tp conf nr agents) = put (2 :: Int) >> put (networkToSpecification t) >> put t >> put w >> put tp >> put conf >> put nr >> put agents
  get = do
    (c :: Int) <- get
    case c of
      0 -> Scalar <$> fmap V.fromList get <*> get
      1 -> do
        m <- M.mapKeys (first V.fromList) . M.map V.fromList <$> get
        d <- V.fromList <$> get
        Table m d <$> get
      2 -> do
        (specT :: SpecNet) <- get
        case unsafePerformIO (networkFromSpecificationWith (NetworkInitSettings UniformInit HMatrix Nothing) specT) of
          SpecConcreteNetwork1D1D (_ :: Network tLayers tShapes) -> do
            (t :: Network tLayers tShapes) <- get
            (w :: Network tLayers tShapes) <- get
            Grenade t w <$> get <*> get <*> get <*> get
          SpecConcreteNetwork1D2D (_ :: Network tLayers tShapes) -> do
            (t :: Network tLayers tShapes) <- get
            (w :: Network tLayers tShapes) <- get
            Grenade t w <$> get <*> get <*> get <*> get
          _ -> error ("Network dimensions not implemented in Serialize Proxy in ML.BORL.Serialisable")
      _ -> error "Unknown constructor for proxy"

-- ^ Replay Memory
instance Serialize ReplayMemory where
  put (ReplayMemory vec sz idx maxIdx) = do
    let xs = unsafePerformIO $ mapM (VM.read vec) [0 .. maxIdx]
    put sz
    put idx
    let mkReplMem ((st, DisallowedActionIndicies as), assel, rew, (st', DisallowedActionIndicies as'), epsEnd) =
          ((V.toList st, VB.toList $ VB.map V.toList as), assel, rew, (V.toList st', VB.toList $ VB.map V.toList as'), epsEnd)
    put $ map mkReplMem xs
    put maxIdx
  get = do
    sz <- get
    idx <- get
    let mkReplMem ((st, as), assel, rew, (st', as'), epsEnd) =
          ( (V.fromList st, DisallowedActionIndicies $ VB.fromList $ map V.fromList as)
          , assel
          , rew
          , (V.fromList st', DisallowedActionIndicies $ VB.fromList $ map V.fromList as')
          , epsEnd)
    (xs :: [Experience]) <- map mkReplMem <$> get
    maxIdx <- get
    return $
      unsafePerformIO $ do
        vec <- VM.new sz
        vec `seq` zipWithM_ (VM.write vec) [0 .. maxIdx] xs
        return (ReplayMemory vec sz idx maxIdx)
