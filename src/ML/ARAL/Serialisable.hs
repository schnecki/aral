{-# LANGUAGE BangPatterns         #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE DeriveAnyClass       #-}
{-# LANGUAGE DeriveGeneric        #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE KindSignatures       #-}
{-# LANGUAGE Rank2Types           #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE StandaloneDeriving   #-}
{-# LANGUAGE Strict               #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE Unsafe               #-}


module ML.ARAL.Serialisable where

import           Control.Applicative          ((<|>))
import           Control.Arrow                (first)
import           Control.Concurrent.MVar
import           Control.DeepSeq
import           Control.Lens
import           Control.Monad                (foldM_, zipWithM_)
import           Control.Monad.IO.Class
import qualified Data.ByteString              as BS
import           Data.Constraint              (Dict (..))
import           Data.Int
import           Data.List                    (foldl')
import qualified Data.Map                     as M
import           Data.Serialize
import           Data.Singletons              (sing, withSingI)
import           Data.Singletons.Prelude.List
import           Data.Typeable                (Typeable)
import qualified Data.Vector                  as VB
import qualified Data.Vector.Mutable          as VM
import qualified Data.Vector.Serialize
import qualified Data.Vector.Storable         as V
import           Data.Word
import           GHC.Generics
import           GHC.TypeLits
import           Grenade
import           System.IO
import           System.IO.Unsafe
import qualified Torch                        as Torch
import qualified Torch.Optim                  as Torch
import qualified Torch.Serialize              as Torch
import           Unsafe.Coerce                (unsafeCoerce)

import           ML.ARAL.Action.Type
import           ML.ARAL.Algorithm
import           ML.ARAL.NeuralNetwork
import           ML.ARAL.Parameters
import           ML.ARAL.Proxy.Proxies
import           ML.ARAL.Proxy.Type
import           ML.ARAL.Reward.Type
import           ML.ARAL.Settings
import           ML.ARAL.Type
import           ML.ARAL.Types
import           ML.ARAL.Workers.Type

import           Debug.Trace


data ARALSerialisable s as = ARALSerialisable
  { serActionList        :: ![as]
  , serS                 :: !s                    -- ^ Current state.
  , serWorkers           :: !(Workers s)          -- ^ Workers
  , serT                 :: !Int                  -- ^ Current time t.
  , serEpisodeNrStart    :: !(Int, Int)           -- ^ Nr of Episode and start period.
  , serParameters        :: !ParameterInitValues  -- ^ Parameter setup.
  , serParameterSetting  :: !ParameterDecaySetting
  , serSettings          :: !Settings  -- ^ Parameter setup.
  , serRewardFutures     :: ![RewardFutureData s]

  -- define algorithm to use
  , serAlgorithm         :: !(Algorithm [Double])
  , serObjective         :: !Objective

  -- Values:
  , serExpSmoothedReward :: !Double                  -- ^ Exponentially smoothed reward
  , serLastVValues       :: ![Value]               -- ^ List of X last V values
  , serLastRewards       :: ![Double]               -- ^ List of X last rewards
  , serPsis              :: !(Value, Value, Value) -- ^ Exponentially smoothed psi values.
  , serProxies           :: !Proxies                -- ^ Scalar, Tables and Neural Networks
  } deriving (Generic, Serialize)

toSerialisable :: (MonadIO m, RewardFuture s) => ARAL s as -> m (ARALSerialisable s as)
toSerialisable = toSerialisableWith id id


toSerialisableWith :: (MonadIO m, RewardFuture s') => (s -> s') -> (StoreType s -> StoreType s') -> ARAL s as -> m (ARALSerialisable s' as)
toSerialisableWith f g (ARAL as _ _ state workers' _ time eNr par dec setts future alg obj expSmthRew v rew psis prS) =
  return $
  ARALSerialisable
    (VB.toList as)
    (f state)
    (mapWorkers f g workers')
    time
    eNr
    par
    dec
    setts
    (VB.toList $ VB.map (mapRewardFutureData f g) future)
    (mapAlgorithmState V.toList alg)
    obj
    expSmthRew
    (VB.toList v)
    (V.toList rew)
    psis
    prS

fromSerialisable :: (MonadIO m, RewardFuture s) => ActionFunction s as -> ActionFilter s -> FeatureExtractor s -> ARALSerialisable s as -> m (ARAL s as)
fromSerialisable = fromSerialisableWith id id

fromSerialisableWith ::
     (MonadIO m, RewardFuture s)
  => (s' -> s)
  -> (StoreType s' -> StoreType s)
  -> ActionFunction s as
  -> ActionFilter s
  -> FeatureExtractor s
  -> ARALSerialisable s' as
  -> m (ARAL s as)
fromSerialisableWith f g asFun aF ftExt (ARALSerialisable as st workers' t e par dec setts future alg obj expSmthRew lastV rew psis prS) =
  return $
  ARAL
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
    (VB.map (mapRewardFutureData f g) (VB.fromList future))
    (mapAlgorithmState V.fromList alg)
    obj
    expSmthRew
    (VB.fromList lastV)
    (V.fromList rew)
    psis
    prS


instance Serialize Proxies
instance Serialize ReplayMemories where
  put (ReplayMemoriesUnified nr r)           = put (0 :: Int) >> put nr >> put r
  put (ReplayMemoriesPerActions nrAs tmp xs) = put (1 :: Int) >> put nrAs >> put tmp >> put (VB.toList xs)
  get = do
    nr <- get
    case (nr :: Int) of
      0 -> ReplayMemoriesUnified <$> get <*> get
      1 -> ReplayMemoriesPerActions <$> get <*> get <*> fmap VB.fromList get
      _ -> error "index error"

data WorkerStateSerialisable s =
  WorkerStateSerialisable
    { _serWorkerNumber        :: Int                   -- ^ Worker nr.
    , _serWorkerS             :: !s                    -- ^ Current state.
    , _serWorkerReplayMemory  :: !ReplayMemories       -- ^ Replay Memories of worker.
    , _serWorkerFutureRewards :: ![RewardFutureData s] -- ^ Future reward data.
    , _serWorkerExpSmthReward :: Double                -- ^ Exponentially smoothed reward with rate 0.0001
    }
  deriving (Generic, Serialize)


instance (Serialize s, RewardFuture s) => Serialize (WorkerState s) where
  put (WorkerState n st rep fts exp) = put (WorkerStateSerialisable n st rep (VB.toList fts) exp)
  get = (\(WorkerStateSerialisable n st rep fts exp) -> (WorkerState n st rep (VB.fromList fts) exp)) <$> get


instance Serialize NNConfig where
  put (NNConfig memSz memStrat batchSz trainIter opt smooth smoothPer decaySetup prS scale scaleOutAlg crop stab stabDec clip autoScale) =
    case opt of
      o@OptSGD{} -> put memSz >> put memStrat >> put batchSz >> put trainIter >> put o >> put smooth >> put smoothPer >> put decaySetup >> put (map V.toList prS) >> put scale >>  put scaleOutAlg >> put crop >> put stab >> put stabDec >> put clip >> put autoScale
      o@OptAdam{} -> put memSz >> put memStrat >> put batchSz >> put trainIter >> put o >> put smooth >> put smoothPer >> put decaySetup >> put (map V.toList prS) >> put scale >> put scaleOutAlg >> put crop >>  put stab >> put stabDec >> put clip >> put autoScale
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
    autoScale <- get
    return $ NNConfig memSz memStrat batchSz trainIter opt smooth smoothPer decaySetup prS scale scaleOutAlg crop stab stabDec clip autoScale


instance Serialize Torch.Parameter where
  put p = put (Torch.toDependent p)
  get = unsafePerformIO . Torch.makeIndependent <$> get

instance Serialize Torch.Tensor where
  put p = put (show $ Torch.dtype p) >> put (Torch.shape p) >> put p'
    where p' :: [Double]
          p' = Torch.asValue $ Torch.toDType Torch.Double $ Torch.reshape [nr] p
          nr = product (Torch.shape p)
  get = do
    (dtype :: Torch.DType) <- read <$> get
    (shape :: [Int]) <- get
    (values :: [Double]) <- get
    return $ Torch.reshape shape $ Torch.toDType dtype $ Torch.asTensor values


deriving instance Serialize Torch.Linear

deriving instance Generic Torch.LinearSpec

deriving instance Serialize Torch.LinearSpec


instance Serialize Torch.Adam where
  put (Torch.Adam b1 b2 m1 m2 iter) = put b1 >> put b2 >> put m1 >> put m2 >> put iter
  get = Torch.Adam <$> get <*> get <*> get <*> get <*> get

instance Serialize AdamW where
  put (AdamW b1 b2 m1 m2 iter l2 wD) = put b1 >> put b2 >> put m1 >> put m2 >> put iter >> put l2 >> put wD
  get = AdamW <$> get <*> get <*> get <*> get <*> get <*> get <*> get


instance Serialize Proxy where
  put (Scalar x nrAs) = put (0 :: Int) >> put (V.toList x) >> put nrAs
  put (Table m d acts) = put (1 :: Int) >> put (M.mapKeys (first V.toList) . M.map V.toList $ m) >> put (V.toList d) >> put acts
  put (Grenade t w tp conf nr agents wel) = put (2 :: Int) >> put (networkToSpecification t) >> put t >> put w >> put tp >> put conf >> put nr >> put agents >> put wel
  put (Hasktorch t w tp conf nr agents adam mdl wel nnActs) =
    put (3 :: Int) >> put (Torch.flattenParameters t) >> put (Torch.flattenParameters w) >> put tp >> put conf >> put nr >> put agents >> put adam >> put mdl >> put wel >> put nnActs
  put (RegressionProxy m acts nnCfg) = put (4 :: Int) >> put m >> put acts >> put nnCfg
  get =
    fmap force $! do
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
              Grenade t w <$> get <*> get <*> get <*> get <*> get
            SpecConcreteNetwork1D2D (_ :: Network tLayers tShapes) -> do
              (t :: Network tLayers tShapes) <- get
              (w :: Network tLayers tShapes) <- get
              Grenade t w <$> get <*> get <*> get <*> get <*> get
            _ -> error ("Network dimensions not implemented in Serialize Proxy in ML.ARAL.Serialisable")
        3 -> do
          (paramsT :: [Torch.Parameter]) <- get
          (paramsW :: [Torch.Parameter]) <- get
          tp <- get
          conf <- get
          nr <- get
          agents <- get
          adam <- get
          mdl <- get
          wel <- get
          nnActs <- get <|> return False
          return $
            unsafePerformIO $ do
              putStrLn "ANN model: "
              putStrLn $ show mdl
              t <- Torch.sample mdl
              w <- Torch.sample mdl
              return $
                if null paramsT
                  then Hasktorch (t {mlpLayers = []}) (w {mlpLayers = []}) tp conf nr agents adam mdl wel nnActs
                  else Hasktorch (Torch.replaceParameters t paramsT) (Torch.replaceParameters w paramsW) tp conf nr agents adam mdl wel nnActs
        4 -> RegressionProxy <$> get <*> get <*> get
        _ -> error $ "Unknown constructor for proxy: " <> show c


-- ^ Replay Memory
instance Serialize ReplayMemory where
  put (ReplayMemory vec sz idx maxIdx) = do
    -- let xs = unsafePerformIO $ mapM (VM.read vec) [0 .. maxIdx]
    put sz
    put idx
    put maxIdx
    let putReplMem :: Int -> PutM ()
    -- (([Int8], Maybe [[Word8]]), ActionChoice, RewardValue, ([Int8], Maybe [[Word8]]), EpisodeEnd, Word8)
        putReplMem (-1) = return ()
        putReplMem idx = do
          let ((st, as), assel, rew, (st', as'), epsEnd, nrAgs) = unsafePerformIO $ VM.read vec idx
          let !stL =  force $ V.toList st
              !asL =  force $ fmap (VB.toList . VB.map V.toList) as
              !stL' = force $ V.toList st'
              !asL' = force $ fmap (VB.toList . VB.map V.toList) as'
          put ((stL, asL), assel, rew, (stL', asL'), epsEnd, nrAgs)
          putReplMem (idx-1)
    putReplMem maxIdx
  get = do
    !sz <- get
    !idx <- get
    !maxIdx <- get
    let !vec = unsafePerformIO $ VM.new sz
    let getReplMem :: Int -> Get ()
        getReplMem (-1) = return ()
        getReplMem idx = do
          ((!st, !as), !assel, !rew, (!st', !as'), !epsEnd, !nrAg) <- get
          let !stV = force $ V.fromList st
              !asV = force $ fmap (VB.fromList . map V.fromList) as
              !stV' = force $ V.fromList st'
              !asV' = force $ fmap (VB.fromList . map V.fromList) as'
              !tuple = ((stV, asV), assel, rew, (stV', asV'), epsEnd, nrAg)
          unsafePerformIO (VM.write vec idx tuple) `seq` getReplMem (idx-1)
    getReplMem maxIdx
    return $! ReplayMemory vec sz idx maxIdx
    -- -- (xs :: [InternalExperience]) <- map (force . mkReplMem) <$> get
    -- return $!
    --   force $!
    --   unsafePerformIO $! do
    --     !vec <- VM.new sz
    --     vec `seq` foldM_ (\i x -> VM.write vec i x >> return (i + 1)) 0 xs
    --     return $! ReplayMemory vec sz idx maxIdx
