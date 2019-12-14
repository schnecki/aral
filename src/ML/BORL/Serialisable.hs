{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Unsafe              #-}


module ML.BORL.Serialisable where

import           ML.BORL.Action
import           ML.BORL.Algorithm
import           ML.BORL.Decay
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import           ML.BORL.Proxy.Proxies
import           ML.BORL.Proxy.Type
import           ML.BORL.Reward.Type
import           ML.BORL.SaveRestore
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Arrow         (first)
import           Control.DeepSeq
import           Control.Lens
import           Control.Monad         (void, zipWithM, zipWithM_)
import           Data.List             (find, foldl')
import           Data.Serialize
import qualified Data.Vector.Mutable   as V
import           GHC.Generics
import           System.IO.Unsafe
import qualified TensorFlow.Core       as TF

import           Debug.Trace


type ActionList s = [ActionIndexed s]
type ActionFilter s = s -> [Bool]
type ProxyNetInput s = s -> [Double]
type TensorflowModelBuilder = TF.Session TensorflowModel


data BORLSerialisable s = BORLSerialisable
  { serS              :: !s                    -- ^ Current state.
  , serT              :: !Int                  -- ^ Current time t.
  , serEpisodeNrStart :: !(Int, Int)           -- ^ Nr of Episode and start period.
  , serParameters     :: !ParameterInitValues  -- ^ Parameter setup.
  , serRewardFutures  :: [RewardFutureData s]

  -- define algorithm to use
  , serAlgorithm      :: !(Algorithm s)
  , serPhase          :: !Phase

  -- Values:
  , serLastVValues    :: ![Double]                 -- ^ List of X last V values
  , serLastRewards    :: ![Double]                 -- ^ List of X last rewards
  , serPsis           :: !(Double, Double, Double)  -- ^ Exponentially smoothed psi values.
  , serProxies        :: Proxies                    -- ^ Scalar, Tables and Neural Networks
  } deriving (Generic, Serialize)

toSerialisable :: (MonadBorl' m, Ord s, RewardFuture s) => BORL s -> m (BORLSerialisable s)
toSerialisable = toSerialisableWith id id


toSerialisableWith :: (MonadBorl' m, Ord s', RewardFuture s') => (s -> s') -> (StoreType s -> StoreType s') -> BORL s -> m (BORLSerialisable s')
toSerialisableWith f g borl@(BORL _ _ s _ t eNr par _ _ alg ph v rew psis prS) = do
  BORL _ _ s _ t eNr par _ future alg ph v rew psis prS <- saveTensorflowModels borl
  return $ BORLSerialisable (f s) t eNr par (map (mapRewardFutureData f g) future) (mapAlgorithm f alg) ph v rew psis prS

fromSerialisable :: (MonadBorl' m, Ord s, NFData s, RewardFuture s) => [Action s] -> ActionFilter s -> Decay -> FeatureExtractor s -> ProxyNetInput s -> TensorflowModelBuilder -> BORLSerialisable s -> m (BORL s)
fromSerialisable = fromSerialisableWith id id

fromSerialisableWith ::
     (MonadBorl' m, Ord s, NFData s, RewardFuture s)
  => (s' -> s)
  -> (StoreType s' -> StoreType s)
  -> [Action s]
  -> ActionFilter s
  -> Decay
  -> FeatureExtractor s
  -> ProxyNetInput s
  -> TensorflowModelBuilder
  -> BORLSerialisable s'
  -> m (BORL s)
fromSerialisableWith f g as aF decay ftExt inp builder (BORLSerialisable s t e par future alg ph lastV rew psis prS) = do
  let aL = zip [idxStart ..] as
      borl = BORL aL aF (f s) ftExt t e par decay (map (mapRewardFutureData f g) future) (mapAlgorithm f alg) ph lastV rew psis prS
      pxs = borl ^. proxies
      borl' =
        flip (foldl' (\b p -> over (proxies . p . proxyTFWorker) (\x -> x {tensorflowModelBuilder = builder}) b)) (allProxiesLenses pxs) $
        flip (foldl' (\b p -> over (proxies . p . proxyTFTarget) (\x -> x {tensorflowModelBuilder = builder}) b)) (allProxiesLenses pxs) borl
  restoreTensorflowModels False borl'
  return $ force borl'


instance Serialize Proxies

instance Serialize NNConfig where
  put (NNConfig memSz batchSz lp decaySetup prS scale stab stabDec upInt trainMax param) = put memSz >> put batchSz >> put lp >> put decaySetup >> put prS >> put scale >> put stab >> put stabDec >> put upInt >> put trainMax >> put param
  get = do
    memSz <- get
    batchSz <- get
    lp <- get
    decaySetup <- get
    prS <- get
    scale <- get
    stab <- get
    stabDec <- get
    upInt <- get
    trainMax <- get
    param <- get
    return $ NNConfig memSz batchSz lp decaySetup prS scale stab stabDec upInt trainMax param


instance Serialize Proxy where
  put (Scalar x)    = put (0::Int) >> put x
  put (Table m d)  = put (1::Int) >> put m >> put d
  put (Grenade t w st tp conf nr) = do
    put (2::Int)
    put t
    put w
    put st
    put tp
    put conf
    put nr
  put (TensorflowProxy t w st tp conf nr) = do
    put (3::Int)
    put t
    put w
    put st
    put tp
    put conf
    put nr
  get = do
    (c::Int) <- get
    case c of
      0 -> get >>= return . Scalar
      1 -> do
        m <- get
        d <- get
        return $ Table m d
      2 -> error "Deserialisation of Grenade proxies is currently no supported!"
        -- Problem: how to save types?
        -- do
        -- t <- get
        -- w <- get
        -- st <- get
        -- tp <- get
        -- conf <- get
        -- nr <- get
        -- return $ Grenade t w st tp conf nr
      3 -> do
        t <- get
        w <- get
        st <- get
        tp <- get
        conf <- get
        nr <- get
        return $ TensorflowProxy t w st tp conf nr
      _ -> error "Unknown constructor for proxy"

-- ^ Replay Memory
instance Serialize ReplayMemory where
  put (ReplayMemory vec sz maxIdx) = do
    let xs = unsafePerformIO $ mapM (V.read vec) [0 .. maxIdx]
    put sz
    put xs
    put maxIdx
  get = do
    sz <- get
    xs :: [((StateFeatures, [ActionIndex]), ActionIndex, Bool, Double, (StateNextFeatures, [ActionIndex]), EpisodeEnd)] <- get
    maxIdx <- get
    return $
      unsafePerformIO $ do
        vec <- V.new sz
        vec `seq` zipWithM_ (V.write vec) [0 .. maxIdx] xs
        return (ReplayMemory vec sz maxIdx)
