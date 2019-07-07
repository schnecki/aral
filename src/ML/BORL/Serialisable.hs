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
import           ML.BORL.Proxy.Type
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Lens
import           Control.Monad         (void, zipWithM)
import           Data.List             (find)
import           Data.Serialize
import qualified Data.Vector.Mutable   as V
import           GHC.Generics
import           System.IO.Unsafe
import qualified TensorFlow.Core       as TF

import           Debug.Trace


data BORLSerialisable s = BORLSerialisable
  { -- serActionList     :: ![ActionIndexed s]    -- ^ List of possible actions in state s.
  -- , serActionFilter   :: !(s -> [Bool])        -- ^ Function to filter actions in state s.
    serS              :: !s                    -- ^ Current state.
  , serT              :: !Integer              -- ^ Current time t.
  , serEpisodeNrStart :: !(Integer, Integer)   -- ^ Nr of Episode and start period.
  , serParameters     :: !Parameters           -- ^ Parameter setup.
  -- , serDecayFunction  :: !Decay                -- ^ Decay function at period t.

  -- define algorithm to use
  , serAlgorithm      :: !Algorithm
  , serPhase          :: !Phase

  -- Values:
  , serLastVValues    :: ![Double] -- ^ List of X last V values
  , serLastRewards    :: ![Double] -- ^ List of X last rewards
  , serPsis           :: !(Double, Double, Double)  -- ^ Exponentially smoothed psi values.
  , serProxies        :: Proxies s                  -- ^ Scalar, Tables and Neural Networks
  } deriving (Generic, Serialize)

saveTensorflowModels :: (MonadBorl' m) => BORL s -> m (BORL s)
saveTensorflowModels borl = do
  mapM_ saveProxy (allProxies $ borl ^. proxies)
  return borl
  where
    saveProxy px =
      case px of
        TensorflowProxy netT netW _ _ _ _ -> saveModelWithLastIO netT >> saveModelWithLastIO netW >> return ()
        _ -> return ()

restoreTensorflowModels :: (MonadBorl' m) => BORL s -> m ()
restoreTensorflowModels borl = do
  buildModels
  mapM_ restoreProxy (allProxies $ borl ^. proxies)
  where
    restoreProxy px =
      case px of
        TensorflowProxy netT netW _ _ _ _ -> restoreModelWithLastIO netT >> restoreModelWithLastIO netW >> return ()
        _ -> return ()
    buildModels =
      case find isTensorflow (allProxies $ borl ^. proxies) of
        Just (TensorflowProxy netT _ _ _ _ _) -> buildTensorflowModel netT
        _                                     -> return ()


toSerialisable :: (MonadBorl' m, Ord s) => BORL s -> m (BORLSerialisable s)
toSerialisable = toSerialisableWith id

toSerialisableWith :: (MonadBorl' m, Ord s') => (s -> s') -> BORL s -> m (BORLSerialisable s')
toSerialisableWith f borl@(BORL _ _ s t e par _ alg ph v rew psis prS) = do
  BORL _ _ s t e par _ alg ph v rew psis prS <- saveTensorflowModels borl
  return $ BORLSerialisable (f s) t e par alg ph v rew psis (mapProxiesForSerialise t f prS)


type ActionList s = [ActionIndexed s]
type ActionFilter s = s -> [Bool]
type ProxyNetInput s = s -> [Double]
type TensorflowModelBuilder = TF.Session TensorflowModel


fromSerialisable :: (MonadBorl' m, Ord s) => [Action s] -> ActionFilter s -> Decay -> TableStateGeneraliser s -> ProxyNetInput s -> TensorflowModelBuilder -> BORLSerialisable s -> m (BORL s)
fromSerialisable = fromSerialisableWith id

fromSerialisableWith :: (MonadBorl' m, Ord s) => (s' -> s) -> [Action s] -> ActionFilter s -> Decay -> TableStateGeneraliser s -> ProxyNetInput s -> TensorflowModelBuilder -> BORLSerialisable s' -> m (BORL s)
fromSerialisableWith f as aF decay gen inp builder (BORLSerialisable s t e par alg ph lastV rew psis prS) = trace ("fromSerialisableWith") $ do
  let aL = zip [idxStart ..] as
      borl = BORL aL aF (f s) t e par decay alg ph lastV rew psis (mapProxiesForSerialise t f prS)
   in do let borl' =
               flip (foldl (\b p -> over (proxies . p . filtered isTensorflow . proxyTFWorker) (\x -> x {tensorflowModelBuilder = builder}) b)) [rhoMinimum, rho, psiV, v, w, r0, r1] $
               flip (foldl (\b p -> over (proxies . p . filtered isTensorflow . proxyTFTarget) (\x -> x {tensorflowModelBuilder = builder}) b)) [rhoMinimum, rho, psiV, v, w, r0, r1] $
               flip (foldl (\b p -> set (proxies . p . filtered isNeuralNetwork . proxyNNConfig . toNetInp) inp b)) [rhoMinimum, rho, psiV, v, w, r0, r1] $
               foldl (\b p -> set (proxies . p . filtered isTable . proxyStateGeneraliser) gen b) borl [rhoMinimum, rho, psiV, v, w, r0, r1]
         restoreTensorflowModels borl'
         return borl'

instance (Ord s, Serialize s) => Serialize (Proxies s)

instance (Serialize s) => Serialize (NNConfig s) where
  put (NNConfig _ memSz batchSz param prS scale upInt trainMax) = put memSz >> put batchSz >> put param >> put prS >> put scale >> put upInt >> put trainMax
  get = trace ("serialize nnconfig") $ do
    memSz <- get
    batchSz <- get
    param <- get
    prS <- get
    scale <- get
    upInt <- get
    trainMax <- get
    return $ NNConfig (error "called netInp") memSz batchSz param prS scale upInt trainMax


instance (Ord s, Serialize s) => Serialize (Proxy s) where
  put (Scalar x)    = put (0::Int) >> put x
  put (Table m d _) = put (1::Int) >> put m >> put d
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
  get = trace ("serialize proxy") $ do
    (c::Int) <- get
    case c of
      0 -> get >>= return . Scalar
      1 -> do
        m <- get
        d <- get
        return $ Table m d (const [])
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
instance (Serialize s) => Serialize (ReplayMemory s) where
  put (ReplayMemory vec sz maxIdx) = do
    let xs = unsafePerformIO $ mapM (V.read vec) [0..maxIdx]
    put sz
    put xs
    put maxIdx
  get = trace ("serialize replay memory") $ do
    sz <- get
    let vec = unsafePerformIO $ V.new sz
    xs :: [(State s, ActionIndex, Bool, Double, StateNext s, EpisodeEnd)] <- get
    maxIdx <- get
    void $ return $ unsafePerformIO $ zipWithM (V.write vec) [0..] xs
    return $ ReplayMemory vec sz maxIdx
