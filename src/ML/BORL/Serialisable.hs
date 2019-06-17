{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE ScopedTypeVariables #-}


module ML.BORL.Serialisable where

import           ML.BORL.Action
import           ML.BORL.Algorithm
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import           ML.BORL.Proxy.Type
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.DeepSeq
import           Control.Lens
import           Control.Monad                (void, zipWithM)
import           Control.Monad.IO.Class       (MonadIO, liftIO)
import qualified Data.Map.Strict              as M
import           Data.Maybe                   (fromMaybe)
import qualified Data.Proxy                   as Type
import           Data.Serialize
import           Data.Singletons.Prelude.List
import           Data.Singletons.Prelude.List
import qualified Data.Vector.Mutable          as V
import           GHC.Generics
import           GHC.TypeLits
import           Grenade
import qualified TensorFlow.Core              as TF
import qualified TensorFlow.Session           as TF
import           Unsafe.Coerce

import           System.IO.Unsafe


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

#ifdef DEBUG
  -- Stats:
  , serVisits         :: !(M.Map s Integer) -- ^ Counts the visits of the states
#endif
  } deriving (Generic, Serialize)


toSerialisable :: BORL s -> BORLSerialisable s
toSerialisable = undefined

fromSerialisable :: BORL s -> BORLSerialisable s
fromSerialisable = undefined
  -- 1. state generaliser for table proxy
  -- 2. toNetInput

instance (Ord s, Serialize s) => Serialize (Proxies s)

instance (Serialize s) => Serialize (NNConfig s)  where
  put (NNConfig _ memSz batchSz param prS scale upInt trainMax) =
    put memSz >> put batchSz >> put param >> put prS >> put scale >> put upInt >> put trainMax
  get = undefined


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
    put (2::Int)
    put t
    put w
    put st
    put tp
    put conf
    put nr
  get = do
    (c::Int) <- get
    case c of
      0 -> undefined
      1 -> do
        m <- get
        d <- get
        return $ Table m d id
      2 -> error "Serialisation of Grenade proxies is currently no supported!"
        -- Problem: how to save types?
        -- do
        -- t <- get
        -- w <- get
        -- st <- get
        -- tp <- get
        -- conf <- get
        -- nr <- get
        -- return $ Grenade t w st tp conf nr
      3 -> undefined
      _ -> error "Unknown constructor for proxy"

-- ^ Replay Memory
instance (Serialize s) => Serialize (ReplayMemory s) where
  put (ReplayMemory vec nr) = do
    let xs = unsafePerformIO $ mapM (V.read vec) [0..V.length vec]
    put (V.length vec)
    put xs
    put nr
  get = do
    len <- get
    let vec = unsafePerformIO $ V.new len
    xs :: [(State s, ActionIndex, Bool, Double, StateNext s, EpisodeEnd)] <- get
    return $ void $ unsafePerformIO $ zipWithM (V.write vec) [0..] xs
    nr <- get
    return $ ReplayMemory vec nr
