{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}


module ML.BORL.NeuralNetwork.ReplayMemory where

import           ML.BORL.Types

import           Control.DeepSeq
import           Control.Lens
import           Control.Monad
import qualified Data.Vector.Mutable as V
import           System.IO.Unsafe
import           System.Random

import           Debug.Trace

data ReplayMemory s = ReplayMemory
  { _replayMemoryVector :: V.IOVector (State s, ActionIndex, Bool, Double, StateNext s, EpisodeEnd)
  , _replayMemorySize   :: Int
  , _replayMemoryMaxIdx :: Int
  }
makeLenses ''ReplayMemory

mapReplayMemoryForSeialisable :: (s -> s') -> ReplayMemory s -> ReplayMemory s'
mapReplayMemoryForSeialisable f (ReplayMemory vec nr maxIdx) =
  let !vec' = unsafePerformIO $ V.new nr
   in unsafePerformIO (mapM (\i -> V.read vec i >>= V.write vec' i . fun) [0 .. maxIdx]) `seq` ReplayMemory vec' nr maxIdx
  where
    fun (s, idx, isRandAct, r, s', episodeEnd) = (f s, idx, isRandAct, r, f s', episodeEnd)


instance NFData (ReplayMemory s) where
  rnf (ReplayMemory !_ s mx) = rnf s `seq` rnf mx

-- | Add an element to the replay memory. Replaces the oldest elements once the predefined replay memory size is
-- reached.
addToReplayMemory :: Period -> (State s, ActionIndex, Bool, Double, StateNext s, EpisodeEnd) -> ReplayMemory s -> IO (ReplayMemory s)
addToReplayMemory p e (ReplayMemory vec sz maxIdx) = do
  let idx = p `mod` fromIntegral sz
  V.write vec (fromIntegral idx) e
  return $ ReplayMemory vec sz (min (maxIdx+1) (sz-1))

-- | Get a list of random input-output tuples from the replay memory.
getRandomReplayMemoryElements :: Batchsize -> ReplayMemory s -> IO [(State s, ActionIndex, Bool, Double, StateNext s, EpisodeEnd)]
getRandomReplayMemoryElements bs (ReplayMemory vec _ maxIdx) = do
  let len = min bs maxIdx
  g <- newStdGen
  let rands = take len $ randomRs (0,maxIdx) g
  mapM (V.read vec) rands
