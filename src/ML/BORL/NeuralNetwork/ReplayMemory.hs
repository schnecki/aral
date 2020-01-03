{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}


module ML.BORL.NeuralNetwork.ReplayMemory where

import           ML.BORL.Types

import           Control.DeepSeq
import           Control.Lens
import qualified Data.Vector.Mutable as V
import           System.Random

import           Debug.Trace


data ReplayMemory = ReplayMemory
  { _replayMemoryVector :: V.IOVector ((StateFeatures, [ActionIndex]), ActionIndex, Bool, Double, (StateNextFeatures, [ActionIndex]), EpisodeEnd)
  , _replayMemorySize   :: Int  -- size
  , _replayMemoryIdx    :: Int  -- index to use when adding the next element
  , _replayMemoryMaxIdx :: Int  -- in {0,..,size-1}
  }
makeLenses ''ReplayMemory

instance NFData ReplayMemory where
  rnf (ReplayMemory !_ s idx mx) = rnf s `seq` rnf idx `seq` rnf mx

-- | Add an element to the replay memory. Replaces the oldest elements once the predefined replay memory size is
-- reached.
addToReplayMemory :: ((StateFeatures, [ActionIndex]), ActionIndex, Bool, Double, (StateNextFeatures, [ActionIndex]), EpisodeEnd) -> ReplayMemory -> IO ReplayMemory
addToReplayMemory e (ReplayMemory vec sz idx maxIdx) = do
  V.write vec (fromIntegral idx) e
  return $ ReplayMemory vec sz ((idx+1) `mod` fromIntegral sz) (min (maxIdx+1) (sz-1))

-- | Get a list of random input-output tuples from the replay memory.
getRandomReplayMemoryElements :: Batchsize -> ReplayMemory -> IO [((StateFeatures, [ActionIndex]), ActionIndex, Bool, Double, (StateNextFeatures, [ActionIndex]), EpisodeEnd)]
getRandomReplayMemoryElements bs (ReplayMemory vec _ _ maxIdx) = do
  let len = min bs maxIdx
  g <- newStdGen
  let rands = take len $ randomRs (0,maxIdx) g
  mapM (V.read vec) rands
