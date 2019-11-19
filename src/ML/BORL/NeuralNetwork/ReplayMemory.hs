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


data ReplayMemory = ReplayMemory
  { _replayMemoryVector :: V.IOVector ((StateFeatures, [ActionIndex]), ActionIndex, Bool, Double, (StateNextFeatures, [ActionIndex]), EpisodeEnd)
  , _replayMemorySize   :: Int  -- size
  , _replayMemoryMaxIdx :: Int  -- in {0,..,size-1}
  }
makeLenses ''ReplayMemory

instance NFData ReplayMemory where
  rnf (ReplayMemory !_ s mx) = rnf s `seq` rnf mx

-- | Add an element to the replay memory. Replaces the oldest elements once the predefined replay memory size is
-- reached.
addToReplayMemory :: Period -> ((StateFeatures, [ActionIndex]), ActionIndex, Bool, Double, (StateNextFeatures, [ActionIndex]), EpisodeEnd) -> ReplayMemory -> IO ReplayMemory
addToReplayMemory p e (ReplayMemory vec sz maxIdx) = do
  let idx = p `mod` fromIntegral sz
  V.write vec (fromIntegral idx) e
  return $ ReplayMemory vec sz (min (maxIdx+1) (sz-1))

-- | Get a list of random input-output tuples from the replay memory.
getRandomReplayMemoryElements :: Batchsize -> ReplayMemory -> IO [((StateFeatures, [ActionIndex]), ActionIndex, Bool, Double, (StateNextFeatures, [ActionIndex]), EpisodeEnd)]
getRandomReplayMemoryElements bs (ReplayMemory vec _ maxIdx) = do
  let len = min bs maxIdx
  g <- newStdGen
  let rands = take len $ randomRs (0,maxIdx) g
  mapM (V.read vec) rands
