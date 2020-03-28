{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE Rank2Types      #-}
{-# LANGUAGE RankNTypes      #-}
{-# LANGUAGE TemplateHaskell #-}


module ML.BORL.NeuralNetwork.ReplayMemory where


import           Control.DeepSeq
import           Control.Lens
import           Data.List            (genericLength)
import qualified Data.Vector.Mutable  as VM
import qualified Data.Vector.Storable as V
import           GHC.Generics
import           System.Random

import           ML.BORL.Types

import           Debug.Trace

------------------------------ Replay Memories ------------------------------

data ReplayMemories
  = ReplayMemoriesUnified ReplayMemory      -- ^ All experiences are saved in a single replay memory.
  | ReplayMemoriesPerActions [ReplayMemory] -- ^ Split replay memory size among different actions and choose bachsize uniformly among all sets of experiences.
  deriving (Generic)

replayMemories :: ActionIndex -> Lens' ReplayMemories ReplayMemory
replayMemories _ f (ReplayMemoriesUnified rm) = ReplayMemoriesUnified <$> f rm
replayMemories idx f (ReplayMemoriesPerActions rs) = (\x -> ReplayMemoriesPerActions (take idx rs ++ x : drop (idx+1) rs)) <$> f (rs !! idx)


instance NFData ReplayMemories where
  rnf (ReplayMemoriesUnified rm)    = rnf rm
  rnf (ReplayMemoriesPerActions rs) = rnf rs

------------------------------ Replay Memory ------------------------------

type Experience = ((StateFeatures, V.Vector ActionIndex), ActionIndex, IsRandomAction, RewardValue, (StateNextFeatures, V.Vector ActionIndex), EpisodeEnd)

data ReplayMemory = ReplayMemory
  { _replayMemoryVector :: VM.IOVector Experience
  , _replayMemorySize   :: Int  -- size
  , _replayMemoryIdx    :: Int  -- index to use when adding the next element
  , _replayMemoryMaxIdx :: Int  -- in {0,..,size-1}
  }
makeLenses ''ReplayMemory

instance NFData ReplayMemory where
  rnf (ReplayMemory !_ s idx mx) = rnf s `seq` rnf idx `seq` rnf mx

addToReplayMemories :: Experience -> ReplayMemories -> IO ReplayMemories
addToReplayMemories e (ReplayMemoriesUnified rm) = ReplayMemoriesUnified <$> addToReplayMemory e rm
addToReplayMemories e@(_, idx, _, _, _, _) (ReplayMemoriesPerActions rs) = do
  let r = rs !! idx
  r' <- addToReplayMemory e r
  return $ ReplayMemoriesPerActions (take idx rs ++ r' : drop (idx + 1) rs)

-- | Add an element to the replay memory. Replaces the oldest elements once the predefined replay memory size is
-- reached.
addToReplayMemory :: Experience -> ReplayMemory -> IO ReplayMemory
addToReplayMemory e (ReplayMemory vec sz idx maxIdx) = do
  VM.write vec (fromIntegral idx) e
  return $ ReplayMemory vec sz ((idx+1) `mod` fromIntegral sz) (min (maxIdx+1) (sz-1))

-- | Get a list of random input-output tuples from the replay memory.
getRandomReplayMemoryElements :: Batchsize -> ReplayMemory -> IO [Experience]
getRandomReplayMemoryElements bs (ReplayMemory vec _ _ maxIdx) = do
  let len = min bs maxIdx
  g <- newStdGen
  let rands = take len $ randomRs (0,maxIdx) g
  mapM (VM.read vec) rands

-- | Get a list of random input-output tuples from the replay memory.
getRandomReplayMemoriesElements :: Batchsize -> ReplayMemories -> IO [Experience]
getRandomReplayMemoriesElements bs (ReplayMemoriesUnified rm) = getRandomReplayMemoryElements bs rm
getRandomReplayMemoriesElements bs (ReplayMemoriesPerActions rs) = concat <$> mapM (getRandomReplayMemoryElements nr) rs
  where nr = ceiling (fromIntegral bs / genericLength rs :: Float)

-- | Size of replay memory (combined if it is a per action replay memory).
replayMemoriesSize :: ReplayMemories -> Int
replayMemoriesSize (ReplayMemoriesUnified m)     = m ^. replayMemorySize
replayMemoriesSize (ReplayMemoriesPerActions ms) = sum $ map (view replayMemorySize) ms

