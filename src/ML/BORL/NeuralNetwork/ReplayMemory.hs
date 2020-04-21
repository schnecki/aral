{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE Rank2Types      #-}
{-# LANGUAGE RankNTypes      #-}
{-# LANGUAGE Strict          #-}
{-# LANGUAGE StrictData      #-}
{-# LANGUAGE TemplateHaskell #-}


module ML.BORL.NeuralNetwork.ReplayMemory where


import           Control.DeepSeq
import           Control.Lens
import qualified Data.Vector          as VI
import qualified Data.Vector.Mutable  as VM
import qualified Data.Vector.Storable as V
import           GHC.Generics
import           System.Random

import           ML.BORL.Types

------------------------------ Replay Memories ------------------------------

data ReplayMemories
  = ReplayMemoriesUnified !ReplayMemory                -- ^ All experiences are saved in a single replay memory.
  | ReplayMemoriesPerActions !(VI.Vector ReplayMemory) -- ^ Split replay memory size among different actions and choose bachsize uniformly among all sets of experiences.
  deriving (Generic)

replayMemories :: ActionIndex -> Lens' ReplayMemories ReplayMemory
replayMemories _ f (ReplayMemoriesUnified rm) = ReplayMemoriesUnified <$> f rm
replayMemories idx f (ReplayMemoriesPerActions rs) = (\x -> ReplayMemoriesPerActions (rs VI.// [(idx, x)])) <$> f (rs VI.! idx)


instance NFData ReplayMemories where
  rnf (ReplayMemoriesUnified rm)    = rnf rm
  rnf (ReplayMemoriesPerActions rs) = rnf rs

------------------------------ Replay Memory ------------------------------

type Experience = ((StateFeatures, V.Vector ActionIndex), ActionIndex, IsRandomAction, RewardValue, (StateNextFeatures, V.Vector ActionIndex), EpisodeEnd)

data ReplayMemory = ReplayMemory
  { _replayMemoryVector :: !(VM.IOVector Experience)
  , _replayMemorySize   :: !Int  -- size
  , _replayMemoryIdx    :: !Int  -- index to use when adding the next element
  , _replayMemoryMaxIdx :: !Int  -- in {0,..,size-1}
  }
makeLenses ''ReplayMemory

instance NFData ReplayMemory where
  rnf (ReplayMemory !_ s idx mx) = rnf s `seq` rnf idx `seq` rnf mx

addToReplayMemories :: Experience -> ReplayMemories -> IO ReplayMemories
addToReplayMemories e (ReplayMemoriesUnified rm) = ReplayMemoriesUnified <$> addToReplayMemory e rm
addToReplayMemories e@(_, idx, _, _, _, _) (ReplayMemoriesPerActions rs) = do
  let r = rs VI.! idx
  r' <- addToReplayMemory e r
  return $!! ReplayMemoriesPerActions (rs VI.// [(idx, r')])

-- | Add an element to the replay memory. Replaces the oldest elements once the predefined replay memory size is
-- reached.
addToReplayMemory :: Experience -> ReplayMemory -> IO ReplayMemory
addToReplayMemory e (ReplayMemory vec sz idx maxIdx) = do
  VM.write vec (fromIntegral idx) e
  return $ ReplayMemory vec sz ((idx+1) `mod` fromIntegral sz) (min (maxIdx+1) (sz-1))

-- | Get a list of random input-output tuples from the replay memory. Returns a list at least the length of the batch size with a list of consecutive experiences without a terminal state in between.
-- In case a terminal state is detected the list is split and the first list exeecds the batchsize.
getRandomReplayMemoryElements :: NStep -> Batchsize -> ReplayMemory -> IO [[Experience]]
getRandomReplayMemoryElements _ _ (ReplayMemory _ _ _ 0) = return []
getRandomReplayMemoryElements 1 bs (ReplayMemory vec _ _ maxIdx) = do
  let len = min bs maxIdx
  g <- newStdGen
  let rands = take len $ randomRs (0, maxIdx) g
  map return <$> mapM (VM.read vec) rands
getRandomReplayMemoryElements nStep bs (ReplayMemory vec _ nxtIdx maxIdx) = do -- get consequitive experiences
  let len = min bs maxIdx
  g <- newStdGen
  let rands = take len $ randomRs (nStep - 1, maxIdx) g
      idxes = map (\r -> filter (>= 0) [r - nStep + 1 .. r]) rands
  concat <$> mapM (fmap splitTerminal . mapM (VM.read vec)) idxes
  where
    isTerminal (_, _, _, _, _, t) = t
    splitTerminal xs = filter (not . null) [takeWhile (not . isTerminal) xs, dropWhile (not . isTerminal) xs]

-- | Get a list of random input-output tuples from the replay memory.
getRandomReplayMemoriesElements :: NStep -> Batchsize -> ReplayMemories -> IO [[Experience]]
getRandomReplayMemoriesElements nStep bs (ReplayMemoriesUnified rm) = getRandomReplayMemoryElements nStep bs rm
getRandomReplayMemoriesElements 1 bs (ReplayMemoriesPerActions rs) = concat <$> mapM (getRandomReplayMemoryElements 1 nr) (VI.toList rs)
  where nr = ceiling (fromIntegral bs / fromIntegral (VI.length rs) :: Float)
getRandomReplayMemoriesElements _ _ ReplayMemoriesPerActions{}  = error "ReplayMemoriesPerActions does not work with nStep > 1!"

-- | Size of replay memory (combined if it is a per action replay memory).
replayMemoriesSize :: ReplayMemories -> Int
replayMemoriesSize (ReplayMemoriesUnified m)     = m ^. replayMemorySize
replayMemoriesSize (ReplayMemoriesPerActions ms) = sum $ VI.map (view replayMemorySize) ms

replayMemoriesSubSize :: ReplayMemories -> Int
replayMemoriesSubSize (ReplayMemoriesUnified m) = m ^. replayMemorySize
replayMemoriesSubSize (ReplayMemoriesPerActions xs)
  | VI.null xs = 0
  | otherwise = VI.head xs ^. replayMemorySize
