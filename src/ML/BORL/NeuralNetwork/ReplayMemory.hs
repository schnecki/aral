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
import           Control.Monad        (foldM)
import           Data.Maybe           (fromJust, isNothing)
import qualified Data.Vector          as VI
import qualified Data.Vector.Mutable  as VM
import qualified Data.Vector.Storable as V
import           GHC.Generics
import           System.Random

import           ML.BORL.Types

import           Debug.Trace

------------------------------ Replay Memories ------------------------------

data ReplayMemories
  = ReplayMemoriesUnified !ReplayMemory                -- ^ All experiences are saved in a single replay memory.
  | ReplayMemoriesPerActions (Maybe ReplayMemory) !(VI.Vector ReplayMemory) -- ^ Split replay memory size among different actions and choose bachsize uniformly among all sets of experiences.
  deriving (Generic)

instance Show ReplayMemories where
  show (ReplayMemoriesUnified r)       = "Unified " <> show r
  show (ReplayMemoriesPerActions _ rs) = "Per Action " <> show (VI.head rs)

replayMemories :: ActionIndex -> Lens' ReplayMemories ReplayMemory
replayMemories _ f (ReplayMemoriesUnified rm) = ReplayMemoriesUnified <$> f rm
replayMemories idx f (ReplayMemoriesPerActions tmp rs) = (\x -> ReplayMemoriesPerActions tmp (rs VI.// [(idx, x)])) <$> f (rs VI.! idx)


instance NFData ReplayMemories where
  rnf (ReplayMemoriesUnified rm)        = rnf rm
  rnf (ReplayMemoriesPerActions tmp rs) = rnf1 tmp `seq` rnf rs

------------------------------ Replay Memory ------------------------------

type Experience = ((StateFeatures, V.Vector ActionIndex), ActionIndex, IsRandomAction, RewardValue, (StateNextFeatures, V.Vector ActionIndex), EpisodeEnd)

data ReplayMemory = ReplayMemory
  { _replayMemoryVector :: !(VM.IOVector Experience)
  , _replayMemorySize   :: !Int  -- size
  , _replayMemoryIdx    :: !Int  -- index to use when adding the next element
  , _replayMemoryMaxIdx :: !Int  -- in {0,..,size-1}
  }
makeLenses ''ReplayMemory

instance Show ReplayMemory where
  show (ReplayMemory _ sz idx maxIdx) = "Replay Memory with size " <> show sz <> ". Next index: " <> show idx <> "/" <> show maxIdx

instance NFData ReplayMemory where
  rnf (ReplayMemory !_ s idx mx) = rnf s `seq` rnf idx `seq` rnf mx

addToReplayMemories :: NStep -> Experience -> ReplayMemories -> IO ReplayMemories
addToReplayMemories _ e (ReplayMemoriesUnified rm) = ReplayMemoriesUnified <$> addToReplayMemory e rm
addToReplayMemories 1 e@(_, idx, _, _, _, _) (ReplayMemoriesPerActions tmp rs) = do
  let r = rs VI.! idx
  r' <- addToReplayMemory e r
  return $!! ReplayMemoriesPerActions tmp (rs VI.// [(idx, r')])
addToReplayMemories _ e (ReplayMemoriesPerActions Nothing rs) = addToReplayMemories 1 e (ReplayMemoriesPerActions Nothing rs) -- cannot use action tmp replay memory, add immediately
addToReplayMemories _ e (ReplayMemoriesPerActions (Just tmpRepMem) rs) = do
  tmpRepMem' <- addToReplayMemory e tmpRepMem
  if tmpRepMem' ^. replayMemoryIdx == 0
    then do -- temporary replay memory full, add experience to corresponding action memory
    let vec = tmpRepMem' ^. replayMemoryVector
    mems <- mapM (VM.read vec) [0.. tmpRepMem' ^. replayMemorySize-1]
    let startIdx = head mems ^. _2
    let r = rs VI.! startIdx
    r' <- foldM (flip addToReplayMemory) r mems
    return $!! ReplayMemoriesPerActions (Just tmpRepMem') (rs VI.// [(startIdx, r')])
    else return $!! ReplayMemoriesPerActions (Just tmpRepMem') rs


-- | Add an element to the replay memory. Replaces the oldest elements once the predefined replay memory size is
-- reached.
addToReplayMemory :: Experience -> ReplayMemory -> IO ReplayMemory
addToReplayMemory e (ReplayMemory vec sz idx maxIdx) = do
  VM.write vec (fromIntegral idx) e
  return $ ReplayMemory vec sz ((idx+1) `mod` fromIntegral sz) (min (maxIdx+1) (sz-1))

type AllExpAreConsecutive = Bool

-- | Get a list of random input-output tuples from the replay memory. Returns a list at least the length of the batch size with a list of consecutive experiences without a terminal state in between.
-- In case a terminal state is detected the list is split and the first list exeecds the batchsize.
getRandomReplayMemoryElements :: AllExpAreConsecutive -> NStep -> Batchsize -> ReplayMemory -> IO [[Experience]]
getRandomReplayMemoryElements _ _ _ (ReplayMemory _ _ _ 0) = return []
getRandomReplayMemoryElements _ 1 bs (ReplayMemory vec size _ maxIdx) = do
  let len = min bs (1 + maxIdx)
  g <- newStdGen
  let rands
        | len == size = take len [0 .. maxIdx]
        | otherwise = take len $ randomRs (0, maxIdx) g
  map return <$> mapM (VM.read vec) rands
getRandomReplayMemoryElements True nStep bs (ReplayMemory vec size _ maxIdx) = do -- get consequitive experiences
  let len = min bs (1 + maxIdx)
  g <- newStdGen
  let rands
        | len * nStep == size = take len [nStep - 1,2 * nStep - 1 .. maxIdx]
        | otherwise = take len $ randomRs (nStep - 1, maxIdx) g
      idxes = map (\r -> filter (>= 0) [r - nStep + 1 .. r]) rands
  concat <$> mapM (fmap splitTerminal . mapM (VM.read vec)) idxes
  where
    isTerminal (_, _, _, _, _, t) = t
    splitTerminal xs = filter (not . null) [takeWhile (not . isTerminal) xs, dropWhile (not . isTerminal) xs]
getRandomReplayMemoryElements False nStep bs (ReplayMemory vec size _ maxIdx)
  | nStep > maxIdx + 1 = return []
  | otherwise = do
    let len = min bs (maxIdx + 1)
    g <- newStdGen
    let rands
          | len * nStep == size = take len [0,nStep .. maxIdx - nStep + 1]
          | otherwise = map (subtract nStep . (* nStep)) $ take len $ randomRs (1, (maxIdx + 1) `div` nStep) g
        idxes = map (\r -> [r .. r + nStep - 1]) rands
    concat <$> mapM (fmap splitTerminal . mapM (VM.read vec)) idxes
  where
    isTerminal (_, _, _, _, _, t) = t
    splitTerminal xs = filter (not . null) [takeWhile (not . isTerminal) xs, dropWhile (not . isTerminal) xs]

-- | Get a list of random input-output tuples from the replay memory.
getRandomReplayMemoriesElements :: NStep -> Batchsize -> ReplayMemories -> IO [[Experience]]
getRandomReplayMemoriesElements nStep bs (ReplayMemoriesUnified rm) = getRandomReplayMemoryElements True nStep bs rm
getRandomReplayMemoriesElements 1 bs (ReplayMemoriesPerActions _ rs) = concat <$> mapM (getRandomReplayMemoryElements False 1 bs) (VI.toList rs)
  -- where nr = ceiling (fromIntegral bs / fromIntegral (VI.length rs) :: Float)
getRandomReplayMemoriesElements nStep bs (ReplayMemoriesPerActions tmpRepMem rs)
  | nStep == 1 || isNothing tmpRepMem = concat <$> mapM (getRandomReplayMemoryElements False 1 bs) (VI.toList rs)
  | otherwise = do
     xs <- concat <$> mapM (getRandomReplayMemoryElements False nStep bs) (VI.toList rs)
     if null xs
       then getRandomReplayMemoryElements False 1 bs (fromJust tmpRepMem)
       else return xs
  -- where nr = ceiling (fromIntegral bs / fromIntegral (VI.length rs) :: Float)


-- | Size of replay memory (combined if it is a per action replay memory).
replayMemoriesSize :: ReplayMemories -> Int
replayMemoriesSize (ReplayMemoriesUnified m)     = m ^. replayMemorySize
replayMemoriesSize (ReplayMemoriesPerActions _ ms) = sum $ VI.map (view replayMemorySize) ms

replayMemoriesSubSize :: ReplayMemories -> Int
replayMemoriesSubSize (ReplayMemoriesUnified m) = m ^. replayMemorySize
replayMemoriesSubSize (ReplayMemoriesPerActions _ xs)
  | VI.null xs = 0
  | otherwise = VI.head xs ^. replayMemorySize


