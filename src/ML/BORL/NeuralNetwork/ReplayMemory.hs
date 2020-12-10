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
import           Control.Monad       (foldM)
import           Data.Maybe          (fromJust, isNothing)
import qualified Data.Vector         as VB
import qualified Data.Vector.Mutable as VM
import           GHC.Generics
import           System.Random

import           ML.BORL.Types


------------------------------ Replay Memories ------------------------------


data ReplayMemories
  = ReplayMemoriesUnified !NumberOfActions !ReplayMemory                                      -- ^ All experiences are saved in a single replay memory.
  | ReplayMemoriesPerActions !NumberOfActions !(Maybe ReplayMemory) !(VB.Vector ReplayMemory) -- ^ Split replay memory size among different actions and choose bachsize uniformly among all sets of
                                                                                            -- experiences. For multiple agents the action that is used to save the experience is selected randomly.
  deriving (Generic)

instance Show ReplayMemories where
  show (ReplayMemoriesUnified _ r)       = "Unified " <> show r
  show (ReplayMemoriesPerActions _ _ rs) = "Per Action " <> show (VB.head rs)

replayMemories :: ActionIndex -> Lens' ReplayMemories ReplayMemory
replayMemories _ f (ReplayMemoriesUnified nr rm) = ReplayMemoriesUnified nr <$> f rm
replayMemories idx f (ReplayMemoriesPerActions nr tmp rs) = (\x -> ReplayMemoriesPerActions nr tmp (rs VB.// [(idx, x)])) <$> f (rs VB.! idx)


instance NFData ReplayMemories where
  rnf (ReplayMemoriesUnified nr rm)        = rnf nr `seq` rnf rm
  rnf (ReplayMemoriesPerActions nr tmp rs) = rnf nr `seq` rnf1 tmp `seq` rnf rs

------------------------------ Replay Memory ------------------------------

type Experience = ((StateFeatures, DisallowedActionIndicies),     -- State Features s & allowed actions per agent
                   ActionChoice,                                  -- true, iff the action was randomly chosen, actionIndex a
                   RewardValue,                                   -- reward r
                   (StateNextFeatures, DisallowedActionIndicies), -- state features s' & allowed actions per agent
                   EpisodeEnd)                                    -- true, iff it is the end of the episode

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
addToReplayMemories _ e (ReplayMemoriesUnified nr rm) = ReplayMemoriesUnified nr <$> addToReplayMemory nr e rm
addToReplayMemories 1 e@(_, actChoices, _, _, _) (ReplayMemoriesPerActions nrAs tmp rs)
  | VB.length actChoices == 1 = do
    let (_,idx) = VB.head actChoices
    let r = rs VB.! idx
    r' <- addToReplayMemory nrAs e r
    return $! ReplayMemoriesPerActions nrAs tmp (rs VB.// [(idx, r')])
addToReplayMemories 1 e@(_, actChoices, _, _, _) (ReplayMemoriesPerActions nrAs tmp rs) = do
  agent <- randomRIO (0, length actChoices - 1) -- randomly choose an agent that specifies the action
  let actionIdx = snd $ actChoices VB.! agent
      memIdx = agent * nrAs + actionIdx
  let r = rs VB.! memIdx
  r' <- addToReplayMemory nrAs e r
  return $! ReplayMemoriesPerActions nrAs tmp (rs VB.// [(memIdx, r')])
addToReplayMemories _ e (ReplayMemoriesPerActions nrAs Nothing rs) = addToReplayMemories 1 e (ReplayMemoriesPerActions nrAs Nothing rs) -- cannot use action tmp replay memory, add immediately
addToReplayMemories _ e (ReplayMemoriesPerActions nrAs (Just tmpRepMem) rs) = do
  tmpRepMem' <- addToReplayMemory nrAs e tmpRepMem
  if tmpRepMem' ^. replayMemoryIdx == 0
    then do -- temporary replay memory full, add experience to corresponding action memory
    let vec = tmpRepMem' ^. replayMemoryVector
    mems <- mapM (VM.read vec) [0.. tmpRepMem' ^. replayMemorySize-1]
    startIdx <- case head mems ^. _2 of
          idxs | VB.length idxs == 1       -> return (snd $ VB.head idxs)
          idxs      -> do
            agent <- randomRIO (0, length idxs - 1) -- randomly choose an agents first action
            return $ agent * nrAs + snd (idxs VB.! agent)
    let r = rs VB.! startIdx
    r' <- foldM (flip $ addToReplayMemory nrAs) r mems
    return $! ReplayMemoriesPerActions nrAs (Just tmpRepMem') (rs VB.// [(startIdx, r')])
    else return $! ReplayMemoriesPerActions nrAs (Just tmpRepMem') rs


-- | Add an element to the replay memory. Replaces the oldest elements once the predefined replay memory size is
-- reached.
addToReplayMemory :: NumberOfActions -> Experience -> ReplayMemory -> IO ReplayMemory
addToReplayMemory nrAs e (ReplayMemory vec sz idx maxIdx) = do
  VM.write vec (fromIntegral idx) (force e)
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
  let len = min bs ((1 + maxIdx) `div` nStep)
  g <- newStdGen
  let rands
        | len * nStep == size = take len [nStep - 1,2 * nStep - 1 .. maxIdx]
        | otherwise = take len $ randomRs (nStep - 1, maxIdx) g
      idxes = map (\r -> filter (>= 0) [r - nStep + 1 .. r]) rands
  concat <$> mapM (fmap splitTerminal . mapM (VM.read vec)) idxes
  where
    splitTerminal xs = filter (not . null) $ splitList isTerminal xs
getRandomReplayMemoryElements False nStep bs (ReplayMemory vec size _ maxIdx)
  | nStep > maxIdx + 1 = return []
  | otherwise = do
    let len = min bs ((maxIdx + 1) `div` nStep)
    g <- newStdGen
    let rands
          | len * nStep == size = take len [0,nStep .. maxIdx - nStep + 1]
          | otherwise = map (subtract nStep . (* nStep)) $ take len $ randomRs (1, (maxIdx + 1) `div` nStep) g
        idxes = map (\r -> [r .. r + nStep - 1]) rands
    concat <$> mapM (fmap splitTerminal . mapM (VM.read vec)) idxes
  where
    splitTerminal xs = filter (not . null) $ splitList isTerminal xs

isTerminal :: Experience -> Bool
isTerminal (_, _, _, _, t) = t


splitList :: (a -> Bool) -> [a] -> [[a]]
splitList _ []     = [[]]
splitList f (x:xs) | f x = [x] : splitList f xs
splitList f (x:xs) = (x : y) : ys
  where
    y:ys = splitList f xs


-- | Get a list of random input-output tuples from the replay memory.
getRandomReplayMemoriesElements :: NStep -> Batchsize -> ReplayMemories -> IO [[Experience]]
getRandomReplayMemoriesElements nStep bs (ReplayMemoriesUnified nrAs rm) = getRandomReplayMemoryElements True nStep bs rm
getRandomReplayMemoriesElements nStep bs (ReplayMemoriesPerActions nrAs tmpRepMem rs) = do
  g <- newStdGen
  let idxs = take bs $ randomRs (0, length rs - 1) g
      rsSel = map (rs VB.!) idxs
  concat <$> mapM getRandomReplayMemoriesElements' rsSel
  where getRandomReplayMemoriesElements' replMem
           | nStep == 1 || isNothing tmpRepMem = getRandomReplayMemoryElements False 1 1 replMem
           | otherwise = do
              xs <- getRandomReplayMemoryElements False nStep 1 replMem
              if null xs
                then getRandomReplayMemoryElements False 1 1 (fromJust tmpRepMem)
                else return xs


-- | Size of replay memory (combined if it is a per action replay memory).
replayMemoriesSize :: ReplayMemories -> Int
replayMemoriesSize (ReplayMemoriesUnified _ m)     = m ^. replayMemorySize
replayMemoriesSize (ReplayMemoriesPerActions _ _ ms) = sum $ VB.map (view replayMemorySize) ms

replayMemoriesSubSize :: ReplayMemories -> Int
replayMemoriesSubSize (ReplayMemoriesUnified _ m) = m ^. replayMemorySize
replayMemoriesSubSize (ReplayMemoriesPerActions _ _ xs)
  | VB.null xs = 0
  | otherwise = VB.head xs ^. replayMemorySize
