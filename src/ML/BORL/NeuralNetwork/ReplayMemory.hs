{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE GADTs           #-}
{-# LANGUAGE Rank2Types      #-}
{-# LANGUAGE RankNTypes      #-}
{-# LANGUAGE Strict          #-}
{-# LANGUAGE StrictData      #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE ViewPatterns    #-}


module ML.BORL.NeuralNetwork.ReplayMemory where


import           Control.DeepSeq
import           Control.Lens
import           Control.Monad               (foldM)
import           Control.Parallel.Strategies
import           Data.Int
import           Data.Maybe                  (fromJust, isNothing)
import qualified Data.Vector                 as VB
import qualified Data.Vector.Mutable         as VM
import qualified Data.Vector.Storable        as VS
import           Data.Word
import           GHC.Generics
import           System.Random

import           ML.BORL.Types


------------------------------ Replay Memories ------------------------------

type IntX = Int16

shift :: Double
shift = 1000 -- 100 for Int8, 100 or 1000 for Int16


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

-- ^ Experience for the outside world. Internal representation is more memory efficient, see @InternalExperience@. The invariant is that the state features are in (-1,1).
type Experience
   = ( (StateFeatures, DisallowedActionIndicies)     -- ^ State Features s & disallowed actions per agent
     , ActionChoice                                  -- ^ True, iff the action was randomly chosen, actionIndex a
     , RewardValue                                   -- ^ Reward r
     , (StateNextFeatures, DisallowedActionIndicies) -- ^ State features s' & disallowed actions per agent
     , EpisodeEnd                                    -- ^ True, iff it is the end of the episode
      )

-- ^ Internal representation, which is more memory efficient. The invariant is that the state features are in (-1,1).
type InternalExperience
   = ((VS.Vector IntX, Maybe (VB.Vector (VS.Vector Word8))), ActionChoice, RewardValue, (VS.Vector IntX, Maybe (VB.Vector (VS.Vector Word8))), EpisodeEnd, Word8)

-- ^ Convert to internal representation.
toInternal :: Experience -> InternalExperience
toInternal ((stFt, DisallowedActionIndicies dis), act, r, (stFt', DisallowedActionIndicies dis'), eps) = ((VS.map toIntX stFt, toDis dis), act, r, (VS.map toIntX stFt', toDis dis'), eps, fromIntegral $ VB.length dis)
  where
    toDis :: VB.Vector (VS.Vector ActionIndex) -> Maybe (VB.Vector (VS.Vector Word8))
    toDis nas | all VS.null nas = Nothing
              | otherwise = Just $ VB.map (VS.map fromIntegral) nas
    toIntX :: Double -> IntX
    toIntX !x = fromIntegral (round (x * shift) :: Int)


-- ^ Convert from internal representation.
fromInternal :: InternalExperience -> Experience
fromInternal ((stFt, dis), act, r, (stFt', dis'), eps, nrAg) = ((VS.map fromIntX stFt, fromDis dis), act, r, (VS.map fromIntX stFt', fromDis dis'), eps)
  where
    fromDis :: Maybe (VB.Vector (VS.Vector Word8)) -> DisallowedActionIndicies
    fromDis Nothing = DisallowedActionIndicies $ VB.generate (fromIntegral nrAg) (const VS.empty)
    fromDis (Just nas) = DisallowedActionIndicies $ VB.map (VS.map fromIntegral) nas
    fromIntX :: IntX -> Double
    fromIntX !x = fromIntegral x / shift


-------------------- Replay Memory --------------------

data ReplayMemory = ReplayMemory
  { _replayMemoryVector :: !(VM.IOVector InternalExperience) -- ^ Memory
  , _replayMemorySize   :: !Int                              -- ^ Memory size
  , _replayMemoryIdx    :: !Int                              -- ^ Index to use when adding the next element
  , _replayMemoryMaxIdx :: !Int                              -- ^ max index which is in {0,..,size-1}
  }
makeLenses ''ReplayMemory

instance Show ReplayMemory where
  show (ReplayMemory _ sz idx maxIdx) = "Replay Memory with size " <> show sz <> ". Next index: " <> show idx <> "/" <> show maxIdx

instance NFData ReplayMemory where
  rnf (ReplayMemory !vec s idx mx) = rnf s `seq` rnf idx `seq` rnf mx


-- | Add an experience to the Replay Memory.
addToReplayMemories :: NStep -> Experience -> ReplayMemories -> IO ReplayMemories
addToReplayMemories _ e (ReplayMemoriesUnified nr rm) = ReplayMemoriesUnified nr <$> addToReplayMemory nr (toInternal e) rm
addToReplayMemories 1 e@(_, actChoices, _, _, _) (ReplayMemoriesPerActions nrAs tmp rs)
  | VB.length actChoices == 1 = do
    let (_,idx) = VB.head actChoices
    let r = rs VB.! idx
    !r' <- addToReplayMemory nrAs (toInternal e) r
    return $! ReplayMemoriesPerActions nrAs tmp (rs VB.// [(idx, r')])
addToReplayMemories 1 e@(_, actChoices, _, _, _) (ReplayMemoriesPerActions nrAs tmp rs) = do
  agent <- randomRIO (0, length actChoices - 1) -- randomly choose an agent that specifies the action
  let actionIdx = snd $ actChoices VB.! agent
      memIdx = agent * nrAs + actionIdx
  let r = rs VB.! memIdx
  r' <- addToReplayMemory nrAs (toInternal e) r
  return $! ReplayMemoriesPerActions nrAs tmp (rs VB.// [(memIdx, r')])
addToReplayMemories _ e (ReplayMemoriesPerActions nrAs Nothing rs) = addToReplayMemories 1 e (ReplayMemoriesPerActions nrAs Nothing rs) -- cannot use action tmp replay memory, add immediately
addToReplayMemories _ e (ReplayMemoriesPerActions nrAs (Just tmpRepMem) rs) = do
  !tmpRepMem' <- addToReplayMemory nrAs (toInternal e) tmpRepMem
  if tmpRepMem' ^. replayMemoryIdx == 0
    then do -- temporary replay memory full, add experience to corresponding action memory
      let !vec = tmpRepMem' ^. replayMemoryVector
      !mems <- mapM (VM.read vec) [0 .. tmpRepMem' ^. replayMemorySize - 1]
      !startIdx <-
        case head mems ^. _2 of
          idxs
            | VB.length idxs == 1 -> return (snd $ VB.head idxs)
          idxs -> do
            agent <- randomRIO (0, length idxs - 1) -- randomly choose an agents first action
            return $! agent * nrAs + snd (idxs VB.! agent)
      let !r = rs VB.! startIdx
      !r' <- foldM (flip $ addToReplayMemory nrAs) r mems
      return $! ReplayMemoriesPerActions nrAs (Just tmpRepMem') (rs VB.// [(startIdx, r')])
    else return $! ReplayMemoriesPerActions nrAs (Just tmpRepMem') rs


-- | Add an element to the replay memory. Replaces the oldest elements once the predefined replay memory size is reached.
addToReplayMemory :: NumberOfActions -> InternalExperience -> ReplayMemory -> IO ReplayMemory
addToReplayMemory nrAs (force -> !e) (ReplayMemory vec sz idx maxIdx) = do
  VM.write vec (fromIntegral idx) e
  return $! ReplayMemory vec sz ((idx+1) `mod` fromIntegral sz) (min (maxIdx+1) (sz-1))


type AllExpAreConsecutive = Bool -- ^ Indicates whether all experiences are consecutive.

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
  map (return . fromInternal) <$> mapM (VM.read vec) rands
getRandomReplayMemoryElements True nStep bs (ReplayMemory vec size _ maxIdx) = do -- get consequitive experiences
  let len = min bs ((1 + maxIdx) `div` nStep)
  g <- newStdGen
  let rands
        | len * nStep == size = take len [nStep - 1,2 * nStep - 1 .. maxIdx]
        | otherwise = take len $ randomRs (nStep - 1, maxIdx) g
      idxes = map (\r -> filter (>= 0) [r - nStep + 1 .. r]) rands
  concat <$> mapM (fmap splitTerminal . mapM (VM.read vec)) idxes
  where
    splitTerminal xs = filter (not . null) $ splitList isTerminal (map fromInternal xs)
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
    splitTerminal xs = filter (not . null) $ splitList isTerminal (map fromInternal xs)

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
  concat <$> sequence (parMap rpar getRandomReplayMemoriesElements' rsSel)
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
