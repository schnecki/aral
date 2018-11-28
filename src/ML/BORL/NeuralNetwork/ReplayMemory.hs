{-# LANGUAGE TemplateHaskell #-}


module ML.BORL.NeuralNetwork.ReplayMemory where

import           ML.BORL.Types

import           Control.Lens
import qualified Data.Vector.Mutable as V
import           System.Random

import           System.IO.Unsafe


data ReplayMemory k = ReplayMemory
  { _replayMemoryVector :: V.IOVector (k, Double)
  , _replayMemoryIndex  :: Int
  , _replayMemorySize   :: Int
  }
makeLenses ''ReplayMemory

-- | Function to create a new replay memory.
mkReplayMemory :: Int -> ReplayMemory k
mkReplayMemory sz =
  unsafePerformIO $ do
    vec <- V.new sz -- does not initialize memory
    return $ ReplayMemory vec 0 sz


-- | Add an element to the replay memory. Replaces the oldest elements once the predefined replay memory size is
-- reached.
addToReplayMemory :: (k,Double) -> ReplayMemory k -> ReplayMemory k
addToReplayMemory e (ReplayMemory vec idx sz) =
  unsafePerformIO $ do
    V.write vec idx e
    return $ ReplayMemory vec ((idx + 1) `mod` sz) sz

-- | Get a list of random input-output tuples from the replay memory.
getRandomReplayMemoryElements :: Period -> Batchsize -> ReplayMemory k -> [(k, Double)]
getRandomReplayMemoryElements t bs (ReplayMemory vec _ sz) =
  unsafePerformIO $ do
    let maxIdx = fromIntegral (min t (fromIntegral sz)) - 1 :: Int
    let len = min bs (maxIdx + 1)
    g <- newStdGen
    let rands = take len $ randomRs (0, maxIdx) g
    mapM (V.read vec) rands
