{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE TemplateHaskell #-}


module ML.BORL.NeuralNetwork.ReplayMemory where

import           ML.BORL.Types

import           Control.DeepSeq
import           Control.Lens
import qualified Data.Vector.Mutable as V
import           System.IO.Unsafe
import           System.Random


data ReplayMemory k = ReplayMemory
  { _replayMemoryVector :: V.IOVector (k, Double)
  , _replayMemorySize   :: Int
  }
makeLenses ''ReplayMemory

instance (NFData k) => NFData (ReplayMemory k) where
  rnf (ReplayMemory !_ s) = rnf s


-- | Function to create a new replay memory.
mkReplayMemory :: Int -> ReplayMemory k
mkReplayMemory sz =
  unsafePerformIO $ do
    vec <- V.unsafeNew sz -- does not initialize memory
    return $ ReplayMemory vec sz


-- | Add an element to the replay memory. Replaces the oldest elements once the predefined replay memory size is
-- reached.
addToReplayMemory :: Period -> (k,Double) -> ReplayMemory k -> IO (ReplayMemory k)
addToReplayMemory p e (ReplayMemory vec sz) = do
  let idx = p `mod` fromIntegral sz
  -- putStrLn ("Add to " ++ show idx)
  V.write vec (fromIntegral idx) e
  return $ ReplayMemory vec sz

-- | Get a list of random input-output tuples from the replay memory.
getRandomReplayMemoryElements :: Period -> Batchsize -> ReplayMemory k -> IO [(k, Double)]
getRandomReplayMemoryElements t bs (ReplayMemory vec sz) = do
  let maxIdx = fromIntegral (min t (fromIntegral sz-1)) :: Int
  let len = min bs (maxIdx + 1)
  -- putStrLn $ "t: " ++ show t
  -- putStrLn $ "sz: " ++ show sz
  -- putStrLn $ "Get from " ++ show (0,maxIdx)

  g <- newStdGen
  let rands = take len $ randomRs (0,maxIdx) g
  -- putStrLn $ "Rands: " ++ show rands
  mapM (V.read vec) rands
