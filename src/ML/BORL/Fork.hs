{-# LANGUAGE BangPatterns #-}
module ML.BORL.Fork
    ( doFork
    , doForkFake
    , collectForkResult
    , IORef
    , ThreadState (..)
    ) where

import           Control.Concurrent          (forkIO, yield)
import           Control.DeepSeq
import           Control.Monad               (void)
import           Control.Parallel.Strategies
import           Data.IORef


doFork :: NFData a => IO a -> IO (IORef (ThreadState a))
doFork ~f = do
  ref <- newIORef NotReady
  -- void $ forkIO (f >>= writeIORef ref . Ready . whnf)
  void $ forkIO (f >>= writeIORef ref . Ready . force)
  return ref
  where whnf !a = a

-- | Does not actually fork, thus runs sequentially, but does not force the result!
doForkFake :: IO a -> IO (IORef (ThreadState a))
doForkFake f = do
  ref <- newIORef NotReady
  (f >>= writeIORef ref . Ready) `using` rparWith rpar
  return ref


collectForkResult :: IORef (ThreadState a) -> IO a
collectForkResult !ref = do
  mRes <- readIORef ref
  case mRes of
    NotReady -> yield >> collectForkResult ref
    Ready a  -> return a

data ThreadState a = NotReady | Ready !a
