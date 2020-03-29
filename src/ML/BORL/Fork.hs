{-# LANGUAGE BangPatterns #-}
module ML.BORL.Fork
    ( doFork
    , collectForkResult
    ) where

import           Control.Concurrent (forkIO, yield)
import           Control.DeepSeq
import           Control.Monad      (void)
import           Data.IORef


doFork :: NFData a => IO a -> IO (IORef (ThreadState a))
doFork !f = do
  ref <- newIORef NotReady
  void $ forkIO (f >>= writeIORef ref . Ready . force)
  return ref

collectForkResult :: IORef (ThreadState a) -> IO a
collectForkResult !ref = do
  mRes <- readIORef ref
  case mRes of
    NotReady -> yield >> collectForkResult ref
    Ready a  -> return a

data ThreadState a = NotReady | Ready !a
