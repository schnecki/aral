{-# LANGUAGE BangPatterns         #-}
{-# LANGUAGE CPP                  #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE DeriveAnyClass       #-}
{-# LANGUAGE DeriveGeneric        #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE OverloadedStrings    #-}
{-# LANGUAGE PolyKinds            #-}
{-# LANGUAGE Rank2Types           #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TemplateHaskell      #-}
{-# LANGUAGE TupleSections        #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE UndecidableInstances #-}

module ML.BORL.Workers.Type where

import           Control.DeepSeq
import           Control.Lens
import           Data.Either           (isRight)
import           Data.Serialize
import           GHC.Generics

import           ML.BORL.Action.Type
import           ML.BORL.NeuralNetwork
import           ML.BORL.Proxy.Proxies
import           ML.BORL.Reward.Type
import           ML.BORL.RewardFuture
import           ML.BORL.Types

type Workers s = [WorkerState s]

data WorkerState s =
  WorkerState
    { _workerNumber        :: Int                   -- ^ Worker nr.
    , _workerS             :: !s                    -- ^ Current state.
    , _workerReplayMemory  :: !ReplayMemories       -- ^ Replay Memories of worker.
    , _workerFutureRewards :: ![RewardFutureData s] -- ^ Future reward data.
    , _workerExpSmthReward :: Double                 -- ^ Exponentially smoothed reward with rate 0.0001
    }
  deriving (Generic)
makeLenses ''WorkerState

mapWorkers :: (RewardFuture s') => (s -> s') -> (StoreType s -> StoreType s') -> Workers s -> Workers s'
mapWorkers f g = map (mapWorkerState f g)

mapWorkerState :: (RewardFuture s') => (s -> s') -> (StoreType s -> StoreType s') -> WorkerState s -> WorkerState s'
mapWorkerState f g (WorkerState nr s px futs rew) = WorkerState nr (f s) px (map (mapRewardFutureData f g) futs) rew

instance NFData s => NFData (WorkerState s) where
  rnf (WorkerState nr state replMem fut rew) = rnf nr `seq` rnf state `seq` rnf replMem `seq` rnf1 fut `seq` rnf rew
