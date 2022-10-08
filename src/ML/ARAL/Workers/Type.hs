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

module ML.ARAL.Workers.Type where

import           Control.DeepSeq
import           Control.Lens
import           Data.Either           (isRight)
import           Data.Serialize
import qualified Data.Vector           as VB
import           GHC.Generics

import           ML.ARAL.Action.Type
import           ML.ARAL.NeuralNetwork
import           ML.ARAL.Proxy.Proxies
import           ML.ARAL.Reward.Type
import           ML.ARAL.RewardFuture
import           ML.ARAL.Types

type Workers s = [WorkerState s]

data WorkerState s =
  WorkerState
    { _workerNumber        :: !Int                              -- ^ Worker nr. >= 1.
    , _workerS             :: !s                                -- ^ Current state.
    , _workerReplayMemory  :: !ReplayMemories                   -- ^ Replay Memories of worker.
    , _workerFutureRewards :: !(VB.Vector (RewardFutureData s)) -- ^ Future reward data.
    , _workerExpSmthReward :: !Double                           -- ^ Exponentially smoothed reward with rate 0.0001
    }
  deriving (Generic)
makeLenses ''WorkerState

mapWorkers :: (RewardFuture s') => (s -> s') -> (StoreType s -> StoreType s') -> Workers s -> Workers s'
mapWorkers f g = map (mapWorkerState f g)

mapWorkerState :: (RewardFuture s') => (s -> s') -> (StoreType s -> StoreType s') -> WorkerState s -> WorkerState s'
mapWorkerState f g (WorkerState nr s px futs rew) = WorkerState nr (f s) px (VB.map (mapRewardFutureData f g) futs) rew

instance NFData s => NFData (WorkerState s) where
  rnf (WorkerState nr state replMem fut rew) = rnf nr `seq` rnf state `seq` rnf replMem `seq` rnf1 fut `seq` rnf rew
