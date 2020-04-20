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
import           Data.Serialize
import           GHC.Generics

import           ML.BORL.Action.Type
import           ML.BORL.NeuralNetwork
import           ML.BORL.Reward.Type
import           ML.BORL.RewardFuture
import           ML.BORL.Types


data Workers s =
  Workers
    { _workersS              :: ![s]
    , _workersFutureRewards  :: ![[RewardFutureData s]]
    , _workersReplayMemories :: ![ReplayMemories]
    }
  deriving (Generic)
makeLenses ''Workers

mapWorkers :: (RewardFuture s') => (s -> s') -> (StoreType s -> StoreType s') -> Workers s -> Workers s'
mapWorkers f g (Workers ss fss repMem) = Workers (map f ss) (map (map (mapRewardFutureData f g)) fss) repMem

instance NFData s => NFData (Workers s) where
  rnf (Workers states fut rep) = rnf1 states `seq` rnf1 fut `seq` rnf1 rep
