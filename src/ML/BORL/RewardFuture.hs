{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}

module ML.BORL.RewardFuture
  ( RewardFutureData (..)
  , futurePeriod
  , futureState
  , futureActionNr
  , futureRandomAction
  , futureReward
  , futureStateNext
  , futureEpisodeEnd
  , mapRewardFutureData
  ) where

import           Control.DeepSeq
import           Control.Lens
import           Data.Serialize
import           GHC.Generics

import           ML.BORL.Reward.Type
import           ML.BORL.Types


-------------------- Main RL Datatype --------------------

data RewardFutureData s = RewardFutureData
                { _futurePeriod       :: Period
                , _futureState        :: State s
                , _futureActionNr     :: ActionIndex
                , _futureRandomAction :: IsRandomAction
                , _futureReward       :: Reward s
                , _futureStateNext    :: StateNext s
                , _futureEpisodeEnd   :: Bool
                } deriving (Generic, NFData, Serialize)
makeLenses ''RewardFutureData


mapRewardFutureData :: (RewardFuture s') => (s -> s') -> (StoreType s -> StoreType s') -> RewardFutureData s -> RewardFutureData s'
mapRewardFutureData f g (RewardFutureData p s aNr rand rew stateNext epEnd) = RewardFutureData p (f s) aNr rand (mapReward g rew) (f stateNext) epEnd


