{-# LANGUAGE DefaultSignatures         #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE PolyKinds                 #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TypeFamilies              #-}


module ML.BORL.Reward.Type where

import           Control.DeepSeq
import           GHC.Generics

import           Data.Serialize

-- type Reward s = Double

type RewardValue = Double


-- ^ A result caused by an action can be a immediate reward, no reward or a future reward determined by the state.
data Reward s
  = Reward RewardValue
  | RewardEmpty
  | (RewardFutureState s) => RewardFuture (Storage s)


instance Num (Reward s) where
  v + w = Reward (rewardValue v + rewardValue w)
  v - w = Reward (rewardValue v - rewardValue w)
  v * w = Reward (rewardValue v * rewardValue w)
  abs v = Reward (abs (rewardValue v))
  signum v = Reward (signum (rewardValue v))
  fromInteger v = Reward (fromIntegral v)

instance Fractional (Reward s) where
  v / w = Reward (rewardValue v / rewardValue w)
  fromRational v = Reward (fromRational v)

rewardValue :: Reward s -> RewardValue
rewardValue (Reward v) = v
rewardValue RewardEmpty = 0
rewardValue (RewardFuture _) = error "reward value of RewardFutureState no known for calculation"

instance NFData (Reward s) where
  rnf (Reward v)       = rnf v
  rnf RewardEmpty      = ()
  rnf (RewardFuture v) = rnf v

instance (RewardFutureState s) => Serialize (Reward s) where
  put (Reward r)       = put (0::Int) >> put r
  put RewardEmpty      = put (1::Int)
  put (RewardFuture f) = put (2::Int) >> put f
  get = do
    (nr :: Int) <- get
    case nr of
      0 -> Reward <$> get
      1 -> return RewardEmpty
      2 -> RewardFuture <$> get
      _ -> error "unmatched case in Serialize instance of Reward"

-- ^ Class that defines the future reward state and storage type.
class (NFData (Storage s), Serialize (Storage s)) => RewardFutureState s where
  type Storage s :: *
  applyState :: Storage s -> s -> Reward s
  mapStorage :: (s -> s') -> Storage s -> Storage s'


mapReward :: (RewardFutureState s') => (s -> s') -> Reward s -> Reward s'
mapReward f (RewardFuture storage) = RewardFuture (mapStorage f storage)
mapReward _ (Reward v)             = Reward v
mapReward _ RewardEmpty            = RewardEmpty
