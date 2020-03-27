{-# LANGUAGE DefaultSignatures         #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE FlexibleInstances         #-}
{-# LANGUAGE MultiParamTypeClasses     #-}
{-# LANGUAGE PolyKinds                 #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TypeFamilies              #-}
{-# LANGUAGE UndecidableInstances      #-}


module ML.BORL.Reward.Type where

import           Control.DeepSeq
import           GHC.Generics

import           Data.ByteString
import           Data.Serialize

type RewardValue = Double


-- ^ A result caused by an action can be a immediate reward, no reward or a future reward determined by the state.
data Reward s
  = Reward RewardValue
  | RewardEmpty
  | (RewardFuture s) => RewardFuture (StoreType s) (StoreType s)

isRewardFuture :: Reward s -> Bool
isRewardFuture RewardFuture{} = True
isRewardFuture _              = False

isRewardEmpty :: Reward s -> Bool
isRewardEmpty RewardEmpty{} = True
isRewardEmpty _             = False

-- ^ Class that defines the future reward state and storage type.
class (NFData (StoreType s), Serialize (StoreType s)) => RewardFuture s where
  type StoreType s :: *
  applyState :: StoreType s -> s -> Reward s


instance NFData (Reward s) where
  rnf (Reward v)          = rnf v
  rnf RewardEmpty         = ()
  rnf (RewardFuture v vw) = rnf v `seq` rnf vw

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


instance RewardFuture s => Serialize (Reward s) where
  put (Reward r)         = put (0::Int) >> put r
  put RewardEmpty        = put (1::Int)
  put (RewardFuture f w) = put (2::Int) >> put f >> put w
  get = do
    (nr :: Int) <- get
    case nr of
      0 -> Reward <$> get
      1 -> return RewardEmpty
      2 -> RewardFuture <$> get <*> get
      _ -> error "unmatched case in Serialize instance of Reward"


-------------------- Helpers --------------------

rewardValue :: Reward s -> RewardValue
rewardValue (Reward v) = v
rewardValue RewardEmpty = 0
rewardValue (RewardFuture _ _) = error "reward value of RewardFutureState not known for calculation"


mapReward :: (RewardFuture s') => (StoreType s -> StoreType s') -> Reward s -> Reward s'
mapReward f (RewardFuture storage storageWorkers) = RewardFuture (f storage) (f storageWorkers)
mapReward _ (Reward v)             = Reward v
mapReward _ RewardEmpty            = RewardEmpty
