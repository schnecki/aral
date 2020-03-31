{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE InstanceSigs        #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-cse #-}
module ML.BORL.Types where

import           Control.Monad.Catch
import           Control.Monad.IO.Class    (liftIO)
import           Control.Monad.IO.Class    (liftIO)
import           Control.Monad.IO.Unlift
import           Control.Monad.Trans.Class (lift)
import qualified Data.Vector               as VB
import qualified Data.Vector.Storable      as V
import qualified HighLevelTensorflow       as TF

type FilteredActionIndices = V.Vector ActionIndex
type ActionIndex = Int
type IsRandomAction = Bool
type ActionFilter s = s -> V.Vector Bool


type Batchsize = Int
type EpisodeEnd = Bool
type InitialState s = s         -- ^ Initial state
type Period = Int
type State s = s
type StateNext s = s
type PsisOld = (Float, Float, Float)
type PsisNew = PsisOld
type RewardValue = Float


type FeatureExtractor s = s -> StateFeatures
type GammaLow = Float
type GammaHigh = Float
type GammaMiddle = Float
type Gamma = Float


type StateFeatures = V.Vector Float
type StateNextFeatures = StateFeatures
type NetInputWoAction = StateFeatures
type NetInput = StateFeatures
type NetInputWithAction = StateFeatures
type NetOutput = V.Vector Float


type MSE = Float               -- ^ Mean squared error
type MaxValue n = n
type MinValue n = n


replace :: Int -> a -> [a] -> [a]
replace idx val ls = take idx ls ++ val : drop (idx+1) ls

class (MonadIO m) => MonadBorl' m where
  liftTf :: TF.SessionT IO a -> m a

instance (MonadBorl' (TF.SessionT IO)) where
  liftTf = id

instance MonadUnliftIO (TF.SessionT IO) where
  askUnliftIO = return $ UnliftIO runMonadBorlTF

-- | This is to ensure that Tensorflow code stays seperated from non TF Code w/o rquiering huge type inference runs.
instance (MonadBorl' IO) where
  liftTf = error "You are using the wrong type: IO instead of Tensorflow's SessionT! See runMonadBorlTF and runMonadBorlIO"

liftTensorflow :: (MonadBorl' m) => TF.SessionT IO a -> m a
liftTensorflow = liftTf

runMonadBorlIO :: IO a -> IO a
runMonadBorlIO = id

runMonadBorlTF :: (MonadIO m, MonadMask m) => TF.SessionT m a -> m a
runMonadBorlTF = TF.runSession


