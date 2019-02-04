{-# LANGUAGE GADTs               #-}
{-# LANGUAGE InstanceSigs        #-}
{-# LANGUAGE ScopedTypeVariables #-}
module ML.BORL.Types where

import           Control.Monad             (join, liftM)
import           Control.Monad.Catch       (MonadMask)
import           Control.Monad.IO.Class    (MonadIO, liftIO)
import           Control.Monad.Trans.Class (lift)
import qualified TensorFlow.Core           as TF
import qualified TensorFlow.Session        as TF


type Period = Integer
type ActionIndex = Int
type InitialState s = s         -- ^ Initial state
type Batchsize = Int


type MSE = Double               -- ^ Mean squared error
type MaxValue = Double
type MinValue = Double


data MonadBorl m a where
  Tensorflow :: (MonadIO m) => TF.SessionT m a -> MonadBorl m a
  Pure :: (MonadIO m) => m a -> MonadBorl m a

instance Functor (MonadBorl m) where
  fmap f (Tensorflow action) = Tensorflow (fmap f action)
  fmap f (Pure action)       = Pure (fmap f action)

instance (MonadIO m) => Applicative (MonadBorl m) where
  pure x = Pure (pure x)
  (<*>) :: forall a b . MonadBorl m (a -> b) -> MonadBorl m a -> MonadBorl m b
  (<*>) (Tensorflow fM) (Tensorflow x) = Tensorflow (do f <- fM
                                                        f <$> x)
  (<*>) (Pure f) (Pure x) = Pure (f <*> x)
  (<*>) (Tensorflow fM) (Pure aM) = Tensorflow $ do f <- fM
                                                    lift $ f <$> aM
  (<*>) (Pure fM) (Tensorflow action) = Tensorflow $ do a <- action
                                                        lift $ fM >>= \f -> return $ f a


  -- liftA2 :: forall a b c . (a -> b -> c) -> MonadBorl m a -> MonadBorl m b -> MonadBorl m c
  -- liftA2 f (Tensorflow a) (Tensorflow b) = Tensorflow (liftA2 f a b)
  -- liftA2 f (Pure a) (Pure b)             = Pure (liftA2 f a b)
  -- -- liftA2 f (Tensorflow a) (Pure b)       = Tensorflow (liftA (`f` b) a)
  -- liftA2 f (Pure a) (Tensorflow b)       = Tensorflow (liftA2 f (a :: TF.SessionT m a) b)

instance (MonadMask m, MonadIO m) => Monad (MonadBorl m) where
  (>>=) :: forall a b. MonadBorl m a -> (a -> MonadBorl m b) -> MonadBorl m b
  (>>=) (Tensorflow a) action = Tensorflow $ do aval <- a
                                                case action aval of
                                                  Pure x       -> lift x
                                                  Tensorflow y -> y

  -- Is there a better way than executing TF.runSession?
  (>>=) (Pure a) action = Pure $ a >>= runMonadBorl . action


instance TF.Session (MonadBorl m) where


runMonadBorl :: (MonadMask m, MonadIO m) => MonadBorl m a -> m a
runMonadBorl (Tensorflow action) = TF.runSession action
runMonadBorl (Pure action)       = action


