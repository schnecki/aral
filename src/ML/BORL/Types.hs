{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE InstanceSigs        #-}
{-# LANGUAGE ScopedTypeVariables #-}
module ML.BORL.Types where

import           Control.Monad             (join, liftM)
import           Control.Monad.Catch       (MonadMask)
import           Control.Monad.Identity
import           Control.Monad.IO.Class    (MonadIO, liftIO)
import           Control.Monad.Trans.Class (MonadTrans, lift)
import           System.IO.Unsafe          (unsafePerformIO)
import qualified TensorFlow.Core           as TF
import qualified TensorFlow.Session        as TF

import           Debug.Trace


type Period = Integer
type ActionIndex = Int
type InitialState s = s         -- ^ Initial state
type Batchsize = Int


type MSE = Double               -- ^ Mean squared error
type MaxValue = Double
type MinValue = Double


data MonadBorl a where
  Tensorflow :: TF.Session a -> MonadBorl a
  Pure :: IO a -> MonadBorl a

instance Functor MonadBorl where
  fmap :: (a->b) -> MonadBorl a -> MonadBorl b
  fmap f (Tensorflow action) = Tensorflow (fmap f action)
  fmap f (Pure action)       = Pure (fmap f action)

instance Applicative MonadBorl where
  pure = Pure . pure
  (<*>) :: forall a b . MonadBorl (a -> b) -> MonadBorl a -> MonadBorl b
  (<*>) (Tensorflow fM) (Tensorflow x) = Tensorflow (do f <- fM
                                                        f <$> x)
  (<*>) (Pure f) (Pure x) = Pure (f <*> x)
  (<*>) (Tensorflow fM) (Pure aM) = Tensorflow $ do f <- fM
                                                    lift $ f <$> aM
  (<*>) (Pure fM) (Tensorflow action) = Tensorflow $ do a <- action
                                                        lift $ fM >>= \f -> return $ f a


instance Monad MonadBorl where
  (>>=) :: forall a b. MonadBorl a -> (a -> MonadBorl b) -> MonadBorl b
  Tensorflow a >>= action = Tensorflow $ do aval <- a
                                            case action aval of
                                              Pure x       -> lift x
                                              Tensorflow y -> y
  Pure a >>= action = unsafePerformIO $ fmap action a
  -- runIdentity <$>
  --   (_ $ a >>= \aval -> case action aval of
  --                                Pure x -> x >>= \xval -> return (Pure xval :: MonadBorl Identity b)
  --                                Tensorflow y -> undefined)

-- instance MonadTrans MonadBorl where
--   lift = Tensorflow . lift

-- instance MonadIO (MonadBorl IO) where
--   liftIO = lift


runMonadBorl :: MonadBorl a -> IO a
runMonadBorl (Tensorflow action) = TF.runSession action
runMonadBorl (Pure action)       = action


