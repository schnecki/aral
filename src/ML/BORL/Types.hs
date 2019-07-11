{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE InstanceSigs        #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-cse #-}
module ML.BORL.Types where

import           Control.Monad.IO.Class
import           Control.Monad.IO.Unlift
import           Control.Monad.Trans.Class (lift)
import           System.IO.Unsafe          (unsafePerformIO)
import qualified TensorFlow.Core           as TF
import qualified TensorFlow.Session        as TF

type ActionIndex = Int
type Batchsize = Int
type EpisodeEnd = Bool
type InitialState s = s         -- ^ Initial state
type IsRandomAction = Bool
type Period = Integer
type State s = s
type StateNext s = s
type PsisOld = (Double, Double, Double)
type PsisNew = PsisOld

type FeatureExtractor s = s -> [Double]
type GammaLow = Double
type GammaHigh = Double
type Gamma = Double


type StateFeatures = [Double]
type StateNextFeatures = [Double]


type MSE = Double               -- ^ Mean squared error
type MaxValue = Double
type MinValue = Double


replace :: Int -> a -> [a] -> [a]
replace idx val ls = take idx ls ++ val : drop (idx+1) ls


class (Monad m) => MonadBorl' m where
  liftTf :: TF.SessionT IO a -> m a
  liftSimple :: IO a -> m a

instance (MonadBorl' (TF.SessionT IO)) where
  liftTf = id
  liftSimple = lift

instance (MonadBorl' IO) where
  liftTf _ = error "You are using the wrong type: IO instead of Tensorflow's SessionT!"
  liftSimple = id

liftTensorflow :: (MonadBorl' m) => TF.SessionT IO a -> m a
liftTensorflow = liftTf

runMonadBorlIO :: IO a -> IO a
runMonadBorlIO = id

runMonadBorlTF :: TF.SessionT IO a -> IO a
runMonadBorlTF = TF.runSession


-- runMonadBorl (Tensorflow action) = TF.runSession action
-- runMonadBorl (Simple action)     = action


-- -- ^ Monad that distinguished between Simple (Grenade, Table) methods and Tensorflow sessions.
-- data MonadBorl a where
--   Tensorflow :: TF.SessionT IO a -> MonadBorl a
--   Simple :: IO a -> MonadBorl a

-- instance Functor MonadBorl where
--   fmap :: (a->b) -> MonadBorl a -> MonadBorl b
--   fmap f (Tensorflow action) = Tensorflow (fmap f action)
--   fmap f (Simple action)     = Simple (fmap f action)

-- instance Applicative MonadBorl where
--   pure = Simple . pure
--   (<*>) :: forall a b . MonadBorl (a -> b) -> MonadBorl a -> MonadBorl b
--   (<*>) (Tensorflow fM) (Tensorflow x) = Tensorflow (do f <- fM
--                                                         f <$> x)
--   (<*>) (Simple f) (Simple x) = Simple (f <*> x)
--   (<*>) (Tensorflow fM) (Simple aM) = Tensorflow $ do f <- fM
--                                                       lift $ f <$> aM
--   (<*>) (Simple fM) (Tensorflow action) = Tensorflow $ do a <- action
--                                                           lift $ fM >>= \f -> return $ f a


-- instance Monad MonadBorl where
--   (>>=) :: forall a b. MonadBorl a -> (a -> MonadBorl b) -> MonadBorl b
--   Tensorflow a >>= action = Tensorflow $ do !aval <- a
--                                             case action aval of
--                                               Simple !x    -> lift x
--                                               Tensorflow !y-> y
--   Simple !a >>= (!action) = unsafePerformIO $ fmap action a
--   {-# NOINLINE (>>=) #-}


-- runMonadBorl :: MonadBorl a -> IO a
-- runMonadBorl (Tensorflow action) = TF.runSession action
-- runMonadBorl (Simple action)     = action


-- instance MonadIO MonadBorl where
--   liftIO = Simple

-- instance MonadUnliftIO MonadBorl where
--   askUnliftIO = return $ UnliftIO runMonadBorl

instance MonadUnliftIO (TF.SessionT IO) where
  askUnliftIO = return $ UnliftIO runMonadBorlTF


