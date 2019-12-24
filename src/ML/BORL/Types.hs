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
import qualified TensorFlow.Core           as TF
import qualified TensorFlow.Session        as TF

type ActionIndex = Int
type Batchsize = Int
type EpisodeEnd = Bool
type InitialState s = s         -- ^ Initial state
type IsRandomAction = Bool
type Period = Int
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

class (MonadIO m) => MonadBorl' m where
  liftTf :: TF.SessionT IO a -> m a

instance (MonadBorl' (TF.SessionT IO)) where
  liftTf = id

instance MonadUnliftIO (TF.SessionT IO) where
  askUnliftIO = return $ UnliftIO runMonadBorlTF

-- | This is to ensure that Tensorflow code stays seperated from non TF Code w/o rquiering huge type inference runs.
instance (MonadBorl' IO) where
  liftTf  =
    -- runMonadBorlTF
    error "You are using the wrong type: IO instead of Tensorflow's SessionT!"

liftTensorflow :: (MonadBorl' m) => TF.SessionT IO a -> m a
liftTensorflow = liftTf

runMonadBorlIO :: IO a -> IO a
runMonadBorlIO = id

-- runMonadBorlTF :: TF.SessionT IO a -> IO a
-- runMonadBorlTF = TF.runSession

runMonadBorlTF :: (MonadIO m, MonadMask m) => TF.SessionT m a -> m a
runMonadBorlTF = TF.runSession


