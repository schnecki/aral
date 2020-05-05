{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE InstanceSigs        #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-cse #-}
module ML.BORL.Types where

import           Control.Monad.Catch
import           Control.Monad.IO.Unlift
import qualified Data.Vector.Storable    as V
import qualified HighLevelTensorflow     as TF

type FilteredActionIndices = V.Vector ActionIndex
type ActionIndex = Int
type IsRandomAction = Bool
type IsOptimalValue = Bool
type ActionFilter s = s -> V.Vector Bool

-- | Agent type. There is only one main agent, but there could be multiple workers (configured via NNConfig).
data AgentType = MainAgent | WorkerAgent Int
  deriving (Show, Read, Eq, Ord)

isMainAgent :: AgentType -> Bool
isMainAgent MainAgent = True
isMainAgent _         = False

isWorkerAgent :: AgentType -> Bool
isWorkerAgent MainAgent = False
isWorkerAgent _         = True


instance Enum AgentType where
  fromEnum MainAgent        = 0
  fromEnum (WorkerAgent nr) = nr
  toEnum nr = (MainAgent : map WorkerAgent [1..]) !! nr

type NStep = Int
type Batchsize = Int
type EpisodeEnd = Bool
type InitialStateFun s = AgentType -> IO s         -- ^ Initial state

liftInitSt :: s -> InitialStateFun s
liftInitSt = const . return

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
type StateActionValuesFiltered = V.Vector Float

type MSE = Float               -- ^ Mean squared error
type MaxValue n = n
type MinValue n = n


replace :: Int -> a -> [a] -> [a]
replace idx val ls = take idx ls ++ val : drop (idx+1) ls

-- | This is to ensure that Tensorflow code stays seperated from non TF Code w/o rquiering huge type inference runs.
--
-- This is BAD and not a Monad anymore!
-- Better refactor!
--

class (MonadIO m) => MonadBorl' m where
  liftTf :: TF.SessionT IO a -> m a

instance (MonadBorl' (TF.SessionT IO)) where
  liftTf = id

instance MonadUnliftIO (TF.SessionT IO) where
  askUnliftIO = return $ UnliftIO runMonadBorlTF

instance (MonadBorl' IO) where
  liftTf = error "You are using the wrong type: IO instead of Tensorflow's SessionT! See runMonadBorlTF and runMonadBorlIO"

liftTensorflow :: (MonadBorl' m) => TF.SessionT IO a -> m a
liftTensorflow = liftTf

runMonadBorlIO :: IO a -> IO a
runMonadBorlIO = id

runMonadBorlTF :: (MonadIO m, MonadMask m) => TF.SessionT m a -> m a
runMonadBorlTF = TF.runSession


