{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
{-# LANGUAGE RankNTypes     #-}
module ML.BORL.Calculation.Type where

import           ML.BORL.Reward
import           ML.BORL.Types

import           Control.DeepSeq
import           GHC.Generics


type ReplMemFun s
   = forall m. (MonadBorl' m) =>
                 (StateFeatures, [ActionIndex]) -> ActionIndex -> Bool -> RewardValue -> (StateNextFeatures, [ActionIndex]) -> EpisodeEnd -> m Calculation


data Calculation = Calculation
  { getRhoMinimumVal' :: Maybe Double
  , getRhoVal'        :: Maybe Double
  , getPsiVValState'  :: Maybe Double -- ^ Deviation of this state
  , getVValState'     :: Maybe Double
  , getPsiWValState'  :: Maybe Double -- ^ Deviation of this state
  , getWValState'     :: Maybe Double
  , getPsiW2ValState' :: Maybe Double -- ^ Deviation of this state
  , getW2ValState'    :: Maybe Double
  , getR0ValState'    :: Maybe Double
  , getR1ValState'    :: Maybe Double
  , getPsiValRho'     :: Maybe Double  -- ^ Scalar deviation over all states (for output only)
  , getPsiValV'       :: Maybe Double  -- ^ Scalar deviation over all states (for output only)
  , getPsiValW'       :: Maybe Double -- ^ Scalar deviation over all states (for output only)
  , getPsiValW2'      :: Maybe Double -- ^ Scalar deviation over all states (for output only)
  , getLastVs'        :: Maybe [Double]
  , getLastRews'      :: [RewardValue]
  , getEpisodeEnd     :: Bool
  } deriving (Show, Generic, NFData)


avgCalculation :: [Calculation] -> Calculation
avgCalculation [] = error "empty calculations. Is the batchsize>0 and replayMemoryMaxSize>0?"
avgCalculation xs =
  Calculation
    (avg <$> mapM getRhoMinimumVal' xs)
    (avg <$> mapM getRhoVal' xs)
    (avg <$> mapM getPsiVValState' xs)
    (avg <$> mapM getVValState' xs)
    (avg <$> mapM getPsiWValState' xs)
    (avg <$> mapM getWValState' xs)
    (avg <$> mapM getPsiW2ValState' xs)
    (avg <$> mapM getW2ValState' xs)
    (avg <$> mapM getR0ValState' xs)
    (avg <$> mapM getR1ValState' xs)
    (avg <$> mapM getPsiValRho' xs)
    (avg <$> mapM getPsiValV' xs)
    (avg <$> mapM getPsiValW' xs)
    (avg <$> mapM getPsiValW2' xs)
    (mapM (fmap avg . getLastVs') xs)
    (map (avg . getLastRews') xs)
    (all getEpisodeEnd xs)
  where
    avg xs = sum xs / fromIntegral (length xs)
