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
  , getPsiVVal'       :: Maybe Double
  , getVValState'     :: Maybe Double
  , getPsiWVal'       :: Maybe Double
  , getWValState'     :: Maybe Double
  , getR0ValState'    :: Maybe Double
  , getR1ValState'    :: Maybe Double
  , getPsiValRho'     :: Maybe Double
  , getPsiValV'       :: Maybe Double
  , getPsiValW'       :: Maybe Double
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
    (avg <$> mapM getPsiVVal' xs)
    (avg <$> mapM getVValState' xs)
    (avg <$> mapM getPsiWVal' xs)
    (avg <$> mapM getWValState' xs)
    (avg <$> mapM getR0ValState' xs)
    (avg <$> mapM getR1ValState' xs)
    (avg <$> mapM getPsiValRho' xs)
    (avg <$> mapM getPsiValV' xs)
    (avg <$> mapM getPsiValW' xs)
    (mapM (fmap avg . getLastVs') xs)
    (map (avg . getLastRews') xs)
    (all getEpisodeEnd xs)
  where
    avg xs = sum xs / fromIntegral (length xs)