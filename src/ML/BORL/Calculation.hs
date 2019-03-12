{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
module ML.BORL.Calculation where

import           ML.BORL.Algorithm
import           ML.BORL.Types

import           Control.DeepSeq
import           GHC.Generics


type ReplMemFun s = s -> ActionIndex -> Bool -> Reward -> s -> EpisodeEnd -> MonadBorl Calculation


data Calculation = Calculation
  { getRhoMinimumVal' :: Maybe Double
  , getRhoVal'        :: Maybe Double
  , getPsiVVal'       :: Maybe Double
  , getVValState'     :: Maybe Double
  , getWValState'     :: Maybe Double
  , getR0ValState'    :: Maybe Double
  , getR1ValState'    :: Double
  , getPsiValRho'     :: Maybe Double
  , getPsiValV'       :: Maybe Double
  , getPsiValW'       :: Maybe Double
  , getLastVs'        :: Maybe [Double]
  , getLastRews'      :: [Reward]
  } deriving (Generic, NFData)


avgCalculation :: [Calculation] -> Calculation
avgCalculation [] = error "empty calculations. Is the batchsize>0 and replayMemoryMaxSize>0?"
avgCalculation xs =
  Calculation
    (avg <$> mapM getRhoMinimumVal' xs)
    (avg <$> mapM getRhoVal' xs)
    (avg <$> mapM getPsiVVal' xs)
    (avg <$> mapM getVValState' xs)
    (avg <$> mapM getWValState' xs)
    (avg <$> mapM getR0ValState' xs)
    (avg $ map getR1ValState' xs)
    (avg <$> mapM getPsiValRho' xs)
    (avg <$> mapM getPsiValV' xs)
    (avg <$> mapM getPsiValW' xs)
    (mapM (fmap avg . getLastVs') xs)
    (map (avg . getLastRews') xs)
  where
    avg xs = sum xs / fromIntegral (length xs)
