{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
module ML.BORL.Calculation where

import           ML.BORL.Types

import           Control.DeepSeq
import           GHC.Generics

type ReplMemFun s = s -> ActionIndex -> Bool -> Reward -> s -> MonadBorl Calculation


data Calculation = Calculation
  { getRhoMinimumVal' :: Double
  , getRhoVal'        :: Double
  , getPsiVVal'       :: Double
  , getPsiWVal'       :: Double
  , getVValState'     :: Double
  , getWValState'     :: Double
  , getR0ValState'    :: Double
  , getR1ValState'    :: Double
  , getPsiValRho'     :: Double
  , getPsiValV'       :: Double
  , getPsiValW'       :: Double
  , getLastVs'        :: [Double]
  , getLastRews'      :: [Reward]
  } deriving (Generic, NFData)


avgCalculation :: [Calculation] -> Calculation
avgCalculation [] = error "empty calculations. Is the batchsize>0 and replayMemoryMaxSize>0?"
avgCalculation xs =
  Calculation
    (avg $ map getRhoMinimumVal' xs)
    (avg $ map getRhoVal' xs)
    (avg $ map getPsiVVal' xs)
    (avg $ map getPsiWVal' xs)
    (avg $ map getVValState' xs)
    (avg $ map getWValState' xs)
    (avg $ map getR0ValState' xs)
    (avg $ map getR1ValState' xs)
    (avg $ map getPsiValRho' xs)
    (avg $ map getPsiValV' xs)
    (avg $ map getPsiValW' xs)
    (map (avg . getLastVs') xs)
    (map (avg . getLastRews') xs)
  where
    avg xs = sum xs / fromIntegral (length xs)
