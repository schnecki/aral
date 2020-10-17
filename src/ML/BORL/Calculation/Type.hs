{-# LANGUAGE BangPatterns   #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
{-# LANGUAGE RankNTypes     #-}
module ML.BORL.Calculation.Type where


import           Control.DeepSeq
import           Control.Monad.IO.Class
import           Data.List
import           GHC.Generics

import           ML.BORL.Reward
import           ML.BORL.Types


type ReplMemFun m s
   = (StateFeatures, FilteredActionIndices)
   -> [ActionIndex]
   -> Bool
   -> RewardValue
   -> (StateNextFeatures, FilteredActionIndices)
   -> EpisodeEnd
   -> ExpectedValuationNext
   -> m ( Calculation, ExpectedValuationNext)

data ExpectedValuationNext =
  ExpectedValuationNext
    { getExpectedValStateNextRho :: Maybe Value -- Float
    , getExpectedValStateNextV   :: Maybe Value -- Float
    , getExpectedValStateNextW   :: Maybe Value -- Float
    , getExpectedValStateNextR0  :: Maybe Value -- Float
    , getExpectedValStateNextR1  :: Maybe Value -- Float
    }
  deriving (Show, Generic, NFData)

emptyExpectedValuationNext :: ExpectedValuationNext
emptyExpectedValuationNext = ExpectedValuationNext Nothing Nothing Nothing Nothing Nothing


data Calculation = Calculation
  { getRhoMinimumVal'     :: Maybe Value
  , getRhoVal'            :: Maybe Value
  , getPsiVValState'      :: Maybe Value -- [Float] -- ^ Deviation of this state
  , getVValState'         :: Maybe Value -- [Float]
  , getPsiWValState'      :: Maybe Value -- [Float] -- ^ Deviation of this state
  , getWValState'         :: Maybe Value -- [Float]
  , getR0ValState'        :: Maybe Value -- [Float]
  , getR1ValState'        :: Maybe Value -- [Float]
  , getPsiValRho'         :: Maybe Value -- Float -- ^ Scalar deviation over all states (for output only)
  , getPsiValV'           :: Maybe Value -- [Float] -- ^ Scalar deviation over all states (for output only)
  , getPsiValW'           :: Maybe Value -- [Float] -- ^ Scalar deviation over all states (for output only)
  , getLastVs'            :: Maybe [Value]
  , getLastRews'          :: [RewardValue]
  , getEpisodeEnd         :: Bool
  , getExpSmoothedReward' :: Float
  } deriving (Show, Generic)


emptyCalculation :: Calculation
emptyCalculation = Calculation Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing [] True 0

instance NFData Calculation where
  rnf calc =
    rnf (getRhoMinimumVal' calc) `seq` rnf (getRhoVal' calc) `seq` rnf (getPsiVValState' calc) `seq` rnf (getVValState' calc) `seq` rnf (getPsiWValState' calc) `seq`
    rnf (getWValState' calc) `seq` rnf (getR0ValState' calc) `seq` rnf (getR1ValState' calc) `seq`
    rnf (getPsiValRho' calc) `seq` rnf (getPsiValV' calc) `seq` rnf (getPsiValW' calc) `seq` rnf1 (getLastVs' calc) `seq` rnf1 (getLastRews' calc) `seq` rnf (getEpisodeEnd calc)


fmapCalculation :: (Float -> Float) -> Calculation -> Calculation
fmapCalculation f calc =
  Calculation
    { getRhoMinimumVal'       = mapValue f <$> getRhoMinimumVal' calc
    , getRhoVal'              = mapValue f <$> getRhoVal' calc
    , getPsiVValState'        = mapValue f <$> getPsiVValState' calc
    , getVValState'           = mapValue f <$> getVValState' calc
    , getPsiWValState'        = mapValue f <$> getPsiWValState' calc
    , getWValState'           = mapValue f <$> getWValState' calc
    , getR0ValState'          = mapValue f <$> getR0ValState' calc
    , getR1ValState'          = mapValue f <$> getR1ValState' calc
    , getPsiValRho'           = mapValue f <$> getPsiValRho' calc
    , getPsiValV'             = mapValue f <$> getPsiValV' calc
    , getPsiValW'             = mapValue f <$> getPsiValW' calc
    , getLastVs'              = getLastVs' calc
    , getLastRews'            = getLastRews' calc
    , getEpisodeEnd           = getEpisodeEnd calc
    , getExpSmoothedReward'   = getExpSmoothedReward' calc
    }

-- avgCalculation :: [Calculation] -> Calculation
-- avgCalculation [] = error "empty calculations. Is the batchsize>0 and replayMemoryMaxSize>0?"
-- avgCalculation xs =
--   Calculation
--     (avg <$> mapM getRhoMinimumVal' xs)
--     (avg <$> mapM getRhoVal' xs)
--     (map avg <$> mapM getPsiVValState' xs)
--     (map avg <$> mapM getVValState' xs)
--     (map avg <$> mapM getPsiWValState' xs)
--     (map avg <$> mapM getWValState' xs)
--     (map avg <$> mapM getR0ValState' xs)
--     (map avg <$> mapM getR1ValState' xs)
--     (avg <$> mapM getPsiValRho' xs)
--     (map avg <$> mapM getPsiValV' xs)
--     (map avg <$> mapM getPsiValW' xs)
--     (map (fmap avg . transpose) <$> mapM getLastVs' xs)
--     (map (avg . getLastRews') xs)
--     (all getEpisodeEnd xs)
--     (avg $ map getExpSmoothedReward' xs)
--   where
--     avg :: [Float] -> Float
--     avg xs' = sum xs' / fromIntegral (length xs')
