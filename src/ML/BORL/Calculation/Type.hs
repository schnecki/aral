{-# LANGUAGE BangPatterns   #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
{-# LANGUAGE RankNTypes     #-}
module ML.BORL.Calculation.Type where


import           Control.DeepSeq
import           Control.Monad.IO.Class
import           Data.List
import qualified Data.Vector            as VB
import qualified Data.Vector.Storable   as V
import           GHC.Generics

import           ML.BORL.Reward
import           ML.BORL.Types


type ReplMemFun m s
   = (StateFeatures, FilteredActionIndices)
   -> ActionChoice
   -> RewardValue
   -> (StateNextFeatures, FilteredActionIndices)
   -> EpisodeEnd
   -> ExpectedValuationNext
   -> m ( Calculation, ExpectedValuationNext)

data ExpectedValuationNext =
  ExpectedValuationNext
    { getExpectedValStateNextRho :: Maybe Value -- Double
    , getExpectedValStateNextV   :: Maybe Value -- Double
    , getExpectedValStateNextW   :: Maybe Value -- Double
    , getExpectedValStateNextR0  :: Maybe Value -- Double
    , getExpectedValStateNextR1  :: Maybe Value -- Double
    }
  deriving (Show, Generic, NFData)

emptyExpectedValuationNext :: ExpectedValuationNext
emptyExpectedValuationNext = ExpectedValuationNext Nothing Nothing Nothing Nothing Nothing


data Calculation = Calculation
  { getRhoMinimumVal'     :: Maybe Value
  , getRhoVal'            :: Maybe Value
  , getPsiVValState'      :: Maybe Value -- ^ Deviation of this state
  , getVValState'         :: Maybe Value
  , getPsiWValState'      :: Maybe Value -- ^ Deviation of this state
  , getWValState'         :: Maybe Value
  , getR0ValState'        :: Maybe Value
  , getR1ValState'        :: Maybe Value
  , getPsiValRho'         :: Maybe Value -- ^ Scalar deviation over all states (for output only)
  , getPsiValV'           :: Maybe Value -- ^ Scalar deviation over all states (for output only)
  , getPsiValW'           :: Maybe Value -- ^ Scalar deviation over all states (for output only)
  , getLastVs'            :: Maybe (VB.Vector Value)
  , getLastRews'          :: V.Vector RewardValue
  , getEpisodeEnd         :: Bool
  , getExpSmoothedReward' :: Double
  } deriving (Show, Generic)


emptyCalculation :: Calculation
emptyCalculation = Calculation Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing mempty True 0

instance NFData Calculation where
  rnf calc =
    rnf (getRhoMinimumVal' calc) `seq` rnf (getRhoVal' calc) `seq` rnf (getPsiVValState' calc) `seq` rnf (getVValState' calc) `seq` rnf (getPsiWValState' calc) `seq`
    rnf (getWValState' calc) `seq` rnf (getR0ValState' calc) `seq` rnf (getR1ValState' calc) `seq`
    rnf (getPsiValRho' calc) `seq` rnf (getPsiValV' calc) `seq` rnf (getPsiValW' calc) `seq` rnf1 (getLastVs' calc) `seq` rnf (getLastRews' calc) `seq` rnf (getEpisodeEnd calc)


fmapCalculation :: (Double -> Double) -> Calculation -> Calculation
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
--     avg :: [Double] -> Double
--     avg xs' = sum xs' / fromIntegral (length xs')
