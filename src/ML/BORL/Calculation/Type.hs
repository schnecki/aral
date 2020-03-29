{-# LANGUAGE BangPatterns   #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
{-# LANGUAGE RankNTypes     #-}
module ML.BORL.Calculation.Type where


import           Control.DeepSeq
import           GHC.Generics

import           ML.BORL.Reward
import           ML.BORL.Types


type ReplMemFun s
   = forall m. (MonadBorl' m) =>
                 (StateFeatures, FilteredActionIndices) -> ActionIndex -> Bool -> RewardValue -> (StateNextFeatures, FilteredActionIndices) -> EpisodeEnd -> m Calculation


data Calculation = Calculation
  { getRhoMinimumVal' :: !(Maybe Float)
  , getRhoVal'        :: !(Maybe Float)
  , getPsiVValState'  :: !(Maybe Float) -- ^ Deviation of this state
  , getVValState'     :: !(Maybe Float)
  , getPsiWValState'  :: !(Maybe Float) -- ^ Deviation of this state
  , getWValState'     :: !(Maybe Float)
  , getR0ValState'    :: !(Maybe Float)
  , getR1ValState'    :: !(Maybe Float)
  , getPsiValRho'     :: !(Maybe Float)  -- ^ Scalar deviation over all states (for output only)
  , getPsiValV'       :: !(Maybe Float)  -- ^ Scalar deviation over all states (for output only)
  , getPsiValW'       :: !(Maybe Float) -- ^ Scalar deviation over all states (for output only)
  , getLastVs'        :: !(Maybe [Float])
  , getLastRews'      :: ![RewardValue]
  , getEpisodeEnd     :: !Bool
  } deriving (Show, Generic, NFData)

fmapCalculation :: (Float -> Float) -> Calculation -> Calculation
fmapCalculation f calc =
  Calculation
    { getRhoMinimumVal' = f <$> getRhoMinimumVal' calc
    , getRhoVal'        = f <$> getRhoVal' calc
    , getPsiVValState'  = f <$> getPsiVValState' calc
    , getVValState'     = f <$> getVValState' calc
    , getPsiWValState'  = f <$> getPsiWValState' calc
    , getWValState'     = f <$> getWValState' calc
    , getR0ValState'    = f <$> getR0ValState' calc
    , getR1ValState'    = f <$> getR1ValState' calc
    , getPsiValRho'     = f <$> getPsiValRho' calc
    , getPsiValV'       = f <$> getPsiValV' calc
    , getPsiValW'       = f <$> getPsiValW' calc
    , getLastVs'        = getLastVs' calc
    , getLastRews'      = getLastRews' calc
    , getEpisodeEnd     = getEpisodeEnd calc
    }

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
