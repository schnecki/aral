module ML.BORL.Calculation where

import           ML.BORL.Types

type ReplMemFun s = s -> ActionIndex -> Bool -> Reward -> s -> MonadBorl Calculation


data Calculation = Calculation
  { getRhoVal'        :: Double
  , getRhoMinimumVal' :: Double
  , getPsiVTblVal'    :: Double
  , getPsiWTblVal'    :: Double
  , getVValStateNew   :: Double
  , getWValState'     :: Double
  , getR0ValState'    :: Double
  , getR1ValState'    :: Double
  , getPsiValRho'     :: Double
  , getPsiValV'       :: Double
  , getPsiValW'       :: Double
  , getLastVs'        :: [Double]
  , getLastRews'      :: [Reward]
  }


