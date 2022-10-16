{-# LANGUAGE DeriveAnyClass    #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE OverloadedStrings #-}
module ML.ARAL.Proxy.Regression.Observation
    ( Observation (..)
    , prettyObservation
    ) where

import           Control.DeepSeq
import           Data.Serialize
import qualified Data.Vector.Storable                        as VS
import           GHC.Generics
import           Prelude                                     hiding ((<>))
import           Statistics.Sample.WelfordOnlineMeanVariance
import           Text.PrettyPrint
import           Text.Printf

import           ML.ARAL.NeuralNetwork.Normalisation
import           ML.ARAL.Types

-- | One `Observation` holds one input and expected output.
data Observation =
  Observation
    { obsPeriod              :: !Period
    , obsInputValues         :: !(VS.Vector Double)
    , obsVarianceRegimeValue :: !Double -- e.g. Reward in RL
    -- , obsAction      :: !Int
    , obsExpectedOutputValue :: !Double
    }
  deriving (Eq, Show, Generic, Serialize, NFData)


prettyObservation :: Maybe (WelfordExistingAggregate (VS.Vector Double)) -> Observation -> Doc
prettyObservation mWelInp (Observation step inpVec _ out) =
  text "t=" <> int step <> comma <+>
  char '[' <> hcat (punctuate comma $ map (prettyFractional 3) (VS.toList inpVec)) <> char ']' <+>
  maybe mempty (\wel ->
  parens (char '[' <> hcat (punctuate comma $ map (prettyFractional 3) (VS.toList $ normaliseStateFeatureUnbounded wel inpVec)) <> char ']')) mWelInp <>
  colon <+> prettyFractional 3 out
  where prettyFractional :: (PrintfArg n) => Int -> n -> Doc
        prettyFractional commas = text . printf ("%+." ++ show commas ++ "f")
