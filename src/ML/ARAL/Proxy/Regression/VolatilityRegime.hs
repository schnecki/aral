{-# LANGUAGE DeriveAnyClass       #-}
{-# LANGUAGE DeriveGeneric        #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TupleSections        #-}
{-# LANGUAGE TypeSynonymInstances #-}
module ML.ARAL.Proxy.Regression.VolatilityRegime
    ( HMM
    , RegimeDetection (..)
    , addValueAndTrainHMM
    , getStateMeans
    , hmm
    , currentRegime
    , State (..)
    ) where

import           Control.DeepSeq
import           Data.Default
import           Data.List                                       (foldl')
import qualified Data.Map.Strict                                 as M
import           Data.Maybe                                      (fromMaybe)
import           Data.Monoid
import           Data.Serialize
import           Data.Vector.Serialize                           ()
import           Debug.Trace
import           GHC.Generics
import qualified Numeric.LAPACK.Matrix.HermitianPositiveDefinite as HermitianPD

import           Statistics.Sample.WelfordOnlineMeanVariance

import qualified Control.Monad.Exception.Synchronous             as Exceptional
import           Foreign.Storable                                (Storable)
import qualified Numeric.LAPACK.Matrix                           as Matrix
import qualified Numeric.LAPACK.Matrix.Square                    as Square
import qualified Numeric.Netlib.Class                            as Class

import qualified Math.HiddenMarkovModel                          as HMM
import qualified Math.HiddenMarkovModel.Distribution             as Distr

import qualified Data.Vector.Storable                            as VS
import qualified Numeric.LAPACK.Matrix.Hermitian                 as Hermitian
import qualified Numeric.LAPACK.Matrix.Layout                    as Layout
import           Numeric.LAPACK.Vector                           (Vector, singleton, (.*|))
import qualified Numeric.LAPACK.Vector                           as Vector

import qualified Data.Array.Comfort.Boxed                        as Array
import qualified Data.Array.Comfort.Shape                        as Shape

import           Data.Function.HT                                (nest)
import qualified Data.NonEmpty                                   as NonEmpty
import qualified Data.NonEmpty.Class                             as NonEmptyC
import           Data.Tuple.HT                                   (mapSnd)


squareFromLists :: (Shape.C sh, Eq sh, Storable a) => sh -> [Vector sh a] -> Matrix.Square sh a
squareFromLists sh = Square.fromFull . Matrix.fromRowArray sh . Array.fromList sh

squareToRowLists :: (Shape.C sh, Eq sh, Class.Floating a, Storable a) => Matrix.Square sh a -> [Vector sh a]
squareToRowLists = Matrix.toRows

gaussianToList :: (Shape.C sh) => Distr.T (Distr.Gaussian emiSh) sh a -> [(a, Vector emiSh a, Matrix.Upper emiSh a)]
gaussianToList (Distr.Gaussian params) = Array.toList params


-- diagGaussianFromList :: (Shape.C sh) => [(Double, Double)] -> Distr.T (Distr.Gaussian emiSh) sh Double
-- diagGaussianFromList xs =

--   Distr.gaussian $ Array.fromList stateSet $ map ((, Hermitian.diagonal Layout.RowMajor (Array.fromList sh (repeat 1))) . fst) xs
  -- map (\(m,var) ->
  --           (singleton   0.5 , Hermitian.identity Layout.RowMajor ()) :
  --           (singleton   5.0 , Hermitian.identity Layout.RowMajor ()) :
  --        )

-- toGaussian ::
--    (Shape.C emiSh, Class.Real prob) =>
--    (Vector emiSh prob, Matrix.HermitianPosDef emiSh prob) ->
--    (prob, Vector emiSh prob, Triangular.Upper emiSh prob)
-- toGaussian (center, covariance) =

--    gaussianFromCholesky center $ HermitianPD.decompose covariance


normalizeProb :: (Shape.C sh, Class.Real a) => Vector sh a -> Vector sh a
normalizeProb = snd . normalizeFactor


normalizeFactor :: (Shape.C sh, Class.Real a) => Vector sh a -> (a, Vector sh a)
normalizeFactor xs =
  let c = Vector.sum xs
   in (c, recip c .*| xs)

------------------------------

data State = Low | High
-- data State = Rising | High | Falling | Low
   deriving (Eq, Ord, Enum, Bounded, NFData, Generic, Serialize,Show)

type StateSet = Shape.Enumeration State

stateSet :: StateSet
stateSet = Shape.Enumeration


type HMM = HMM.Gaussian () StateSet Double

data RegimeDetection =
  RegimeDetection
    { regimeTrainedModel  :: Maybe (HMM.Gaussian () StateSet Double)
    , regimeInputCache    :: VS.Vector Double
    , regimeWelfordStates :: M.Map State (WelfordExistingAggregate Double) -- ^ One Welford for every state.
    , regimeWelfordAll    :: WelfordExistingAggregate Double             -- ^ Welford for values.
    , regimeTrainSize     :: Int
    }
  deriving (NFData, Generic, Serialize)

instance Show RegimeDetection where
  show (RegimeDetection (Just model) _ wel welAll _) = show (HMM.transition model) ++ "\n" ++ show (HMM.distribution model) ++ "\n" ++ show wel ++ "\n" ++ show welAll
  show (RegimeDetection Nothing _ _ _ _)             = "Empty HMM model"


instance Default RegimeDetection where
  def = RegimeDetection Nothing VS.empty (M.fromList $ zip [minBound .. maxBound] (repeat WelfordExistingAggregateEmpty)) WelfordExistingAggregateEmpty 1000


instance Serialize HMM where
  put mdl = put (HMM.toCSV mdl)
  get = do
    (eTxt :: String) <- get
    return $ error $ "TODO:\n" ++ eTxt
      -- case Exceptional.toEither (HMM.fromCSV (Shape.zeroBasedSize) eTxt) of
      --   Right (hmm :: HMM) -> hmm -- (hmm :: HMM)
      --   Left str           -> undefined -- error str


hmm :: HMM
hmm =
   HMM.Cons {
      HMM.initial = normalizeProb $ Vector.one stateSet,
      HMM.transition =
         squareFromLists stateSet $
            -- --          Ris Hig Fall Low
            -- stateVector 0.99 0.0 0.0 0.01 : -- Ris
            -- stateVector 0.01 0.99 0.0 0.0 : -- High
            -- stateVector 0.0 0.01 0.99 0.0 : -- Fall
            -- stateVector 0.0 0.0 0.01 0.99 : -- Low
            --           Low Hig
            stateVector 0.95 0.05 : -- Low
            stateVector 0.05 0.95 : -- High
            [],
      HMM.distribution =
         Distr.gaussian $ Array.fromList stateSet $
            (singleton   0.5 , Hermitian.identity Layout.RowMajor ()) :
            (singleton   5.0 , Hermitian.identity Layout.RowMajor ()) :
            -- (singleton   0 , Hermitian.identity Layout.RowMajor ()) :
            -- (singleton (-1), Hermitian.identity Layout.RowMajor ()) :
            []
   }

-- stateVector :: Double -> Double -> Double -> Double -> Vector StateSet Double
-- stateVector x0 x1 x2 x3 = Vector.fromList stateSet [x0,x1,x2,x3]

stateVector :: Double -> Double -> Vector StateSet Double
stateVector x0 x1 = Vector.fromList stateSet [x0,x1]


{- |
>>> take 20 $ map fst $ NonEmpty.flatten sineWaveLabeled
[Rising,Rising,High,High,High,Falling,Falling,Falling,Low,Low,Low,Rising,Rising,Rising,Rising,High,High,High,Falling,Falling]
-}
sineWaveLabeled :: NonEmpty.T [] (State, Double)
sineWaveLabeled =
   NonEmpty.mapTail (take 200) $
   fmap (\x -> (toEnum $ mod (floor (x*2/pi+0.5)) 4, sin x)) $
   NonEmptyC.iterate (0.5+) 0

sineWave :: NonEmpty.T [] Double
sineWave = fmap snd sineWaveLabeled

{- |
>>> take 20 $ NonEmpty.flatten revealed
[Rising,Rising,High,High,High,Falling,Falling,Falling,Low,Low,Low,Low,Rising,Rising,Rising,High,High,High,Falling,Falling]
-}
revealed :: NonEmpty.T [] State
revealed = HMM.reveal hmmTrainedSupervised $ fmap singleton sineWave

hmmTrainedSupervised :: HMM
hmmTrainedSupervised =
   HMM.finishTraining $ HMM.trainSupervised stateSet $
   fmap (mapSnd singleton) sineWaveLabeled

hmmTrainedUnsupervised :: HMM
hmmTrainedUnsupervised =
   HMM.finishTraining $ HMM.trainUnsupervised hmm $ fmap singleton sineWave

-- hmmIterativelyTrained :: HMM
-- hmmIterativelyTrained =
--    nest 100
--       (\model ->
--          HMM.finishTraining $ HMM.trainUnsupervised model $
--          fmap singleton sineWave)
--       hmm

transformValue :: Double -> Double
transformValue = id

hermitianFromList ::
   (Shape.C sh, Class.Floating a) => sh -> [a] -> Hermitian.Hermitian sh a
hermitianFromList = Hermitian.fromList Layout.RowMajor


addValueAndTrainHMM :: RegimeDetection -> [Double] -> RegimeDetection
addValueAndTrainHMM (RegimeDetection Nothing cache welSt welAll sz) xs
  | VS.length cache < max 1000 (3 * sz) = RegimeDetection Nothing cache' welSt welAll' sz
  | nrStates == 1 = error "addValueAndTrainHMM: Need at least 2 possible states"
  | otherwise = addValueAndTrainHMM (RegimeDetection (Just initModel) cache welSt welAll sz) xs
  where
    welAll' = foldl' addValue welAll xs
    cache' = cache VS.++ VS.fromList (map transformValue xs)
    (mean, _, variance) = finalize welAll'
    stdev = max 1e-3 (sqrt variance)
    maxVal = mean + stdev
    step = (maxVal - mean) / fromIntegral (nrStates - 1)
    nrStates = length ([minBound .. maxBound] :: [State])
    -- hermitianPD =
    --            HermitianPD.assurePositiveDefiniteness .
    --            hermitianFromList stateSet
    -- cov0 = hermitianPD [0.10, -0.09, 0.10]
    -- cov1 = hermitianPD [0.10,  0.09, 0.10]
    distr =
      Distr.gaussian $
      Array.fromList stateSet $ reverse $ take nrStates $ map (\x -> (singleton x, Hermitian.identity Layout.RowMajor ())) [mean,mean + step ..]
    initModel = HMM.uniform distr
addValueAndTrainHMM (RegimeDetection (Just model) cache welSt welAll sz) xs
  | VS.length cache < sz = RegimeDetection (Just model) cache' welSt welAll' sz
  | isBroken model' =
    trace ("\nOld model: " ++ show model)
    trace ("\nBroken model: " ++ show model')
    trace ("\nNew size: " ++ show (2*sz))
    RegimeDetection (Just model) cache' welSt welAll' (2*sz) -- increase number of values
  | otherwise =
    trace ("\n\nNEW MODEL: " ++ show model')
    RegimeDetection (Just model') (VS.singleton $ VS.last cache') welSt' welAll' sz
  where
    welAll' = foldl' addValue welAll xs
    cache' = cache VS.++ VS.fromList (map transformValue xs)
    (mean, _, variance) = finalize welAll'
    stdev = max 1e-3 (sqrt variance)
    trainExamplesU = VS.head cache' NonEmpty.!: VS.toList (VS.tail cache')
    trainExamplesSLs = map label (VS.toList cache')
    trainExamplesS = fromMaybe (error "empty vector") $ NonEmpty.fetch $ fmap (mapSnd singleton) trainExamplesSLs
      -- label (VS.head cache') NonEmpty.!: map label (VS.toList (VS.tail cache'))
    label x = (fst $ head $ filter (\(s,v) -> abs x < v) $ zip states [mean + step, mean + 2*step ..] ++ [(last states, abs x + 1)], x)
    mkValues :: State -> [Double]
    mkValues st = map snd $ filter ((== st ) . fst ) trainExamplesSLs
    welSt' = foldl' (\m st -> M.alter (Just . maybe (addValues WelfordExistingAggregateEmpty (mkValues st) ) (`addValues` mkValues st)) st m) welSt states
    maxVal = mean + 0.618 * stdev
    step = (maxVal - mean) / fromIntegral (nrStates - 1)
    states = [minBound .. maxBound] :: [State]
    nrStates = length states
    distrs = HMM.distribution $ HMM.finishTraining $ HMM.trainSupervised stateSet trainExamplesS -- distributions are learned supervised to prevent divergence
    setDistr hmm = hmm { HMM.distribution = distrs }
    model' =
      -- trace ("trainExamplesL: " ++ show trainExamplesS)
      fixInitialProbs $ setDistr $ HMM.finishTraining $ HMM.trainUnsupervised (setDistr model) $ fmap singleton trainExamplesU
      -- trace ("model: " ++ show model)
      -- trace ("trainingData: " ++ show trainExamples)
      -- nest 100 (\mdl -> HMM.finishTraining $ HMM.trainUnsupervised mdl $ fmap singleton trainExamples) model
    fixInitialProbs mdl =
      mdl { HMM.initial = normalizeProb $ Vector.fromList stateSet $ map (max 0.01 . min 0.99) $ Vector.toList (HMM.initial mdl)}
    isBroken mdl =
      any (\x -> -- trace ("x: " ++ show x ++ " test: " ++ show (x > 0.9999, x < 0.0001, isNaN x))
              x > 0.9999 || x < 0.0001 || isNaN x) (concatMap Vector.toList $ squareToRowLists $ HMM.transition mdl)
      -- &&
      -- all (\(x, vec, mat) -> not . isNaN $ x) (gaussianToList $ HMM.distribution mdl)

getStateMeans :: RegimeDetection -> [Double]
getStateMeans (RegimeDetection (Just model) _ _ _ _) = concatMap (\(weigth,center,cov) -> Vector.toList center) $ gaussianToList $ HMM.distribution model
getStateMeans _                                      = replicate (length [minBound..(maxBound :: State)]) 0

-- | Ensure you have added the data.
currentRegime :: RegimeDetection -> VS.Vector Double -> State
currentRegime (RegimeDetection (Just model) _ _ _ _) vec
  | VS.length vec < 3 = toEnum 0
  | otherwise = last $ NonEmpty.flatten $ HMM.reveal model $ fmap singleton $ transformValue (VS.head vec) NonEmpty.!: VS.toList (VS.map transformValue $ VS.tail vec)
currentRegime _ _ = toEnum 0
