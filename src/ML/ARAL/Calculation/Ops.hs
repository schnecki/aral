{-# LANGUAGE CPP                 #-}
{-# LANGUAGE ScopedTypeVariables #-}
module ML.ARAL.Calculation.Ops
    ( mkCalculation
    , rhoValueWith
    , rValue
    , rValueAgentWith
    , rValueWith
    , rValueNoUnscaleWith
    , eValue
    , eValueFeat
    , eValueAvgCleaned
    , eValueAvgCleanedAgent
    , eValueAvgCleanedFeat
    , vValue
    , vValueAgentWith
    , vValueWith
    , vValueNoUnscaleWith
    , wValueFeat
    , rhoValue
    , rhoValueAgentWith
    , overEstimateRhoCalc
    , RSize (..)
    , expSmthPsi
    ) where

import           Control.Lens
import           Control.Monad.IO.Class
import qualified Data.Vector.Storable as V
import qualified Data.Vector                  as VB
import           Control.Parallel.Strategies    hiding (r0)
import           Data.Maybe                     (fromMaybe)
import           Data.List (zipWith6,zipWith5,zipWith4)
import           Control.DeepSeq
import Control.Applicative ((<|>))

import           ML.ARAL.Algorithm
import           ML.ARAL.Calculation.Type
import           ML.ARAL.Decay                  (decaySetup)
import           ML.ARAL.NeuralNetwork.NNConfig
import           ML.ARAL.NeuralNetwork.ReplayMemory
import           ML.ARAL.Parameters
import           ML.ARAL.Properties
import           ML.ARAL.Settings
import           ML.ARAL.Proxy                  as P
import           ML.ARAL.Type
import           ML.ARAL.Types

import Debug.Trace

#ifdef DEBUG
import Prelude hiding (maximum, minimum)
import qualified Prelude (maximum, minimum)
import qualified Data.List
maximumBy, minimumBy :: (a -> a -> Ordering) -> [a] -> a
maximumBy _ [] = error "empty input to maximumBy in ML.ARAL.Calculation.Ops"
maximumBy f xs = Data.List.maximumBy f xs
minimumBy _ [] = error "empty input to minimumBy in ML.ARAL.Calculation.Ops"
minimumBy f xs = Data.List.minimumBy f xs
maximum, minimum :: (Ord a) => [a] -> a
maximum [] = error "empty input to maximum in ML.ARAL.Calculation.Ops"
maximum xs = Data.List.maximum xs
minimum [] = error "empty input to minimum in ML.ARAL.Calculation.Ops"
minimum xs = Data.List.minimum xs
#else
import           Data.List                      (maximumBy, minimumBy)
#endif


-- | Used to select a discount factor.
data RSize
  = RSmall
  | RBig


expSmthPsi :: Double
expSmthPsi = 0.001

-- expSmthReward :: Double
-- expSmthReward = 0.001


keepXLastValues :: Int
keepXLastValues = 100

mkCalculation ::
     (MonadIO m)
  => AgentType
  -> ARAL s as
  -> (StateFeatures, DisallowedActionIndicies) -- ^ State features and filtered actions for each agent
  -> ActionChoice -- ^ ActionIndex for each agent
  -> RewardValue
  -> (StateNextFeatures, DisallowedActionIndicies) -- ^ State features and filtered actions for each agent
  -> EpisodeEnd
  -> ExpectedValuationNext
  -> m (Calculation, ExpectedValuationNext)
mkCalculation agTp borl state as reward stateNext episodeEnd =
  mkCalculation' agTp borl state as reward stateNext episodeEnd (borl ^. algorithm)

ite :: Bool -> p -> p -> p
ite True thenPart _  = thenPart
ite False _ elsePart = elsePart
{-# INLINE ite #-}

rhoMinimumState' :: ARAL s as -> Value -> Value
rhoMinimumState' borl rhoVal' = mapValue go rhoVal'
  where
    go v =
      case borl ^. objective of
        Maximise
          | v >= 0 -> max (v - 2) (0.975 * v)
          | otherwise -> max (v - 2) (1.025 * v)
        Minimise
          | v >= 0 -> min (v + 2) (1.025 * v)
          | otherwise -> min (v + 2) (0.975 * v)

-- | Get an exponentially smoothed parameter. Due to lazy evaluation the calculation for the other parameters are
-- ignored!
getExpSmthParam :: ARAL s as -> ((Proxy -> Const Proxy Proxy) -> Proxies -> Const Proxy Proxies) -> Getting Double (Parameters Double) Double -> Double
getExpSmthParam borl p param
  | isANN = 1
  | otherwise = params' ^. param
  where
    isANN = P.isNeuralNetwork px && borl ^. t >= px ^?! proxyNNConfig . replayMemoryMaxSize
    px = borl ^. proxies . p
    params' = decayedParameters borl

-- | Overestimates the average reward. This ensures that we constantly aim for better policies.
overEstimateRhoCalc :: ARAL s as -> Double -> Double
overEstimateRhoCalc borl rhoVal = max' (max' expSmthRho rhoVal) (rhoVal + 0.1 * diff)
  where
    expSmthRho = borl ^. expSmoothedReward
    diff = rhoVal - expSmthRho
    max' :: (Ord x) => x -> x -> x
    max' =
      case borl ^. objective of
        Maximise -> max
        Minimise -> min

shareRhoVal :: ARAL s as -> Value -> Value
shareRhoVal borl v@(AgentValue vec)
  | borl ^. settings . independentAgentsSharedRho = AgentValue $ V.map (const val) vec
  | otherwise = v
  where
    val = V.sum vec / fromIntegral (V.length vec)
    -- val = maxOrMin vec
    -- maxOrMin =
    --   case borl ^. objective of
    --     Maximise -> V.maximum
    --     Minimise -> V.minimum


mkCalculation' ::
     (MonadIO m)
  => AgentType
  -> ARAL s as
  -> (StateFeatures, DisallowedActionIndicies)
  -> ActionChoice
  -> RewardValue
  -> (StateNextFeatures, DisallowedActionIndicies)
  -> EpisodeEnd
  -> Algorithm NetInputWoAction
  -> ExpectedValuationNext
  -> m (Calculation, ExpectedValuationNext)
mkCalculation' agTp borl (state, _) as reward (stateNext, stateNextActIdxes) episodeEnd (AlgNBORL ga0 ga1 avgRewardType mRefState) expValStateNext = do
  let params' = decayedParameters borl
  let aNr = VB.map snd as
      randomAction = any fst as
  let isRefState = mRefState == Just (state, VB.toList aNr)
  let alp = getExpSmthParam borl rho alpha
      bta = getExpSmthParam borl v beta
      dltW = getExpSmthParam borl w delta
      gamR0 = getExpSmthParam borl r0 gamma
      gamR1 = getExpSmthParam borl r1 gamma
      alpRhoMin = getExpSmthParam borl rhoMinimum alphaRhoMin
      xiVal = params' ^. xi
      zetaVal = params' ^. zeta
      period = borl ^. t
      (psiValRho, psiValV, psiValW) = borl ^. psis -- exponentially smoothed Psis
  let agents = borl ^. settings . independentAgents
  let learnFromRandom = params' ^. exploration > params' ^. learnRandomAbove
  let label = (state, aNr)
      epsEnd
        | episodeEnd = 0
        | otherwise = 1
      randAct
        | randomAction = 0
        | otherwise = 1
      nonRandAct
        | learnFromRandom = 1
        | otherwise = 1 - randAct
  let expSmth
        | learnFromRandom = expSmthPsi
        | otherwise = randAct * expSmthPsi
  let lastRews' =
        case avgRewardType of
          ByMovAvg movAvgLen -> V.take movAvgLen $ reward `V.cons` (borl ^. lastRewards)
          _ -> V.take keepXLastValues $ reward `V.cons` (borl ^. lastRewards)
  vValState <- vValueWith agTp Worker borl state aNr `using` rpar
  rhoMinimumState <- rhoMinimumValueFeat agTp borl state aNr `using` rpar
  vValStateNext <- vStateValueWith agTp Target borl (stateNext, stateNextActIdxes) `using` rpar
  rhoVal <- rhoValueWith agTp Worker borl state aNr `using` rpar
  wValState <- wValueFeat agTp borl state aNr `using` rpar
  wValStateNext <- wStateValue agTp borl (stateNext, stateNextActIdxes) `using` rpar
  psiVState <- P.lookupProxy agTp period Worker label (borl ^. proxies . psiV) `using` rpar
  psiWState <- P.lookupProxy agTp period Worker label (borl ^. proxies . psiW) `using` rpar
  r0ValState <- rValueWith agTp Worker borl RSmall state aNr `using` rpar
  r0ValStateNext <- rStateValueWith agTp Target borl RSmall (stateNext, stateNextActIdxes) `using` rpar
  r1ValState <- rValueWith agTp Worker borl RBig state aNr `using` rpar
  r1ValStateNext <- rStateValueWith agTp Target borl RBig (stateNext, stateNextActIdxes) `using` rpar
  -- Rho
  rhoState <-
    case avgRewardType of
      Fixed x -> return $ toValue agents x
      ByMovAvg l
        | isUnichain borl -> return $ toValue agents $ sum (V.toList lastRews') / fromIntegral l
      ByMovAvg _ -> error "ByMovAvg is not allowed in multichain setups"
      ByReward -> return $ toValue agents reward
      ByStateValues -> return $ reward .+ vValStateNext - vValState
      ByStateValuesAndReward ratio decay -> return $ (1 - ratio') .* (reward .+ vValStateNext - vValState) +. ratio' * reward
        where ratio' = decaySetup decay period ratio
  let maxOrMin =
        case borl ^. objective of
          Maximise -> max
          Minimise -> min
  let rhoVal'
        | randomAction && not learnFromRandom = shareRhoVal borl $ rhoVal
        | otherwise =
          shareRhoVal borl $
          zipWithValue maxOrMin rhoMinimumState $
          case avgRewardType of
            ByMovAvg _ -> rhoState
            Fixed x -> toValue agents x
            _ -> (1 - alp) .* rhoVal + alp .* rhoState
  -- RhoMin
  let rhoMinimumVal'
        | randomAction = rhoMinimumState
        | otherwise = zipWithValue maxOrMin rhoMinimumState $ (1 - alpRhoMin) .* rhoMinimumState + alpRhoMin .* rhoMinimumState' borl rhoVal'
  -- PsiRho (should converge to 0)
  psiRho <- ite (isUnichain borl) (return $ rhoVal' - rhoVal) (subtract rhoVal' <$> rhoStateValue agTp borl (stateNext, stateNextActIdxes))
  -- V
  let rhoValOverEstimated
        | borl ^. settings . overEstimateRho = mapValue (overEstimateRhoCalc borl) rhoVal'
        | otherwise = rhoVal'
  let vValState' = (1 - bta) .* vValState + bta .* (reward .- rhoValOverEstimated + epsEnd .* vValStateNext + nonRandAct .* (psiVState + zetaVal .* psiWState))
      psiV = reward .+ vValStateNext - rhoValOverEstimated - vValState' -- should converge to 0
      psiVState' = (1 - xiVal * bta) .* psiVState + bta * xiVal .* psiV
  -- LastVs
  let lastVs' = VB.take keepXLastValues $ vValState' `VB.cons` (borl ^. lastVValues)
  -- W
  let wValState'
        | isRefState = 0
        | otherwise = (1 - dltW) .* wValState + dltW .* (-vValState' + epsEnd .* wValStateNext + nonRandAct .* psiWState)
      psiW = wValStateNext - vValState' - wValState'
      psiWState'
        | isRefState = 0
        | otherwise = (1 - xiVal * dltW) .* psiWState + dltW * xiVal .* psiW
  -- R0/R1
  let r0ValState' = (1 - gamR0) .* r0ValState + gamR0 .* (reward .+ epsEnd * ga0 .* r0ValStateNext)
  let r1ValState' = (1 - gamR1) .* r1ValState + gamR1 .* (reward .+ epsEnd * ga1 .* r1ValStateNext)
  -- Psis Scalar calues for output only
  let psiValRho' = (1 - expSmth) .* psiValRho + expSmth .* abs psiRho
  let psiValV' = (1 - expSmth) .* psiValV + expSmth .* abs psiVState'
  let psiValW' = (1 - expSmth) .* psiValW + expSmth .* abs psiWState'
  return $
    ( Calculation
        { getRhoMinimumVal' = Just rhoMinimumVal'
        , getRhoVal' = Just rhoVal'
        , getPsiVValState' = Just psiVState'
        , getVValState' = Just vValState'
        , getPsiWValState' = Just psiWState' -- $ ite isRefState 0 psiWState'
        , getWValState' = Just $ ite isRefState 0 wValState'
        , getR0ValState' = Just r0ValState'
        , getR1ValState' = Just r1ValState'
        , getPsiValRho' = Just psiValRho'
        , getPsiValV' = Just psiValV'
        , getPsiValW' = Just psiValW'
        , getLastVs' = Just lastVs'
        , getLastRews' = lastRews'
        , getEpisodeEnd = episodeEnd
        , getExpSmoothedReward' = ite (randomAction && not learnFromRandom) (borl ^. expSmoothedReward) ((1 - alp) * borl ^. expSmoothedReward + alp * reward)
        }
    , ExpectedValuationNext
        { getExpectedValStateNextRho = Nothing
        , getExpectedValStateNextV = error "N-Step not implemented in AlgNBORL"
        , getExpectedValStateNextW = Nothing
        , getExpectedValStateNextR0 = Nothing
        , getExpectedValStateNextR1 = Nothing
        })

mkCalculation' agTp borl sa@(state, _) as reward (stateNext, stateNextActIdxes) episodeEnd (AlgARAL ga0 ga1 avgRewardType) expValStateNext = do
  let aNr = VB.map snd as
      randomAction = any fst as
  rhoMinimumState <- rhoMinimumValueFeat agTp borl state aNr
  rhoVal <- rhoValueWith agTp Worker borl state aNr
  r0ValState <- rValueWith agTp Worker borl RSmall state aNr `using` rpar
  let ~r0StateNext = rStateValueWith agTp Target borl RSmall (stateNext, stateNextActIdxes) `using` rpar
  r1ValState <- rValueWith agTp Worker borl RBig state aNr `using` rpar
  let ~r1StateNext = rStateValueWith agTp Target borl RBig (stateNext, stateNextActIdxes) `using` rpar
  r1StateNextWorker <- rStateValueWith agTp Worker borl RBig (stateNext, stateNextActIdxes) `using` rpar
  let alp = getExpSmthParam borl rho alpha
      alpRhoMin = getExpSmthParam borl rhoMinimum alphaRhoMin
      gam = getExpSmthParam borl r1 gamma
  let agents = borl ^. settings . independentAgents
  let params' = decayedParameters borl
  let learnFromRandom = params' ^. exploration > params' ^. learnRandomAbove -- && maybe True ((borl ^. t - 1000 >) . replayMemoriesSize) (borl ^. proxies . replayMemory)
  let initPhase = maybe False ((borl ^. t <=) . replayMemoriesSize) (borl ^. proxies . replayMemory)
  let fixedPhase = maybe False ((borl ^. t - 500 <=) . replayMemoriesSize) (borl ^. proxies . replayMemory)
  let epsEnd
        | episodeEnd = 0
        | otherwise = 1
  let lastRews' =
        case avgRewardType of
          ByMovAvg movAvgLen -> V.take movAvgLen $ reward `V.cons` (borl ^. lastRewards)
          _ -> V.take keepXLastValues $ reward `V.cons` (borl ^. lastRewards)
  -- Rho
  rhoState <-
    case avgRewardType of
      Fixed x -> return $ toValue agents x
      ByMovAvg l -> return $ toValue agents $ sum (V.toList lastRews') / fromIntegral l
      ByReward -> return $ toValue agents reward
      ByStateValues -> return $ reward .+ r1StateNextWorker - r1ValState
      ByStateValuesAndReward ratio decay -> return $ (1 - ratio') .* (reward .+ r1StateNextWorker - r1ValState) +. ratio' * reward
        where ratio' = decaySetup decay (borl ^. t) ratio
  let maxOrMin =
        case borl ^. objective of
          Maximise -> max
          Minimise -> min
  let rhoVal'
        | initPhase = AgentValue $ V.fromList $ replicate agents (borl ^. expSmoothedReward)
        | fixedPhase = rhoVal
        | randomAction && not learnFromRandom = shareRhoVal borl $ zipWithValue maxOrMin rhoMinimumState rhoVal
        | otherwise =
          shareRhoVal borl $ zipWithValue maxOrMin rhoMinimumState $
          case avgRewardType of
            ByMovAvg _ -> rhoState
            Fixed x -> toValue agents x
            _ -> (1 - alp) .* rhoVal + alp .* rhoState
      rhoValOverEstimated
        | borl ^. settings . overEstimateRho = mapValue (overEstimateRhoCalc borl) rhoVal'
        | otherwise = rhoVal'
  -- RhoMin
  let rhoMinimumVal'
        | randomAction && not learnFromRandom = rhoMinimumState
        | otherwise = shareRhoVal borl $ zipWithValue maxOrMin rhoMinimumState $ (1 - alpRhoMin) .* rhoMinimumState + alpRhoMin .* rhoMinimumState' borl rhoVal'
  expStateNextValR0 <-
    if randomAction && not learnFromRandom
      then r0StateNext
      else maybe r0StateNext return (getExpectedValStateNextR0 expValStateNext)
  expStateNextValR1 <-
    if randomAction && not learnFromRandom
      then r1StateNext
      else maybe r1StateNext return (getExpectedValStateNextR1 expValStateNext)
  let expStateValR0 = reward .- rhoValOverEstimated + ga0 * epsEnd .* expStateNextValR0
      expStateValR1 = reward .- rhoValOverEstimated + ga1 * epsEnd .* expStateNextValR1
  let r0ValState' = (1 - gam) .* r0ValState + gam .* expStateValR0
  let r1ValState' = (1 - gam) .* r1ValState + gam .* expStateValR1
  let expSmthRewRate
        | initPhase = 0.01
        | otherwise = min alp 0.001
      expSmthRew'
        | not initPhase && randomAction && not learnFromRandom = borl ^. expSmoothedReward
        | otherwise = (1 - expSmthRewRate) * borl ^. expSmoothedReward + expSmthRewRate * reward
  return
    ( Calculation
        { getRhoMinimumVal' = Just rhoMinimumVal'
        , getRhoVal' = Just rhoVal'
        , getPsiVValState' = Nothing
        , getVValState' = Nothing
        , getPsiWValState' = Nothing
        , getWValState' = Nothing
        , getR0ValState' = Just r0ValState' -- gamma middle/low
        , getR1ValState' = Just r1ValState' -- gamma High
        , getPsiValRho' = Nothing
        , getPsiValV' = Nothing
        , getPsiValW' = Nothing
        , getLastVs' = Nothing
        , getLastRews' = lastRews'
        , getEpisodeEnd = episodeEnd
        , getExpSmoothedReward' = expSmthRew'
        }
    , ExpectedValuationNext
        { getExpectedValStateNextRho = Nothing
        , getExpectedValStateNextV = Nothing
        , getExpectedValStateNextW = Nothing
        , getExpectedValStateNextR0 = Just expStateValR0
        , getExpectedValStateNextR1 = Just expStateValR1
        })
mkCalculation' agTp borl (state, _) as reward (stateNext, stateNextActIdxes) episodeEnd (AlgARALVOnly avgRewardType mRefState) expValStateNext = do
  let aNr = VB.map snd as
      randomAction = any fst as
  let alp = getExpSmthParam borl rho alpha
      alpRhoMin = getExpSmthParam borl rhoMinimum alphaRhoMin
      bta = getExpSmthParam borl v beta
      agents = borl ^. settings . independentAgents
  let epsEnd
        | episodeEnd = 0
        | otherwise = 1
  let learnFromRandom = params' ^. exploration > params' ^. learnRandomAbove
      params' = decayedParameters borl
  rhoVal <- rhoValueWith agTp Worker borl state aNr `using` rpar
  vValState <- vValueWith agTp Worker borl state aNr `using` rpar
  vValStateNext <- vStateValueWith agTp Target borl (stateNext, stateNextActIdxes) `using` rpar
  let lastRews' =
        case avgRewardType of
          ByMovAvg movAvgLen -> V.take movAvgLen $ reward `V.cons` (borl ^. lastRewards)
          _ -> V.take keepXLastValues $ reward `V.cons` (borl ^. lastRewards)
  rhoMinimumState <- rhoMinimumValueFeat agTp borl state aNr `using` rpar
  rhoState <-
    case avgRewardType of
      Fixed x -> return $ toValue agents x
      ByMovAvg _ -> return $ toValue agents $ sum (V.toList lastRews') / fromIntegral (V.length lastRews')
      ByReward -> return $ toValue agents reward
      ByStateValues -> return $ reward .+ vValStateNext - vValState
      ByStateValuesAndReward ratio decay -> return $ (1 - ratio') .* (reward .+ vValStateNext - vValState) +. ratio' * reward
        where ratio' = decaySetup decay (borl ^. t) ratio
  let maxOrMin =
        case borl ^. objective of
          Maximise -> max
          Minimise -> min
  let rhoVal'
        | randomAction = shareRhoVal borl rhoVal
        | otherwise =
          shareRhoVal borl $ zipWithValue maxOrMin rhoMinimumState $
          case avgRewardType of
            ByMovAvg _ -> rhoState
            Fixed x -> toValue agents x
            _ -> (1 - alp) .* rhoVal + alp .* rhoState
      rhoValOverEstimated
        | borl ^. settings . overEstimateRho = shareRhoVal borl $ mapValue (overEstimateRhoCalc borl) rhoVal'
        | otherwise = shareRhoVal borl rhoVal'
  let rhoMinimumVal'
        | randomAction = shareRhoVal borl rhoMinimumState
        | otherwise = shareRhoVal borl $ zipWithValue maxOrMin rhoMinimumState $ (1 - alpRhoMin) .* rhoMinimumState + alpRhoMin .* rhoMinimumState' borl rhoVal'
  let expStateNextValV
        | randomAction = epsEnd .* vValStateNext
        | otherwise = fromMaybe (epsEnd .* vValStateNext) (getExpectedValStateNextV expValStateNext)
      expStateValV = reward .- rhoValOverEstimated + expStateNextValV
  let vValState' = (1 - bta) .* vValState + bta .* (reward .- rhoValOverEstimated + expStateValV)
  let lastVs' = VB.take keepXLastValues $ vValState' `VB.cons` (borl ^. lastVValues)
  return $
    ( Calculation
        { getRhoMinimumVal' = Just rhoMinimumVal'
        , getRhoVal' = Just rhoVal'
        , getPsiVValState' = Nothing
        , getVValState' = Just $ ite (mRefState == Just (state, VB.toList aNr)) 0 vValState'
        , getPsiWValState' = Nothing
        , getWValState' = Nothing
        , getR0ValState' = Nothing
        , getR1ValState' = Nothing
        , getPsiValRho' = Nothing
        , getPsiValV' = Nothing
        , getPsiValW' = Nothing
        , getLastVs' = Just $ lastVs'
        , getLastRews' = lastRews'
        , getEpisodeEnd = episodeEnd
        , getExpSmoothedReward' = ite (randomAction && not learnFromRandom) (borl ^. expSmoothedReward) ((1 - alp) * borl ^. expSmoothedReward + alp * reward)
        }
    , ExpectedValuationNext
        { getExpectedValStateNextRho = Nothing
        , getExpectedValStateNextV = Just expStateValV
        , getExpectedValStateNextW = Nothing
        , getExpectedValStateNextR0 = Nothing
        , getExpectedValStateNextR1 = Nothing
        })
mkCalculation' agTp borl (state, _) as reward (stateNext, stateNextActIdxes) episodeEnd AlgRLearning expValStateNext = do
  let aNr = VB.map snd as
      isRandomAction = any fst as
  let alp = getExpSmthParam borl rho alpha
      alpRhoMin = getExpSmthParam borl rhoMinimum alphaRhoMin
      bta = getExpSmthParam borl v beta
  let epsEnd
        | episodeEnd = 0
        | otherwise = 1
  let learnFromRandom = params' ^. exploration > params' ^. learnRandomAbove
      params' = decayedParameters borl
  rhoVal <- rhoValueWith agTp Worker borl state aNr `using` rpar
  vValState <- vValueWith agTp Worker borl state aNr `using` rpar
  vValStateNext <- vStateValueWith agTp Target borl (stateNext, stateNextActIdxes) `using` rpar
  rhoMinimumState <- rhoMinimumValueFeat agTp borl state aNr `using` rpar
  let rhoState = reward .+ vValStateNext - vValState -- r_imm + U_R(s') - U_R(s)
  let maxOrMin =
        case borl ^. objective of
          Maximise -> max
          Minimise -> min
  let rhoVal'
        | isRandomAction = rhoVal -- shareRhoVal borl rhoVal
        | otherwise = shareRhoVal borl $ zipWithValue maxOrMin rhoMinimumState $ (1 - alp) .* rhoVal + alp .* rhoState
      rhoValOverEstimated
        | borl ^. settings . overEstimateRho = shareRhoVal borl $ mapValue (overEstimateRhoCalc borl) rhoVal'
        | otherwise = shareRhoVal borl rhoVal'
  let rhoMinimumVal'
        | isRandomAction = shareRhoVal borl rhoMinimumState
        | otherwise = shareRhoVal borl $ zipWithValue maxOrMin rhoMinimumState $ (1 - alpRhoMin) .* rhoMinimumState + alpRhoMin .* rhoMinimumState' borl rhoVal'
  let expStateNextValV
        | isRandomAction = epsEnd .* vValStateNext
        | otherwise = fromMaybe (epsEnd .* vValStateNext) (getExpectedValStateNextV expValStateNext)
      expStateValV = reward .- rhoValOverEstimated + expStateNextValV
  let vValState' = (1 - bta) .* vValState + bta .* (reward .- rhoValOverEstimated + expStateValV)
  let lastVs' = VB.take keepXLastValues $ vValState' `VB.cons` (borl ^. lastVValues)
  return $
    ( Calculation
        { getRhoMinimumVal' = Just rhoMinimumVal'
        , getRhoVal' = Just rhoVal'
        , getPsiVValState' = Nothing
        , getVValState' = Just vValState'
        , getPsiWValState' = Nothing
        , getWValState' = Nothing
        , getR0ValState' = Nothing
        , getR1ValState' = Nothing
        , getPsiValRho' = Nothing
        , getPsiValV' = Nothing
        , getPsiValW' = Nothing
        , getLastVs' = Just $ lastVs'
        , getLastRews' = V.take keepXLastValues $ reward `V.cons` (borl ^. lastRewards)
        , getEpisodeEnd = episodeEnd
        , getExpSmoothedReward' = ite (isRandomAction && not learnFromRandom) (borl ^. expSmoothedReward) ((1 - alp) * borl ^. expSmoothedReward + alp * reward)
        }
    , ExpectedValuationNext
        { getExpectedValStateNextRho = Nothing
        , getExpectedValStateNextV = Just expStateValV
        , getExpectedValStateNextW = Nothing
        , getExpectedValStateNextR0 = Nothing
        , getExpectedValStateNextR1 = Nothing
        })

mkCalculation' agTp borl (state, _) as reward (stateNext, stateNextActIdxes) episodeEnd (AlgDQN ga _) expValStateNext = do
  let aNr = VB.map snd as
      randomAction = any fst as
  let gam = getExpSmthParam borl r1 gamma
  let epsEnd
        | episodeEnd = 0
        | otherwise = 1
  let learnFromRandom = params' ^. exploration > params' ^. learnRandomAbove
      params' = decayedParameters borl
  let lastRews' = V.take keepXLastValues $ reward `V.cons` (borl ^. lastRewards)
  r1ValState <- rValueWith agTp Worker borl RBig state aNr `using` rpar
  r1StateNext <- rStateValueWith agTp Target borl RBig (stateNext, stateNextActIdxes) `using` rpar
  let expStateNextValR1
        | randomAction && not learnFromRandom = epsEnd .* r1StateNext
        | otherwise = fromMaybe (epsEnd .* r1StateNext) (getExpectedValStateNextR1 expValStateNext)
      expStateValR1 = reward .+ (epsEnd * ga) .* expStateNextValR1
  let r1ValState' = (1 - gam) .* r1ValState + gam .* expStateValR1
  return
    ( emptyCalculation
        { getR1ValState' = Just r1ValState'
        , getLastRews' = lastRews'
        , getEpisodeEnd = episodeEnd
        , getExpSmoothedReward' = ite (randomAction && not learnFromRandom) (borl ^. expSmoothedReward) ((1-gam) * borl ^. expSmoothedReward + gam * reward)
        }
    , emptyExpectedValuationNext {getExpectedValStateNextR1 = Just expStateValR1})

-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
rhoMinimumValue :: (MonadIO m) => AgentType -> ARAL s as -> State s -> AgentActionIndices -> m Value
rhoMinimumValue agTp borl state = rhoMinimumValueWith agTp Worker borl (ftExt state)
  where
    ftExt = borl ^. featureExtractor

rhoMinimumValueFeat :: (MonadIO m) => AgentType -> ARAL s as -> StateFeatures -> AgentActionIndices -> m Value
rhoMinimumValueFeat agTp = rhoMinimumValueWith agTp Worker

rhoMinimumValueWith :: (MonadIO m) => AgentType -> LookupType -> ARAL s as -> StateFeatures -> AgentActionIndices -> m Value
rhoMinimumValueWith agTp lkTp borl state a = P.lookupProxy agTp (borl ^. t) lkTp (state,a) (borl ^. proxies.rhoMinimum)

-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
rhoValue :: (MonadIO m) => AgentType -> ARAL s as -> State s -> AgentActionIndices -> m Value
rhoValue agTp borl s = rhoValueWith agTp Worker borl (ftExt s)
  where
    ftExt = borl ^. featureExtractor

-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
rhoValueAgentWith :: (MonadIO m) => AgentType -> LookupType -> ARAL s as -> P.AgentNumber -> State s -> ActionIndex -> m Double
rhoValueAgentWith agTp lkTp borl agent s a = P.lookupProxyAgent agTp (borl ^. t) lkTp agent (ftExt s, a) (borl ^. proxies . rho)
  where
    ftExt = borl ^. featureExtractor


rhoValueWith :: (MonadIO m) => AgentType -> LookupType -> ARAL s as -> StateFeatures -> AgentActionIndices -> m Value
rhoValueWith agTp lkTp borl state a = P.lookupProxy agTp (borl ^. t) lkTp (state,a) (borl ^. proxies.rho)

rhoStateValue :: (MonadIO m) => AgentType -> ARAL s as -> (StateFeatures, DisallowedActionIndicies) -> m Value
rhoStateValue agTp borl (state, actIdxes) =
  case borl ^. proxies . rho of
    Scalar r _ -> return $ AgentValue r
    _ -> reduceValues maxOrMin <$> lookupState agTp Target (state, actIdxes) (borl ^. proxies . rho)
      -- V.mapM (rhoValueWith Target borl state) actIdxes
  where
    maxOrMin =
      case borl ^. objective of
        Maximise -> V.maximum
        Minimise -> V.minimum

-- | Bias value from Worker net.
vValue :: (MonadIO m) => AgentType -> ARAL s as -> State s -> AgentActionIndices -> m Value
vValue agTp borl s = vValueWith agTp Worker borl (ftExt s)
  where
    ftExt = borl ^. featureExtractor

-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
vValueAgentWith :: (MonadIO m) => AgentType -> LookupType -> ARAL s as -> P.AgentNumber -> State s -> ActionIndex -> m Double
vValueAgentWith agTp lkTp borl agent s a = P.lookupProxyAgent agTp (borl ^. t) lkTp agent (ftExt s, a) (borl ^. proxies . v)
  where
    ftExt = borl ^. featureExtractor


-- | Get bias value from specified net and with features.
vValueWith :: (MonadIO m) => AgentType -> LookupType -> ARAL s as -> StateFeatures -> AgentActionIndices -> m Value
vValueWith agTp lkTp borl state a = P.lookupProxy agTp (borl ^. t) lkTp (state, a) (borl ^. proxies . v)

-- | For DEBUGGING only!
vValueNoUnscaleWith :: (MonadIO m) => AgentType -> LookupType -> ARAL s as -> StateFeatures -> AgentActionIndices -> m Value
vValueNoUnscaleWith agTp lkTp borl state a = P.lookupProxyNoUnscale agTp (borl ^. t) lkTp (state, a) (borl ^. proxies . v)


-- | Get maximum bias value of state of specified net.
vStateValueWith :: (MonadIO m) => AgentType -> LookupType -> ARAL s as -> (StateFeatures, DisallowedActionIndicies) -> m Value
vStateValueWith agTp lkTp borl (state, asIdxes) = reduceValues maxOrMin <$> lookupState agTp lkTp (state, asIdxes) (borl ^. proxies . v)
  -- V.mapM (vValueWith lkTp borl state) asIdxes
  where
    maxOrMin =
      case borl ^. objective of
        Maximise -> V.maximum
        Minimise -> V.minimum


-- psiVValueWith :: (MonadIO m) => LookupType -> ARAL s as -> StateFeatures -> ActionIndex -> m Double
-- psiVValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies . psiV)

wValue :: (MonadIO m) => AgentType -> ARAL s as -> State s -> AgentActionIndices -> m Value
wValue agTp borl state a = wValueWith agTp Worker borl (ftExt state) a
  where
    ftExt = borl ^. featureExtractor


wValueFeat :: (MonadIO m) => AgentType -> ARAL s as -> StateFeatures -> AgentActionIndices -> m Value
wValueFeat agTp = wValueWith agTp Worker

wValueWith :: (MonadIO m) => AgentType -> LookupType -> ARAL s as -> StateFeatures -> AgentActionIndices -> m Value
wValueWith agTp lkTp borl state a = P.lookupProxy agTp (borl ^. t) lkTp (state, a) (borl ^. proxies . w)

wStateValue :: (MonadIO m) => AgentType -> ARAL s as -> (StateFeatures, DisallowedActionIndicies) -> m Value
wStateValue agTp borl (state, asIdxes) = reduceValues maxOrMin <$> lookupState agTp Target (state, asIdxes) (borl ^. proxies . w)
  -- V.mapM (wValueWith Target borl state) asIdxes
  where
    maxOrMin =
      case borl ^. objective of
        Maximise -> V.maximum
        Minimise -> V.minimum


-- | Calculates the expected discounted value with the provided gamma (small/big).
rValue :: (MonadIO m) => AgentType -> ARAL s as -> RSize -> State s -> AgentActionIndices -> m Value
rValue agTp borl size s aNr = rValueWith agTp Worker borl size (ftExt s) aNr
  where ftExt = borl ^. featureExtractor

-- | Calculates the expected discounted value with the provided gamma (small/big).
rValueAgentWith :: (MonadIO m) => AgentType -> LookupType -> ARAL s as -> RSize -> AgentNumber -> State s -> ActionIndex -> m Double
rValueAgentWith agTp lkTp borl size agent s aNr = P.lookupProxyAgent agTp (borl ^. t) lkTp agent (ftExt s, aNr) mr
  where
    ftExt = borl ^. featureExtractor
    mr =
      case size of
        RSmall -> borl ^. proxies . r0
        RBig -> borl ^. proxies . r1


-- | Calculates the expected discounted value with the provided gamma (small/big).
rValueWith :: (MonadIO m) => AgentType -> LookupType -> ARAL s as -> RSize -> StateFeatures -> VB.Vector ActionIndex -> m Value
rValueWith agTp lkTp borl size state a = P.lookupProxy agTp (borl ^. t) lkTp (state, a) mr
  where
    mr =
      case size of
        RSmall -> borl ^. proxies.r0
        RBig   -> borl ^. proxies.r1

-- | For DEBUGGING only! Same as rValueWith but without unscaling.
rValueNoUnscaleWith :: (MonadIO m) => AgentType -> LookupType -> ARAL s as -> RSize -> StateFeatures -> AgentActionIndices -> m Value
rValueNoUnscaleWith agTp lkTp borl size state a = P.lookupProxyNoUnscale agTp (borl ^. t) lkTp (state, a) mr
  where
    mr =
      case size of
        RSmall -> borl ^. proxies.r0
        RBig   -> borl ^. proxies.r1


rStateValueWith :: (MonadIO m) => AgentType -> LookupType -> ARAL s as -> RSize -> (StateFeatures, DisallowedActionIndicies) -> m Value
rStateValueWith agTp lkTp borl size (state, actIdxes) = reduceValues maxOrMin <$> lookupState agTp lkTp (state, actIdxes) mr
  where
    maxOrMin =
      case borl ^. objective of
        Maximise -> V.maximum
        Minimise -> V.minimum
    mr =
      case size of
        RSmall -> borl ^. proxies . r0
        RBig -> borl ^. proxies . r1

-- | Calculates the difference between the expected discounted values: e_gamma0 - e_gamma1 (Small-Big).
eValue :: (MonadIO m) => AgentType -> ARAL s as -> s -> AgentActionIndices -> m Value
eValue agTp borl state act = eValueFeat agTp borl (borl ^. featureExtractor $ state, act)

-- | Calculates the difference between the expected discounted values: e_gamma0 - e_gamma1 (Small-Big).
eValueFeat :: (MonadIO m) => AgentType -> ARAL s as -> (StateFeatures, AgentActionIndices) -> m Value
eValueFeat agTp borl (stateFeat, act) = do
  big <- rValueWith agTp Target borl RBig stateFeat act
  small <- rValueWith agTp Target borl RSmall stateFeat act
  return $ small - big

-- | Calculates the difference between the expected discounted values: e_gamma1 - e_gamma0 - avgRew * (1/(1-gamma1)+1/(1-gamma0)).
eValueAvgCleanedFeat :: (MonadIO m) => AgentType -> ARAL s as -> StateFeatures -> AgentActionIndices -> m Value
eValueAvgCleanedFeat agTp borl state act =
  case borl ^. algorithm of
    AlgNBORL gamma0 gamma1 _ _ -> avgRewardClean gamma0 gamma1
    AlgARAL gamma0 gamma1 _ -> avgRewardClean gamma0 gamma1
    _ -> error "eValueAvgCleaned can only be used with AlgNBORL in Calculation.Ops"
  where
    avgRewardClean gamma0 gamma1 = do
      rBig <- rValueWith agTp Target borl RBig state act
      rSmall <- rValueWith agTp Target borl RSmall state act
      rhoVal <- rhoValueWith agTp Worker borl state act
      return $ rBig - rSmall - rhoVal *. (1 / (1 - gamma1) - 1 / (1 - gamma0))
    agents = borl ^. settings . independentAgents


-- | Calculates the difference between the expected discounted values: e_gamma1 - e_gamma0 - avgRew * (1/(1-gamma1)+1/(1-gamma0)).
eValueAvgCleaned :: (MonadIO m) => AgentType -> ARAL s as -> s -> AgentActionIndices -> m Value
eValueAvgCleaned agTp borl state = eValueAvgCleanedFeat agTp borl sFeat
  where
    sFeat = (borl ^. featureExtractor) state

-- | Calculates the difference between the expected discounted values: e_gamma1 - e_gamma0 - avgRew * (1/(1-gamma1)+1/(1-gamma0)).
eValueAvgCleanedAgent :: (MonadIO m) => AgentType -> ARAL s as -> AgentNumber -> s -> ActionIndex -> m Double
eValueAvgCleanedAgent agTp borl agent state act =
  case borl ^. algorithm of
    AlgNBORL gamma0 gamma1 _ _ -> avgRewardClean gamma0 gamma1
    AlgARAL gamma0 gamma1 _ -> avgRewardClean gamma0 gamma1
    _ -> error "eValueAvgCleaned can only be used with AlgNBORL in Calculation.Ops"
  where
    avgRewardClean gamma0 gamma1 = do
      rBig <- rValueAgentWith agTp Target borl RBig agent state act
      rSmall <- rValueAgentWith agTp Target borl RSmall agent state act
      rhoVal <- rhoValueAgentWith agTp Worker borl agent state act
      return $ rBig - rSmall - rhoVal * (1 / (1 - gamma1) - 1 / (1 - gamma0))
    agents = borl ^. settings . independentAgents
