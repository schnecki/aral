{-# LANGUAGE CPP                 #-}
{-# LANGUAGE ScopedTypeVariables #-}
module ML.BORL.Calculation.Ops
    ( mkCalculation
    , rValue
    , eValue
    , eValueAvgCleaned
    , vValue
    , vValueFeat
    , wValueFeat
    , rValueFeat
    , rhoValue
    , RSize (..)
    ) where

import           Control.DeepSeq
import           ML.BORL.Algorithm
import           ML.BORL.Calculation.Type
import           ML.BORL.Decay                  (DecaySetup (..), decaySetup,
                                                 exponentialDecayValue)
import           ML.BORL.NeuralNetwork.NNConfig
import           ML.BORL.NeuralNetwork.Scaling  (scaleMaxVValue, scaleMinVValue)
import           ML.BORL.Parameters
import           ML.BORL.Properties
import           ML.BORL.Proxy                  as P
import           ML.BORL.Reward
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Arrow                  (first)
import           Control.Lens
import           Control.Monad                  (when)
import           Control.Monad.IO.Class         (liftIO)
import           Control.Parallel.Strategies    hiding (r0)
import           Data.Function                  (on)
import           Data.List                      (maximumBy, minimumBy, sortBy)
import           Data.Maybe                     (fromMaybe)
import           System.Random                  (randomRIO)


-- | Used to select a discount factor.
data RSize
  = RSmall
  | RBig


expSmthPsi :: Double
expSmthPsi = 0.001

keepXLastValues :: Int
keepXLastValues = 100

approxAvg :: Double
approxAvg = fromIntegral (100 :: Int)


mkCalculation :: (MonadBorl' m) => BORL s -> (StateFeatures, [ActionIndex]) -> ActionIndex -> Bool -> RewardValue -> (StateNextFeatures, [ActionIndex]) -> EpisodeEnd -> m Calculation
mkCalculation borl state aNr randomAction reward stateNext episodeEnd =
  mkCalculation' borl state aNr randomAction reward stateNext episodeEnd (borl ^. algorithm)

ite :: Bool -> p -> p -> p
ite True thenPart _  = thenPart
ite False _ elsePart = elsePart
{-# INLINE ite #-}

rhoMinimumState' :: (Ord a, Fractional a) => a -> a
rhoMinimumState' rhoVal' = max (rhoVal' - 2) (0.975 * rhoVal')

mkCalculation' ::
     (MonadBorl' m)
  => BORL s
  -> (StateFeatures, [ActionIndex])
  -> ActionIndex
  -> Bool
  -> RewardValue
  -> (StateNextFeatures, [ActionIndex])
  -> EpisodeEnd
  -> Algorithm NetInputWoAction
  -> m Calculation
mkCalculation' borl (state, stateActIdxes) aNr randomAction reward (stateNext, stateNextActIdxes) episodeEnd (AlgBORL ga0 ga1 avgRewardType mRefState) = do
  let params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
  let isRefState = mRefState == Just (state, aNr)
  let getExpSmthParam p paramANN param
        | isANN && useOne = 1
        | isANN = params' ^. paramANN
        | otherwise = params' ^. param
        where
          isANN = P.isNeuralNetwork px && borl ^. t >= px ^?! proxyNNConfig . replayMemoryMaxSize
          useOne = px ^?! proxyNNConfig . setExpSmoothParamsTo1
          px = borl ^. proxies . p
  let alp = getExpSmthParam rho alphaANN alpha
      bta = getExpSmthParam v betaANN beta
      dltW = getExpSmthParam w deltaANN delta
      gamR0 = getExpSmthParam r0 gammaANN gamma
      gamR1 = getExpSmthParam r1 gammaANN gamma
      xiVal = params' ^. xi
      zetaVal = params' ^. zeta
      period = borl ^. t
      (psiValRho, psiValV, psiValW) = borl ^. psis -- exponentially smoothed Psis
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
          ByMovAvg movAvgLen -> take movAvgLen $ reward : borl ^. lastRewards
          _                  -> take keepXLastValues $ reward : borl ^. lastRewards
  vValState <- vValueFeat borl state aNr `using` rpar
  rhoMinimumState <- rhoMinimumValueFeat borl state aNr `using` rpar
  vValStateNext <- snd <$> vStateValue borl (stateNext, stateNextActIdxes) `using` rpar
  rhoVal <- rhoValueFeat borl state aNr `using` rpar
  wValState <- wValueFeat borl state aNr `using` rpar
  wValStateNext <- wStateValue borl (stateNext, stateNextActIdxes) `using` rpar
  r0ValState <- rValueFeat borl RSmall state aNr `using` rpar
  r1ValState <- rValueFeat borl RBig state aNr `using` rpar
  psiVState <- P.lookupProxy period Worker label (borl ^. proxies . psiV) `using` rpar
  psiWState <- P.lookupProxy period Worker label (borl ^. proxies . psiW) `using` rpar
  -- Stabilization
  let mStabVal = borl ^? proxies . v . proxyNNConfig . stabilizationAdditionalRho
      mStabValDec = borl ^? proxies . v . proxyNNConfig . stabilizationAdditionalRhoDecay
      stabilization = fromMaybe 0 $ decaySetup <$> mStabValDec <*> pure period <*> mStabVal
  -- Rho
  rhoState <-
    case avgRewardType of
      Fixed x -> return x
      ByMovAvg l
        | isUnichain borl -> return $ sum lastRews' / fromIntegral l
      ByMovAvg _ -> error "ByMovAvg is not allowed in multichain setups"
      ByReward -> return reward
      ByStateValues -> return $ reward + vValStateNext - vValState
      ByStateValuesAndReward ratio decay -> return $ ratio' * (reward + vValStateNext - vValState) + (1 - ratio') * reward
        where ratio' = decaySetup decay period ratio
  let rhoVal'
        | randomAction && not learnFromRandom = rhoVal
        | otherwise =
          max rhoMinimumState $
          case avgRewardType of
            ByMovAvg _ -> rhoState
            Fixed x    -> x
            _          -> (1 - alp) * rhoVal + alp * rhoState
  -- RhoMin
  let rhoMinimumVal'
        | rhoState < rhoMinimumState = rhoMinimumState
        | otherwise = max rhoMinimumState $ (1 - expSmthPsi / 50) * rhoMinimumState + expSmthPsi / 50 * rhoMinimumState' rhoVal'
  -- PsiRho (should converge to 0)
  psiRho <- ite (isUnichain borl) (return $ rhoVal' - rhoVal) (subtract rhoVal' <$> rhoStateValue borl (stateNext, stateNextActIdxes))
  -- V
  let vValState' = (1 - bta) * vValState + bta * (reward - rhoVal' + epsEnd * vValStateNext + nonRandAct * (psiVState + zetaVal * psiWState) - stabilization)
      psiV = reward + vValStateNext - rhoVal' - vValState' -- should converge to 0
      psiVState' = (1 - xiVal * bta) * psiVState + bta * xiVal * psiV
  -- LastVs
  let lastVs' = take keepXLastValues $ vValState' : borl ^. lastVValues
  -- W
  let wValState'
        | isRefState = 0
        | otherwise = (1 - dltW) * wValState + dltW * (-vValState' + epsEnd * wValStateNext + nonRandAct * psiWState - stabilization)
      psiW = wValStateNext - vValState' - wValState'
      psiWState'
        | isRefState = 0
        | otherwise = (1 - xiVal * dltW) * psiWState + dltW * xiVal * psiW
  -- R0/R1
  rSmall <- rStateValue borl RSmall (stateNext, stateNextActIdxes)
  rBig <- rStateValue borl RBig (stateNext, stateNextActIdxes)
  let r0ValState' = (1 - gamR0) * r0ValState + gamR0 * (reward + epsEnd * ga0 * rSmall)
  let r1ValState' = (1 - gamR1) * r1ValState + gamR1 * (reward + epsEnd * ga1 * rBig)
  -- Psis Scalar calues for output only
  let psiValRho' = (1 - expSmth) * psiValRho + expSmth * abs psiRho
  let psiValV' = (1 - expSmth) * psiValV + expSmth * abs psiVState'
  let psiValW' = (1 - expSmth) * psiValW + expSmth * abs psiWState'

  return $
    Calculation
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
      , getLastVs' = force $ Just lastVs'
      , getLastRews' = force lastRews'
      , getEpisodeEnd = episodeEnd
      }

mkCalculation' borl (state, _) aNr randomAction reward (stateNext, stateNextActIdxes) episodeEnd (AlgBORLVOnly avgRewardType mRefState) = do
  let params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
  let learnFromRandom = params' ^. exploration > params' ^. learnRandomAbove
      period = borl ^. t
  let getExpSmthParam p paramANN param
        | isANN && useOne = 1
        | isANN = params' ^. paramANN
        | otherwise = params' ^. param
        where
          isANN = P.isNeuralNetwork px && borl ^. t >= px ^?! proxyNNConfig . replayMemoryMaxSize
          useOne = px ^?! proxyNNConfig . setExpSmoothParamsTo1
          px = borl ^. proxies . p
      alp = getExpSmthParam rho alphaANN alpha
      bta = getExpSmthParam v betaANN beta
  let epsEnd
        | episodeEnd = 0
        | otherwise = 1
  rhoVal <- rhoValueFeat borl state aNr `using` rpar
  vValState <- vValueFeat borl state aNr `using` rpar
  (_, vValStateNext) <- vStateValue borl (stateNext, stateNextActIdxes) `using` rpar
  let lastRews' =
        case avgRewardType of
          ByMovAvg movAvgLen -> take movAvgLen $ reward : borl ^. lastRewards
          _                  -> take keepXLastValues $ reward : borl ^. lastRewards
  rhoMinimumState <- rhoMinimumValueFeat borl state aNr `using` rpar
  rhoState <-
    case avgRewardType of
      Fixed x -> return x
      ByMovAvg _ -> return $ sum lastRews' / fromIntegral (length lastRews')
      ByReward -> return reward
      ByStateValues -> return $ reward + vValStateNext - vValState
      ByStateValuesAndReward ratio decay -> return $ ratio' * (reward + vValStateNext - vValState) + (1 - ratio') * reward
        where ratio' = decaySetup decay (borl ^. t) ratio
  let rhoVal'
        | randomAction && not learnFromRandom = rhoVal
        | otherwise =
          max rhoMinimumState $
          case avgRewardType of
            ByMovAvg _ -> rhoState
            Fixed x    -> x
            _          -> (1 - alp) * rhoVal + alp * rhoState
  let rhoMinimumVal'
        | rhoState < rhoMinimumState = rhoMinimumState
        | otherwise = max rhoMinimumState $ (1 - expSmthPsi / 50) * rhoMinimumState + expSmthPsi / 50 * rhoMinimumState' rhoVal'
  let vValState' = (1 - bta) * vValState + bta * (reward - rhoVal' + epsEnd * vValStateNext)
  let lastVs' = take keepXLastValues $ vValState' : borl ^. lastVValues
  return $
    Calculation
      { getRhoMinimumVal' = Just rhoMinimumVal'
      , getRhoVal' = Just rhoVal'
      , getPsiVValState' = Nothing
      , getVValState' = Just $ ite (mRefState == Just (state, aNr)) 0 vValState'
      , getPsiWValState' = Nothing
      , getWValState' = Nothing
      , getR0ValState' = Nothing
      , getR1ValState' = Nothing
      , getPsiValRho' = Nothing
      , getPsiValV' = Nothing
      , getPsiValW' = Nothing
      , getLastVs' = Just lastVs'
      , getLastRews' = lastRews'
      , getEpisodeEnd = episodeEnd
      }

mkCalculation' borl (state, _) aNr _ reward (stateNext, stateNextActIdxes) episodeEnd (AlgDQN ga) = do
  let params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
  let getExpSmthParam p paramANN param
        | isANN && useOne = 1
        | isANN = params' ^. paramANN
        | otherwise = params' ^. param
        where
          isANN = P.isNeuralNetwork px && borl ^. t >= px ^?! proxyNNConfig . replayMemoryMaxSize
          useOne = px ^?! proxyNNConfig . setExpSmoothParamsTo1
          px = borl ^. proxies . p
      gam = getExpSmthParam r1 gammaANN gamma
  let epsEnd
        | episodeEnd = 0
        | otherwise = 1
  let lastRews' = take keepXLastValues $ reward : borl ^. lastRewards
  r1ValState <- rValueFeat borl RBig state aNr `using` rpar
  r1StateNext <- rStateValue borl RBig (stateNext, stateNextActIdxes) `using` rpar
  let r1ValState' = (1 - gam) * r1ValState + gam * (reward + epsEnd * ga * r1StateNext)
  return $
    Calculation
      { getRhoMinimumVal' = Nothing
      , getRhoVal' = Nothing
      , getPsiVValState' = Nothing
      , getVValState' = Nothing
      , getPsiWValState' = Nothing
      , getWValState' = Nothing
      , getR0ValState' = Nothing
      , getR1ValState' = Just r1ValState'
      , getPsiValRho' = Nothing
      , getPsiValV' = Nothing
      , getPsiValW' = Nothing
      , getLastVs' = Nothing
      , getLastRews' = lastRews'
      , getEpisodeEnd = episodeEnd
      }
mkCalculation' borl (state, _) aNr randomAction reward (stateNext, stateNextActIdxes) episodeEnd (AlgDQNAvgRewardFree ga0 ga1 avgRewardType) = do
  rhoMinimumState <- rhoMinimumValueFeat borl state aNr `using` rpar
  rhoVal <- rhoValueFeat borl state aNr `using` rpar
  r0ValState <- rValueFeat borl RSmall state aNr `using` rpar
  r0StateNext <- rStateValue borl RSmall (stateNext, stateNextActIdxes) `using` rpar
  r1ValState <- rValueFeat borl RBig state aNr `using` rpar
  r1StateNext <- rStateValue borl RBig (stateNext, stateNextActIdxes) `using` rpar
  let params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
  let learnFromRandom = params' ^. exploration > params' ^. learnRandomAbove
      period = borl ^. t
  let getExpSmthParam p paramANN param
        | isANN && useOne = 1
        | isANN = params' ^. paramANN
        | otherwise = params' ^. param
        where
          isANN = P.isNeuralNetwork px && borl ^. t >= px ^?! proxyNNConfig . replayMemoryMaxSize
          useOne = px ^?! proxyNNConfig . setExpSmoothParamsTo1
          px = borl ^. proxies . p
      alp = getExpSmthParam rho alphaANN alpha
      gam = getExpSmthParam r1 gammaANN gamma
  let epsEnd
        | episodeEnd = 0
        | otherwise = 1
  let lastRews' =
        case avgRewardType of
          ByMovAvg movAvgLen -> take movAvgLen $ reward : borl ^. lastRewards
          _                  -> take keepXLastValues $ reward : borl ^. lastRewards
  -- Rho
  rhoState <-
    case avgRewardType of
      Fixed x -> return x
      ByMovAvg l -> return $ sum lastRews' / fromIntegral l
      ByReward -> return reward
      ByStateValues -> return $ reward + r1StateNext - r1ValState
      ByStateValuesAndReward ratio decay -> return $ ratio' * (reward + r1StateNext - r1ValState) + (1 - ratio') * reward
        where ratio' = decaySetup decay (borl ^. t) ratio
  let rhoVal'
        | randomAction -- && not learnFromRandom
        = rhoVal
        | otherwise =
          max rhoMinimumState $
          case avgRewardType of
            ByMovAvg _ -> rhoState
            Fixed x    -> x
            _          -> (1 - alp) * rhoVal + alp * rhoState
  -- RhoMin
  let rhoMinimumVal'
        | rhoState < rhoMinimumState = rhoMinimumState
        | otherwise = max rhoMinimumState $ (1 - expSmthPsi / 50) * rhoMinimumState + expSmthPsi / 50 * rhoMinimumState' rhoVal'
  let r0ValState' = (1 - gam) * r0ValState + gam * (reward + epsEnd * ga0 * r0StateNext - rhoVal')
  let r1ValState' = (1 - gam) * r1ValState + gam * (reward + epsEnd * ga1 * r1StateNext - rhoVal')
  return $
    Calculation
      { getRhoMinimumVal' = Just rhoMinimumVal'
      , getRhoVal' = Just rhoVal'
      , getPsiVValState' = Nothing
      , getVValState' = Nothing
      , getPsiWValState' = Nothing
      , getWValState' = Nothing
      , getR0ValState' = Just r0ValState'
      , getR1ValState' = Just r1ValState'
      , getPsiValRho' = Nothing
      , getPsiValV' = Nothing
      , getPsiValW' = Nothing
      , getLastVs' = Nothing
      , getLastRews' = lastRews'
      , getEpisodeEnd = episodeEnd
      }


-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
rhoMinimumValue :: (MonadBorl' m) => BORL s -> State s -> ActionIndex -> m Double
rhoMinimumValue borl state = rhoMinimumValueWith Worker borl (ftExt state)
  where
    ftExt = borl ^. featureExtractor

rhoMinimumValueFeat :: (MonadBorl' m) => BORL s -> StateFeatures -> ActionIndex -> m Double
rhoMinimumValueFeat = rhoMinimumValueWith Worker

rhoMinimumValueWith :: (MonadBorl' m) => LookupType -> BORL s -> StateFeatures -> ActionIndex -> m Double
rhoMinimumValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state,a) (borl ^. proxies.rhoMinimum)

-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
rhoValue :: (MonadBorl' m) => BORL s -> State s -> ActionIndex -> m Double
rhoValue borl s a = rhoValueWith Worker borl (ftExt s) a
  where
    ftExt = borl ^. featureExtractor

rhoValueFeat :: (MonadBorl' m) => BORL s -> StateFeatures -> ActionIndex -> m Double
rhoValueFeat = rhoValueWith Worker

rhoValueWith :: (MonadBorl' m) => LookupType -> BORL s -> StateFeatures -> ActionIndex -> m Double
rhoValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state,a) (borl ^. proxies.rho)

rhoStateValue :: (MonadBorl' m) => BORL s -> (StateFeatures, [ActionIndex]) -> m Double
rhoStateValue borl (state, actIdxes) =
  case borl ^. proxies . rho of
    Scalar r -> return r
    _        -> maximum <$> mapM (rhoValueWith Target borl state) actIdxes

vValue :: (MonadBorl' m) => BORL s -> State s -> ActionIndex -> m Double
vValue borl s a = vValueWith Worker borl (ftExt s) a
  where
    ftExt = borl ^. featureExtractor

vValueFeat :: (MonadBorl' m) => BORL s -> StateFeatures -> ActionIndex -> m Double
vValueFeat = vValueWith Worker

vValueWith :: (MonadBorl' m) => LookupType -> BORL s -> StateFeatures -> ActionIndex -> m Double
vValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies . v)

vStateValue :: (MonadBorl' m) => BORL s -> (StateFeatures, [ActionIndex]) -> m (ActionIndex, Double)
vStateValue borl (state, asIdxes) = do
  xs <- mapM (vValueWith Target borl state) asIdxes
  return $ maximumBy (compare `on` snd) $ zip asIdxes xs

-- psiVValueWith :: (MonadBorl' m) => LookupType -> BORL s -> StateFeatures -> ActionIndex -> m Double
-- psiVValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies . psiV)

wValue :: (MonadBorl' m) => BORL s -> State s -> ActionIndex -> m Double
wValue borl state a = wValueWith Worker borl (ftExt state) a
  where
    ftExt = borl ^. featureExtractor


wValueFeat :: (MonadBorl' m) => BORL s -> StateFeatures -> ActionIndex -> m Double
wValueFeat = wValueWith Worker

wValueWith :: (MonadBorl' m) => LookupType -> BORL s -> StateFeatures -> ActionIndex -> m Double
wValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies . w)

wStateValue :: (MonadBorl' m) => BORL s -> (StateFeatures, [ActionIndex]) -> m Double
wStateValue borl (state, asIdxes) = maximum <$> mapM (wValueWith Target borl state) asIdxes


-- | Calculates the expected discounted value with the provided gamma (small/big).
rValue :: (MonadBorl' m) => BORL s -> RSize -> State s -> ActionIndex -> m Double
rValue borl size s aNr = rValueWith Worker borl size (ftExt s) aNr
  where ftExt = case size of
          RSmall -> borl ^. featureExtractor
          RBig   -> borl ^. featureExtractor

rValueFeat :: (MonadBorl' m) => BORL s -> RSize -> StateFeatures -> ActionIndex -> m Double
rValueFeat = rValueWith Worker


-- | Calculates the expected discounted value with the provided gamma (small/big).
rValueWith :: (MonadBorl' m) => LookupType -> BORL s -> RSize -> StateFeatures -> ActionIndex -> m Double
rValueWith lkTp borl size state a = P.lookupProxy (borl ^. t) lkTp (state, a) mr
  where
    mr =
      case size of
        RSmall -> borl ^. proxies.r0
        RBig   -> borl ^. proxies.r1

rStateValue :: (MonadBorl' m) => BORL s -> RSize -> (StateFeatures, [ActionIndex]) -> m Double
rStateValue borl size (state, actIdxes) = maximum <$> mapM (rValueWith Target borl size state) actIdxes

-- | Calculates the difference between the expected discounted values: e_gamma0 - e_gamma1 (Small-Big).
eValue :: (MonadBorl' m) => BORL s -> s -> ActionIndex -> m Double
eValue borl state act = do
  big <- rValueWith Target borl RBig (ftExtBig state) act
  small <- rValueWith Target borl RSmall (ftExtSmall state) act
  return $ small - big

  where ftExtSmall = borl ^. featureExtractor
        ftExtBig = borl ^. featureExtractor

-- | Calculates the difference between the expected discounted values: e_gamma1 - e_gamma0 - avgRew * (1/(1-gamma1)+1/(1-gamma0)).
eValueAvgCleaned :: (MonadBorl' m) => BORL s -> s -> ActionIndex -> m Double
eValueAvgCleaned borl state act = case borl ^. algorithm of
  AlgBORL gamma0 gamma1 _ _ -> do
    rBig <- rValueWith Target borl RBig (ftExtBig state) act
    rSmall <- rValueWith Target borl RSmall (ftExtSmall state) act
    rhoVal <- rhoValue borl state act
    return $ rBig - rSmall - rhoVal * (1/(1-gamma1) - 1/(1-gamma0))
  _ -> error "eValueAvgCleaned can only be used with AlgBORL in Calculation.Ops"

  where ftExtSmall = borl ^. featureExtractor
        ftExtBig = borl ^. featureExtractor


--  | Calculates the difference between the expected discounted values.
-- eStateValue :: (Ord s) => BORL s -> s -> Double
-- eStateValue borl state = maximum (map (rValueWith Target borl RBig state) as) - reduce (map (rValueWith Target borl RSmall state) as)
--   where as = actionsIndexed borl state
