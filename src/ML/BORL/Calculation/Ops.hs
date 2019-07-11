{-# LANGUAGE CPP                 #-}
{-# LANGUAGE ScopedTypeVariables #-}
module ML.BORL.Calculation.Ops
    ( mkCalculation
    , expSmthPsi
    , keepXLastValues
    , approxAvg
    , rValue
    , rValueWith
    , rStateValue
    , eValue
    , wStateValue
    , wValueWith
    , wValue
    , vStateValue
    , vValue
    , rhoStateValue
    , rhoValueWith
    , rhoValue
    , rhoMinimumValueWith
    , rhoMinimumValue
    , RSize (..)
    ) where

import           ML.BORL.Algorithm
import           ML.BORL.Calculation.Type
import           ML.BORL.Parameters
import           ML.BORL.Properties
import           ML.BORL.Proxy               as P
import           ML.BORL.Reward
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Lens
import           Control.Parallel.Strategies hiding (r0)


-- | Used to select a discount factor.
data RSize
  = RSmall
  | RBig


expSmthPsi :: Double
expSmthPsi = 0.03

keepXLastValues :: Int
keepXLastValues = 100

approxAvg :: Double
approxAvg = fromIntegral (100 :: Int)


mkCalculation :: (MonadBorl' m, Ord s) => BORL s -> (StateFeatures, [ActionIndex]) -> ActionIndex -> Bool -> RewardValue -> (StateNextFeatures, [ActionIndex]) -> EpisodeEnd -> m Calculation
mkCalculation borl state aNr randomAction reward stateNext episodeEnd =
  mkCalculation' borl state aNr randomAction reward stateNext episodeEnd (borl ^. algorithm)


mkCalculation' :: (MonadBorl' m, Ord s) => BORL s -> (StateFeatures, [ActionIndex]) -> ActionIndex -> Bool -> RewardValue -> (StateNextFeatures, [ActionIndex]) -> EpisodeEnd -> Algorithm -> m Calculation
mkCalculation' borl (state, stateActIdxes) aNr randomAction reward (stateNext, stateNextActIdxes) episodeEnd (AlgBORL ga0 ga1 avgRewardType stValHandling decideOnVPlusPsiV) = do
  let params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
  let alp = params' ^. alpha
      bta = params' ^. beta
      dlt = params' ^. delta
      gam = params' ^. gamma
      xiVal = params' ^. xi
      period = borl ^. t
      (psiValRho, psiValV, psiValW) = borl ^. psis -- exponentially smoothed Psis
  let label = (state, aNr)
      epsEnd
        | episodeEnd = 0
        | otherwise = 1
  let lastRews' =
        case avgRewardType of
          ByMovAvg movAvgLen -> take movAvgLen $ reward : borl ^. lastRewards
          _                  -> take keepXLastValues $ reward : borl ^. lastRewards
  rhoMinimumState <- rhoMinimumValueFeat borl state aNr `using` rpar
  vValState <- vValueFeat False borl state aNr `using` rpar
  vValStateNext <- vStateValue decideOnVPlusPsiV borl (stateNext, stateNextActIdxes)  `using` rpar
  rhoVal <- rhoValueFeat borl state aNr `using` rpar
  wValState <- wValueFeat borl state aNr `using` rpar
  wValStateNext <- wStateValue borl (state, stateActIdxes) `using` rpar
  r0ValState <- rValueFeat borl RSmall state aNr `using` rpar
  r1ValState <- rValueFeat borl RBig state aNr `using` rpar
  psiVTblVal <- P.lookupProxy period Worker label (borl ^. proxies . psiV) `using` rpar
  rhoState <-
    if isUnichain borl
      then case avgRewardType of
             Fixed x       -> return x
             ByMovAvg l    -> return $ sum lastRews' / fromIntegral l -- (length lastRews')
             ByReward      -> return reward
             ByStateValues -> return (reward + vValStateNext - vValState)
      else do
        rhoStateValNext <- rhoStateValue borl (stateNext, stateNextActIdxes)
        return $ (epsEnd * approxAvg * rhoStateValNext + reward) / (epsEnd * approxAvg + 1) -- approximation
  let rhoVal' =
        max rhoMinimumState $
        case avgRewardType of
          ByMovAvg _ -> rhoState
          Fixed x    -> x
          _          -> (1 - alp) * rhoVal + alp * rhoState
  let rhoMinimumVal'
        | rhoState < rhoMinimumState = rhoMinimumState
        | otherwise = (1 - expSmthPsi / 200) * rhoMinimumState + expSmthPsi / 200 * rhoState
  let psiRho = rhoVal' - rhoVal -- should converge to 0
  let vValState' = (1 - bta) * vValState + bta * (reward - rhoVal' + epsEnd * vValStateNext)
      psiV = reward + vValStateNext - rhoVal' - vValState' -- should converge towards 0
      lastVs' =
        case stValHandling of
          Normal -> take keepXLastValues $ vValState' : borl ^. lastVValues
          DivideValuesAfterGrowth nr _ -> take nr $ vValState' : borl ^. lastVValues
  let wValState' = (1 - dlt) * wValState + dlt * (-vValState' + epsEnd * wValStateNext)
  let psiW = wValStateNext - vValState' - wValState' -- should converge towards 0
  let randAct
        | randomAction = 0
        | otherwise = 1
  let psiValRho' = (1 - expSmthPsi) * psiValRho + expSmthPsi * randAct * abs psiRho
      psiValV' = (1 - expSmthPsi) * psiValV + expSmthPsi * randAct * abs psiV
      psiValW' = (1 - expSmthPsi) * psiValW + expSmthPsi * randAct * abs psiW
  let psiVTblVal' = (1 - expSmthPsi) * psiVTblVal + expSmthPsi * psiV
  let vValStateNew -- enforce bias optimality (correction of V(s,a) values)
         --  | randomAction && (psiV > eps || (psiV <= eps && psiV > -eps && psiW > eps)) = vValState' -- interesting action
         --  | randomAction = vValState' -- psiW and psiV should not be 0!
        | abs psiV < abs psiW = (1 - xiVal) * vValState' + xiVal * (vValState' + clip (abs vValState') psiW)
        | otherwise = (1 - xiVal) * vValState' + xiVal * (vValState' + clip (abs vValState') psiV)
      clip minmax val = max (-minmax) $ min minmax val
   -- R0/R1
  rSmall <- rStateValue borl RSmall (stateNext, stateNextActIdxes)
  rBig <- rStateValue borl RBig (stateNext, stateNextActIdxes)
  let r0ValState' = (1 - gam) * r0ValState + gam * (reward + epsEnd * ga0 * rSmall)
  let r1ValState' = (1 - gam) * r1ValState + gam * (reward + epsEnd * ga1 * rBig)
  return $
    Calculation
      (Just rhoMinimumVal')
      (Just rhoVal')
      (Just psiVTblVal')
      (Just vValStateNew)
      (Just wValState')
      (Just r0ValState')
      r1ValState'
      (Just psiValRho')
      (Just psiValV')
      (Just psiValW')
      (Just lastVs')
      lastRews'
      episodeEnd

mkCalculation' borl (state, _) aNr randomAction reward (stateNext, stateNextActIdxes) episodeEnd (AlgDQN ga) = do
  let params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
  let gam = params' ^. gamma
  let epsEnd
        | episodeEnd = 0
        | otherwise = 1
  let lastRews' = take keepXLastValues $ reward : borl ^. lastRewards
  r1ValState <- rValueFeat borl RBig state aNr `using` rpar
  rBig <- rStateValue borl RBig (stateNext, stateNextActIdxes) `using` rpar
  let r1ValState' = (1 - gam) * r1ValState + gam * (reward + epsEnd * ga * rBig)
  return $ Calculation Nothing Nothing Nothing Nothing Nothing Nothing r1ValState' Nothing Nothing Nothing Nothing lastRews' episodeEnd

mkCalculation' borl (state,stateActIdxes) aNr randomAction reward (stateNext,stateNextActIdxes) episodeEnd (AlgDQNAvgRew ga avgRewardType) = do
  let params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
  let alp = params' ^. alpha
      gam = params' ^. gamma
  let epsEnd
        | episodeEnd = 0
        | otherwise = 1
  rhoVal <- rhoValueFeat borl state aNr `using` rpar
  let lastRews' =
        case avgRewardType of
          ByMovAvg movAvgLen -> take movAvgLen $ reward : borl ^. lastRewards
          _                  -> take keepXLastValues $ reward : borl ^. lastRewards
  rhoMinimumState <- rhoMinimumValueFeat borl state aNr `using` rpar
  rhoState <-
    if isUnichain borl
      then case avgRewardType of
             Fixed x -> return x
             ByMovAvg l -> return $ sum lastRews' / fromIntegral l -- (length lastRews')
             ByReward -> return reward
             ByStateValues -> error "Average reward using `ByStateValues` not supported for AlgDQNAvgRew"
      else do
        rhoStateValNext <- rhoStateValue borl (stateNext, stateNextActIdxes)
        return $ (epsEnd * approxAvg * rhoStateValNext + reward) / (epsEnd * approxAvg + 1) -- approximation
  let rhoVal' =
        max rhoMinimumState $
        case avgRewardType of
          ByMovAvg _ -> rhoState
          Fixed x    -> x
          _          -> (1 - alp) * rhoVal + alp * rhoState
  let rhoMinimumVal'
        | rhoState < rhoMinimumState = rhoMinimumState
        | otherwise = (1 - expSmthPsi / 200) * rhoMinimumState + expSmthPsi / 200 * rhoState
  let lastRews' = take keepXLastValues $ reward : borl ^. lastRewards
  r1ValState <- rValueFeat borl RBig state aNr `using` rpar
  rBig <- rStateValue borl RBig (stateNext, stateNextActIdxes) `using` rpar
  let r1ValState' = (1 - gam) * r1ValState + gam * (reward - rhoVal' + epsEnd * ga * rBig)
  return $ Calculation (Just rhoMinimumVal') (Just rhoVal') Nothing Nothing Nothing Nothing r1ValState' Nothing Nothing Nothing Nothing lastRews' episodeEnd

-- TODO maybe integrate learnRandomAbove, etc.:
  -- let borl'
  --       | randomAction && params' ^. exploration <= borl ^. parameters . learnRandomAbove = borl -- multichain ?
  --       | otherwise = set v mv' $ set w mw' $ set rho rhoNew $ set r0 mr0' $ set r1 mr1' borl


-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
rhoMinimumValue :: (MonadBorl' m) => BORL s -> State s -> ActionIndex -> m Double
rhoMinimumValue borl state a = rhoMinimumValueWith Worker borl (ftExt state) a
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
    _        -> maximum <$> mapM (rhoValueWith Target borl state) actIdxes -- (map fst $ actionsIndexed borl state)

vValue :: (MonadBorl' m) => Bool -> BORL s -> State s -> ActionIndex -> m Double
vValue addPsiV borl s a = vValueWith Worker addPsiV borl (ftExt s) a
  where
    ftExt = borl ^. featureExtractor

vValueFeat :: (MonadBorl' m) => Bool -> BORL s -> StateFeatures -> ActionIndex -> m Double
vValueFeat = vValueWith Worker

vValueWith :: (MonadBorl' m) => LookupType -> Bool -> BORL s -> StateFeatures -> ActionIndex -> m Double
vValueWith lkTp addPsiV borl state a = do
  vVal <- P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies . v)
  ~psiV <-
    if addPsiV
      then P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies . psiV)
      else return 0
  return (vVal + psiV)

vStateValue :: (MonadBorl' m, Ord s) => Bool -> BORL s -> (StateFeatures, [ActionIndex]) -> m Double
vStateValue addPsiV borl (state, asIdxes) = maximum <$> mapM (vValueWith Target addPsiV borl state) asIdxes
                                                                                                            -- (map fst $ actionsIndexed borl state)


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
                                                                                            -- (map fst $ actionsIndexed borl state)


-- | Calculates the expected discounted value with the provided gamma (small/big).
rValue :: (MonadBorl' m) => BORL s -> RSize -> State s -> ActionIndex -> m Double
rValue borl size s a = rValueWith Worker borl size (ftExt s) a
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
rStateValue borl size (state, actIdxes) = maximum <$> mapM (rValueWith Target borl size state) actIdxes -- (actionsIndexed borl state)

-- | Calculates the difference between the expected discounted values.
eValue :: (MonadBorl' m) => BORL s -> s -> ActionIndex -> m Double
eValue borl state act = do
  big <- rValueWith Target borl RBig (ftExtBig state) act
  small <- rValueWith Target borl RSmall (ftExtSmall state) act
  return $ big - small

  where ftExtSmall = borl ^. featureExtractor
        ftExtBig = borl ^. featureExtractor


--  | Calculates the difference between the expected discounted values.
-- eStateValue :: (Ord s) => BORL s -> s -> Double
-- eStateValue borl state = maximum (map (rValueWith Target borl RBig state) as) - reduce (map (rValueWith Target borl RSmall state) as)
--   where as = actionsIndexed borl state
