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
import           Control.Monad               (when)
import           Control.Parallel.Strategies hiding (r0)
import           Data.Function               (on)
import           Data.List                   (sortBy)

import           Debug.Trace


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
      zetaVal = params' ^. zeta
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
  vValState <- vValueFeat False borl state aNr `using` rpar
  rhoMinimumState <- rhoMinimumValueFeat borl state aNr `using` rpar
  vValStateNext <- vStateValue decideOnVPlusPsiV borl (stateNext, stateNextActIdxes) `using` rpar
  rhoVal <- rhoValueFeat borl state aNr `using` rpar
  wValState <- wValueFeat borl state aNr `using` rpar
  wValStateNext <- wStateValue borl (stateNext, stateNextActIdxes) `using` rpar
  r0ValState <- rValueFeat borl RSmall state aNr `using` rpar
  r1ValState <- rValueFeat borl RBig state aNr `using` rpar
  psiVState <- P.lookupProxy period Worker label (borl ^. proxies . psiV) `using` rpar

  -- Rho
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

  -- RhoMin
  let rhoMinimumVal'
        | rhoState < rhoMinimumState = rhoMinimumState
        | otherwise = (1 - expSmthPsi / 200) * rhoMinimumState + expSmthPsi / 200 * rhoVal' -- rhoState
  psiRho <- if isUnichain borl
            then return $ rhoVal' - rhoVal -- should converge to 0
            else do rhoStateValNext <- rhoStateValue borl (stateNext, stateNextActIdxes)
                    return $ rhoStateValNext - rhoVal'

  -- V
  let vValState' = (1 - bta) * vValState + bta * (reward - rhoVal' + epsEnd * vValStateNext)
      psiV = reward - rhoVal' - vValState' + vValStateNext             -- should converge to 0

  let lastVs' =
        case stValHandling of
          Normal -> take keepXLastValues $ vValState' : borl ^. lastVValues
          DivideValuesAfterGrowth nr _ -> take nr $ vValState' : borl ^. lastVValues

  -- W
  let wValState' = (1 - dlt) * wValState + dlt * (-vValState' + epsEnd * wValStateNext)
      -- psiW = vValState' + wValState' - wValStateNext
      psiW = wValStateNext - vValState' - wValState

   -- R0/R1
  rSmall <- rStateValue borl RSmall (stateNext, stateNextActIdxes)
  rBig <- rStateValue borl RBig (stateNext, stateNextActIdxes)
  let r0ValState' = (1 - gam) * r0ValState + gam * (reward + epsEnd * ga0 * rSmall)
  let r1ValState' = (1 - gam) * r1ValState + gam * (reward + epsEnd * ga1 * rBig)

  -- Psis
  let randAct
        | randomAction = 0
        | otherwise = 1
  let expSmth = randAct * expSmthPsi
  let psiValRho' = (1 - expSmth) * psiValRho + expSmth * abs psiRho
  let psiValV' = (1 - expSmth) * psiValV + expSmth * abs psiV
  let psiValW' = (1 - expSmth) * psiValW + expSmth * abs psiW

  let psiVState' = (1 - expSmthPsi) * psiVState + expSmthPsi * psiV

  -- enforce values
  let maxDeviation = sortBy (compare `on` abs) [psiW,psiV,psiW+psiV,-psiV,-psiW,-psiV-psiW]
  let vValStateNew = vValState' -- + xiVal * psiW
                     + clip (max 0.2 (0.1*abs vValState')) (xiVal * head maxDeviation)
                     -- + clip (max 0.2 (0.1*abs wValState')) (xiVal * psiV)
        --  | psiValV' < zetaVal = vValState' + clip vValState' (xiVal * psiW  + xiVal * psiV)
        --  | otherwise = vValState' + xiVal * psiV  + clip vValState' (xiVal * psiV)

      wValStateNew = wValState' -- clip (max 0.1 (0.1*abs wValState')) (xiVal * psiV)

      -- xiVal * psiVState' -- signum psiV * psiValV'
        --  | otherwise = wValState' -- - (1-xiVal) * psiV


        -- (xiVal * psiV + 2*xiVal * psiW)
  --   | abs psiV > abs psiW = vValState' - if randomAction then 0 else xiVal * psiV + xiVal * psiV
  --   | otherwise = vValState' - if randomAction then 0 else xiVal * psiW + xiVal * psiV


        -- xiVal * psiV

      clip minmax val = max (-minmax) $ min minmax val

  when (period == 0) $ liftSimple $ writeFile "psiValues" "Period\tPsiV\tPsiW\tZeta\t-Zeta\n"
  liftSimple $ appendFile "psiValues" (show period ++ "\t" ++ show (if randomAction then 0 else psiV) ++ "\t" ++ show (if randomAction then 0 else psiW) ++ "\t"  ++ show zetaVal ++ "\t" ++ show (-zetaVal) ++ "\n")


  -- let borl' | randomAction && borl ^. parameters.exploration <= borl ^. parameters.learnRandomAbove = borl -- multichain ?
  --           | otherwise = set v (M.insert state vValStateNew mv) $ set w (M.insert state wValState' mw) $ set rho rhoNew $
  --                         set r0 (M.insert state r0ValState' mr0) $ set r1 (M.insert state r1ValState' mr1) borl


  return $
    Calculation
      (Just rhoMinimumVal')
      (Just rhoVal')
      (Just psiVState')
      (Just vValStateNew)
      (Just wValStateNew)
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
    _        -> maximum <$> mapM (rhoValueWith Target borl state) actIdxes

vValue :: (MonadBorl' m) => Bool -> BORL s -> State s -> ActionIndex -> m Double
vValue addPsiV borl s a = vValueWith Worker addPsiV borl (ftExt s) a
  where
    ftExt = borl ^. featureExtractor

vValueFeat :: (MonadBorl' m) => Bool -> BORL s -> StateFeatures -> ActionIndex -> m Double
vValueFeat = vValueWith Worker

vValueWith :: (MonadBorl' m) => LookupType -> Bool -> BORL s -> StateFeatures -> ActionIndex -> m Double
vValueWith lkTp addPsiV borl state a = do
  vVal <- P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies . v)
  psiV <-
    if addPsiV
      then P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies . psiV)
      else return 0
  return (vVal + psiV)

vStateValue :: (MonadBorl' m, Ord s) => Bool -> BORL s -> (StateFeatures, [ActionIndex]) -> m Double
vStateValue addPsiV borl (state, asIdxes) = maximum <$> mapM (vValueWith Target addPsiV borl state) asIdxes


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
rValue borl size s = rValueWith Worker borl size (ftExt s)
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
