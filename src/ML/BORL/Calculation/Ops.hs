{-# LANGUAGE CPP                 #-}
{-# LANGUAGE ScopedTypeVariables #-}
module ML.BORL.Calculation.Ops
    ( mkCalculation
    , rValue
    , eValue
    , vValue
    , vValueFeat
    , rValueFeat
    , rhoValue
    , RSize (..)
    ) where

import           ML.BORL.Algorithm
import           ML.BORL.Calculation.Type
import           ML.BORL.NeuralNetwork.NNConfig
import           ML.BORL.Parameters
import           ML.BORL.Properties
import           ML.BORL.Proxy                  as P
import           ML.BORL.Reward
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Arrow                  (first)
import           Control.Lens
import           Control.Monad                  (when)
import           Control.Parallel.Strategies    hiding (r0)
import           Data.Function                  (on)
import           Data.List                      (maximumBy, minimumBy, sortBy)
import           System.Random                  (randomRIO)

import           Debug.Trace


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


mkCalculation :: (MonadBorl' m, Ord s) => BORL s -> (StateFeatures, [ActionIndex]) -> ActionIndex -> Bool -> RewardValue -> (StateNextFeatures, [ActionIndex]) -> EpisodeEnd -> m Calculation
mkCalculation borl state aNr randomAction reward stateNext episodeEnd =
  mkCalculation' borl state aNr randomAction reward stateNext episodeEnd (borl ^. algorithm)

ite :: Bool -> p -> p -> p
ite b t e
  | b = t
  | otherwise = e


mkCalculation' :: (MonadBorl' m, Ord s) => BORL s -> (StateFeatures, [ActionIndex]) -> ActionIndex -> Bool -> RewardValue -> (StateNextFeatures, [ActionIndex]) -> EpisodeEnd -> Algorithm s -> m Calculation
mkCalculation' borl (state, stateActIdxes) aNr randomAction reward (stateNext, stateNextActIdxes) episodeEnd (AlgBORL ga0 ga1 avgRewardType stValHandling decideOnVPlusPsiV mRefState) = do
  let params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
  let isANN p p2 = P.isNeuralNetwork (borl ^. proxies . p) && borl ^. t > borl ^?! proxies . p2 . proxyNNConfig . replayMemoryMaxSize
  let alp = params' ^. alpha
      alpANN = params' ^. alphaANN
      bta = params' ^. beta
      btaANN = params' ^. betaANN
      dlt = params' ^. delta
      dltANN = params' ^. deltaANN
      gamR0 = ite (isANN r0 r0) 1 (params' ^. gamma)
      gamR1 = ite (isANN r1 r1) 1 (params' ^. gamma)
      xiVal = params' ^. xi
      zetaVal = params' ^. zeta
      period = borl ^. t
      (psiValRho, psiValV, psiValW) = borl ^. psis -- exponentially smoothed Psis
  let label = (state, aNr)
      epsEnd
        | episodeEnd = 0
        | otherwise = 1
      randAct
        | randomAction = 0
        | otherwise = 1
  let learnFromRandom = params' ^. exploration > params' ^. learnRandomAbove
  let expSmth
        | learnFromRandom = expSmthPsi
        | otherwise = randAct * expSmthPsi
  let lastRews' =
        case avgRewardType of
          ByMovAvg movAvgLen -> take movAvgLen $ reward : borl ^. lastRewards
          _                  -> take keepXLastValues $ reward : borl ^. lastRewards
  vValState <- vValueFeat False borl state aNr `using` rpar
  rhoMinimumState <- rhoMinimumValueFeat borl state aNr `using` rpar
  vValStateNext <- vStateValue False borl (stateNext, stateNextActIdxes) `using` rpar
  rhoVal <- rhoValueFeat borl state aNr `using` rpar
  wValState <- wValueFeat borl state aNr `using` rpar
  wValStateNext <- wStateValue borl (stateNext, stateNextActIdxes) `using` rpar
  w2ValState <- w2ValueFeat borl state aNr `using` rpar
  w2ValStateNext <- w2StateValue borl (stateNext, stateNextActIdxes) `using` rpar
  r0ValState <- rValueFeat borl RSmall state aNr `using` rpar
  r1ValState <- rValueFeat borl RBig state aNr `using` rpar
  psiVState <- P.lookupProxy period Worker label (borl ^. proxies . psiV) `using` rpar
  psiWState <- P.lookupProxy period Worker label (borl ^. proxies . psiW) `using` rpar
  psiW2State <- P.lookupProxy period Worker label (borl ^. proxies . psiW2) `using` rpar
  -- Rho
  let rhoState =
        case avgRewardType of
          Fixed x -> x
          ByMovAvg l
            | isUnichain borl -> sum lastRews' / fromIntegral l
          ByMovAvg _ -> error "ByMovAvg is not allowed in multichain setups"
          ByReward -> reward
          ByStateValues -> reward + vValStateNext - vValState
          ByStateValuesAndReward -> 0.5 * (reward + vValStateNext - vValState) + 0.5 * reward
  let rhoVal' alphaVal =
        max rhoMinimumState $
        case avgRewardType of
          ByMovAvg _ -> rhoState
          Fixed x    -> x
          _          -> (1 - alphaVal) * rhoVal + alphaVal * rhoState
  -- RhoMin
  let rhoMinimumVal'
        | rhoState < rhoMinimumState = rhoMinimumState
        | otherwise = (1 - expSmthPsi / 200) * rhoMinimumState + expSmthPsi / 200 * rhoVal' alp -- rhoState
  -- PsiRho (should converge to 0)
  psiRho <- ite (isUnichain borl) (return $ rhoVal' alp - rhoVal) (subtract (rhoVal' alp) <$> rhoStateValue borl (stateNext, stateNextActIdxes))
  -- V
  let vValState' betaVal = (1 - betaVal) * vValState + betaVal * (reward - rhoVal' alp + epsEnd * vValStateNext)
      psiV = reward + vValStateNext - rhoVal' alp - vValState' bta -- should converge to 0
      psiVState' = (1 - bta) * psiVState + bta * psiV
  -- LastVs
  let lastVs' =
        case stValHandling of
          Normal -> take keepXLastValues $ vValState' bta : borl ^. lastVValues
          DivideValuesAfterGrowth nr _ -> take nr $ vValState' bta : borl ^. lastVValues
  -- W
  let wValState' deltaVal = (1 - deltaVal) * wValState + deltaVal * (-vValState' bta + epsEnd * wValStateNext)
      psiW = wValStateNext - vValState' bta - wValState' dlt
      psiWState' = (1 - dlt) * psiWState + dlt * psiW
  -- W2
  let w2ValState' deltaVal = (1 - deltaVal) * w2ValState + deltaVal * (-wValState' bta + epsEnd * w2ValStateNext)
      psiW2 = w2ValStateNext - wValState' dlt - w2ValState' dlt
      psiW2State' = (1 - 0.5 * dlt) * psiW2State + 0.5 * dlt * psiW2
   -- R0/R1
  rSmall <- rStateValue borl RSmall (stateNext, stateNextActIdxes)
  rBig <- rStateValue borl RBig (stateNext, stateNextActIdxes)
  let r0ValState' = (1 - gamR0) * r0ValState + gamR0 * (reward + epsEnd * ga0 * rSmall)
  let r1ValState' = (1 - gamR1) * r1ValState + gamR1 * (reward + epsEnd * ga1 * rBig)
  -- Psis Scalar calues for output only
  let psiValRho' = (1 - expSmth) * psiValRho + expSmth * abs psiRho
  let psiValV' = (1 - expSmth) * psiValV + expSmth * abs psiVState'
  let psiValW' = (1 - expSmth) * psiValW + expSmth * abs psiWState'
  -- enforce values
  let loss x = 0.5 * signum x * x ^ 2
  let wValStateNew betaVal
        | randomAction && not learnFromRandom = wValState' betaVal
        | otherwise = wValState' betaVal -- + xiVal * loss psiW2State'
        -- | randomAction && not learnFromRandom = wValState' betaVal
        -- | abs psiVState' > params' ^. epsilon && period `mod` 2 == 0 =
        --    (1-xiVal) * wValState' dlt + xiVal * (wValState' dlt + loss psiWState')
        -- | otherwise =
        --    (1-xiVal) * wValState' dlt + xiVal * (wValState' dlt + loss psiW2State') -- original !!!
        -- let vValStateNew betaVal
        -- | randomAction && not learnFromRandom = vValState' betaVal
        -- | abs psiVState' > params' ^. epsilon && period `mod` 2 == 0 =
        --    (1-xiVal) * vValState' betaVal + xiVal * (vValState' betaVal + loss psiVState')
        -- | otherwise =
        --    (1-xiVal) * vValState' betaVal + xiVal * (vValState' betaVal + loss psiWState') -- original !!!
  let vValStateNew betaVal
        | randomAction && not learnFromRandom = vValState' betaVal
        | otherwise = vValState' betaVal + xiVal * err
        where
          err = psiVState' + 0.03 * psiWState' - 0.01 * psiW2State'
  -- let vValStateNew betaVal
  --       | randomAction && not learnFromRandom = vValState' betaVal
  --       -- | abs psiVState' > params' ^. epsilon && period `mod` 2 == 0 = vValState' betaVal + xiVal * psiVState'
  --       | otherwise = vValState' betaVal + xiVal * loss err
  --         -- psiWState'
  --       -- | randomAction && not learnFromRandom = vValState' betaVal
  --       -- | otherwise =
  --       --    (1 - xiVal) * vValState' betaVal + xiVal * (vValState' betaVal + err)
  --       where
  --       --   err = signum psiWState' * psiWState'^2
  --         err = psiVState' + 0.03 * psiWState' -- 0.01 * psiW2State'
  when (period == 0) $ liftSimple $ writeFile "psiValues" "Period\tPsiV_ExpSmth\tPsiW_ExpSmth\tZeta\t-Zeta\n"
  liftSimple $
    appendFile
      "psiValues"
      (show period ++ "\t" ++ show (ite randomAction 0 psiVState') ++ "\t" ++ show (ite randomAction 0 psiWState') ++ "\t" ++ show zetaVal ++ "\t" ++ show (-zetaVal) ++ "\n")
  return $
    Calculation
      { getRhoMinimumVal' = Just rhoMinimumVal'
      , getRhoVal' = Just $ rhoVal' (ite (isANN rho rho) alpANN alp)
      , getPsiVValState' = Just psiVState'
      , getVValState' = Just $ vValStateNew (ite (isANN v v) btaANN bta)
      , getPsiWValState' -- Just $ ite ((first (borl ^. featureExtractor) <$> mRefState) == Just (state, aNr)) 0 (psiWState')
         = Just psiWState'
      , getWValState' -- Just $ ite ((first (borl ^. featureExtractor) <$> mRefState) == Just (state, aNr)) 0 (wValStateNew (ite (isANN w w) dltANN dlt))
         = Just (wValStateNew (ite (isANN w w) dltANN dlt))
      , getPsiW2ValState' = Just psiW2State'
      , getW2ValState' = Just $ ite ((first (borl ^. featureExtractor) <$> mRefState) == Just (state, aNr)) 0 (w2ValState' (ite (isANN w2 w2) (0.5 * dltANN) (0.5 * dlt)))
      , getR0ValState' = Just r0ValState'
      , getR1ValState' = Just r1ValState'
      , getPsiValRho' = Just psiValRho'
      , getPsiValV' = Just psiValV'
      , getPsiValW' = Just psiValW'
      , getLastVs' = Just lastVs'
      , getLastRews' = lastRews'
      , getEpisodeEnd = episodeEnd
      }

mkCalculation' borl (state, stateActIdxes) aNr randomAction reward (stateNext, stateNextActIdxes) episodeEnd (AlgBORLVOnly avgRewardType mRefState) = do
  let params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
  let isANN p p2 = P.isNeuralNetwork (borl ^. proxies . p) && borl ^. t > borl ^?! proxies . p2 . proxyNNConfig . replayMemoryMaxSize
      alp = params' ^. alpha
      alpANN = params' ^. alphaANN
      bta = params' ^. beta
      btaANN = params' ^. betaANN
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
             ByMovAvg _ -> return $ sum lastRews' / fromIntegral (length lastRews')
             ByReward -> return reward
             ByStateValues -> error "Average reward using `ByStateValues` not supported for AlgBORLVOnly"
             ByStateValuesAndReward -> error "Average reward using `ByStateValuesAndReward` not supported for AlgBORLVOnly"
      else do
        rhoStateValNext <- rhoStateValue borl (stateNext, stateNextActIdxes)
        return $ (epsEnd * approxAvg * rhoStateValNext + reward) / (epsEnd * approxAvg + 1) -- approximation
  let rhoVal' alpVal =
        max rhoMinimumState $
        case avgRewardType of
          ByMovAvg _ -> rhoState
          Fixed x    -> x
          _          -> (1 - alpVal) * rhoVal + alpVal * rhoState
  let rhoMinimumVal'
        | rhoState < rhoMinimumState = rhoMinimumState
        | otherwise = (1 - expSmthPsi / 200) * rhoMinimumState + expSmthPsi / 200 * rhoState
  vValState <- vValueFeat False borl state aNr `using` rpar
  vValStateNext <- vStateValue False borl (stateNext, stateNextActIdxes) `using` rpar
  let vValState' btaVal = (1 - btaVal) * vValState + btaVal * (reward - rhoVal' alp + epsEnd * vValStateNext)
  let lastVs' = take keepXLastValues $ vValState' bta : borl ^. lastVValues
  return $
    Calculation
      { getRhoMinimumVal' = Just rhoMinimumVal'
      , getRhoVal' = Just $ rhoVal' (ite (isANN rho rho) alpANN alp)
      , getPsiVValState' = Nothing
      , getVValState' = Just $ ite ((first (borl ^. featureExtractor) <$> mRefState) == Just (state, aNr)) 0 (vValState' (ite (isANN v v) btaANN bta))
      , getPsiWValState' = Nothing
      , getWValState' = Nothing
      , getPsiW2ValState' = Nothing
      , getW2ValState' = Nothing
      , getR0ValState' = Nothing
      , getR1ValState' = Nothing
      , getPsiValRho' = Nothing
      , getPsiValV' = Nothing
      , getPsiValW' = Nothing
      , getLastVs' = Just lastVs'
      , getLastRews' = lastRews'
      , getEpisodeEnd = episodeEnd
      }

mkCalculation' borl (state, _) aNr randomAction reward (stateNext, stateNextActIdxes) episodeEnd (AlgDQN ga) = do
  let params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
  let isANN = P.isNeuralNetwork (borl ^. proxies . r1) && borl ^. t > borl ^?! proxies . r1 . proxyNNConfig . replayMemoryMaxSize
      gam
        | isANN = params' ^. gammaANN
        | otherwise = params' ^. gamma
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
      , getPsiW2ValState' = Nothing
      , getW2ValState' = Nothing
      , getR0ValState' = Nothing
      , getR1ValState' = Just r1ValState'
      , getPsiValRho' = Nothing
      , getPsiValV' = Nothing
      , getPsiValW' = Nothing
      , getLastVs' = Nothing
      , getLastRews' = lastRews'
      , getEpisodeEnd = episodeEnd
      }
mkCalculation' borl (state, _) aNr randomAction reward (stateNext, stateNextActIdxes) episodeEnd (AlgDQNAvgRewardFree ga0 ga1 avgRewardType)
  -- let decay = 0.5 ** (fromIntegral (borl ^. t) / 100000)
  --     ga1Diff = 1 - ga1
  --     ga1' = ga1 + ga1Diff - (ga1Diff * decay)
 = do
  rhoMinimumState <- rhoMinimumValueFeat borl state aNr `using` rpar
  rhoVal <- rhoValueFeat borl state aNr `using` rpar
  r0ValState <- rValueFeat borl RSmall state aNr `using` rpar
  r0StateNext <- rStateValue borl RSmall (stateNext, stateNextActIdxes) `using` rpar
  r1ValState <- rValueFeat borl RBig state aNr `using` rpar
  r1StateNext <- rStateValue borl RBig (stateNext, stateNextActIdxes) `using` rpar
  let params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
  let isANN = P.isNeuralNetwork (borl ^. proxies . r1) && borl ^. t > borl ^?! proxies . r1 . proxyNNConfig . replayMemoryMaxSize
      gam
        | isANN = params' ^. gammaANN
        | otherwise = params' ^. gamma
      alp
        | isANN = params' ^. alphaANN
        | otherwise = params' ^. alpha
  let epsEnd
        | episodeEnd = 0
        | otherwise = 1
  let lastRews' =
        case avgRewardType of
          ByMovAvg movAvgLen -> take movAvgLen $ reward : borl ^. lastRewards
          _                  -> take keepXLastValues $ reward : borl ^. lastRewards
  -- Rho
  rhoState <-
    if isUnichain borl
      then case avgRewardType of
             Fixed x       -> return x
             ByMovAvg l    -> return $ sum lastRews' / fromIntegral l
             ByReward      -> return reward
             ByStateValues -> return $ reward + r1StateNext - r1ValState
             ByStateValuesAndReward -> return $ 0.5 * (reward + r1StateNext - r1ValState) + 0.5 * reward
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
        | otherwise = (1 - expSmthPsi / 200) * rhoMinimumState + expSmthPsi / 200 * rhoVal'
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
      , getPsiW2ValState' = Nothing
      , getW2ValState' = Nothing
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

vStateValue :: (MonadBorl' m) => Bool -> BORL s -> (StateFeatures, [ActionIndex]) -> m Double
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


w2Value :: (MonadBorl' m) => BORL s -> State s -> ActionIndex -> m Double
w2Value borl state a = w2ValueWith Worker borl (ftExt state) a
  where
    ftExt = borl ^. featureExtractor

w2ValueFeat :: (MonadBorl' m) => BORL s -> StateFeatures -> ActionIndex -> m Double
w2ValueFeat = w2ValueWith Worker

w2ValueWith :: (MonadBorl' m) => LookupType -> BORL s -> StateFeatures -> ActionIndex -> m Double
w2ValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies . w2)

w2StateValue :: (MonadBorl' m) => BORL s -> (StateFeatures, [ActionIndex]) -> m Double
w2StateValue borl (state, asIdxes) =
  minimum <$> mapM (w2ValueWith Target borl state) asIdxes
  -- maximum <$> mapM (w2ValueWith Target borl state) asIdxes


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
