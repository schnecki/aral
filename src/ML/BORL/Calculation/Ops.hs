{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ViewPatterns        #-}
module ML.BORL.Calculation.Ops
    ( mkCalculation
    , actionsIndexed
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

import           ML.BORL.Action
import           ML.BORL.Algorithm
import           ML.BORL.Calculation.Type
import           ML.BORL.Fork
import           ML.BORL.NeuralNetwork.Tensorflow (buildTensorflowModel,
                                                   restoreModelWithLastIO,
                                                   saveModelWithLastIO)
import           ML.BORL.Parameters
import           ML.BORL.Properties
import           ML.BORL.Proxy                    as P
import           ML.BORL.SaveRestore
import           ML.BORL.Serialisable
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Applicative              ((<|>))
import           Control.DeepSeq                  (NFData, force)
import           Control.Lens
import           Control.Monad
import           Control.Monad.IO.Class           (MonadIO, liftIO)
import           Control.Parallel.Strategies      hiding (r0)
import           Data.Function                    (on)
import           Data.List                        (find, groupBy, sortBy)
import qualified Data.Map.Strict                  as M
import           Data.Maybe                       (fromMaybe, isJust)
import           System.Directory
import           System.IO
import           System.Random

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


mkCalculation :: (MonadBorl' m, Ord s) => BORL s -> State s -> ActionIndex -> Bool -> Reward -> StateNext s -> EpisodeEnd -> m Calculation
mkCalculation borl state aNr randomAction reward stateNext episodeEnd = do
  let params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
  let alp = params' ^. alpha
      bta = params' ^. beta
      dlt = params' ^. delta
      gam = params' ^. gamma
      alg = borl ^. algorithm
      xiVal = params' ^. xi
      period = borl ^. t
      (psiValRho, psiValV, psiValW) = borl ^. psis -- exponentially smoothed Psis
  let label = (state, aNr)
      epsEnd
        | episodeEnd = 0
        | otherwise = 1
  case borl ^. algorithm of
    AlgBORL ga0 ga1 avgRewardType stValHandling decideOnVPlusPsiV -> do
      let lastRews' =
            case avgRewardType of
              ByMovAvg movAvgLen -> take movAvgLen $ reward : borl ^. lastRewards
              _                  -> take keepXLastValues $ reward : borl ^. lastRewards
      rhoMinimumState <- rhoMinimumValue borl state aNr `using` rpar
      vValState <- vValue False borl state aNr `using` rpar
      vValStateNext <- vStateValue decideOnVPlusPsiV borl stateNext `using` rpar
      rhoVal <- rhoValue borl state aNr `using` rpar
      wValState <- wValue borl state aNr `using` rpar
      wValStateNext <- wStateValue borl state `using` rpar
      r0ValState <- rValue borl RSmall state aNr `using` rpar
      r1ValState <- rValue borl RBig state aNr `using` rpar
      psiVTblVal <- P.lookupProxy period Worker label (borl ^. proxies . psiV) `using` rpar
      rhoState <-
        if isUnichain borl
          then case avgRewardType of
                 Fixed x       -> return x
                 ByMovAvg l    -> return $ sum lastRews' / fromIntegral l -- (length lastRews')
                 ByReward      -> return reward
                 ByStateValues -> return (reward + vValStateNext - vValState)
          else do
            rhoStateValNext <- rhoStateValue borl stateNext
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
      rSmall <- rStateValue borl RSmall stateNext
      rBig <- rStateValue borl RBig stateNext
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
    AlgDQN ga -> do
      let lastRews' = take keepXLastValues $ reward : borl ^. lastRewards
      r1ValState <- rValue borl RBig state aNr `using` rpar
      rBig <- rStateValue borl RBig stateNext `using` rpar
      let r1ValState' = (1 - gam) * r1ValState + gam * (reward + epsEnd * ga * rBig)
      return $ Calculation Nothing Nothing Nothing Nothing Nothing Nothing r1ValState' Nothing Nothing Nothing Nothing lastRews' episodeEnd
    AlgDQNAvgRew ga avgRewardType -> do
      rhoVal <- rhoValue borl state aNr `using` rpar
      let lastRews' =
            case avgRewardType of
              ByMovAvg movAvgLen -> take movAvgLen $ reward : borl ^. lastRewards
              _                  -> take keepXLastValues $ reward : borl ^. lastRewards
      rhoMinimumState <- rhoMinimumValue borl state aNr `using` rpar
      rhoState <-
        if isUnichain borl
          then case avgRewardType of
                 Fixed x -> return x
                 ByMovAvg l -> return $ sum lastRews' / fromIntegral l -- (length lastRews')
                 ByReward -> return reward
                 ByStateValues -> error "Average reward using `ByStateValues` not supported for AlgDQNAvgRew"
          else do
            rhoStateValNext <- rhoStateValue borl stateNext
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
      r1ValState <- rValue borl RBig state aNr `using` rpar
      rBig <- rStateValue borl RBig stateNext `using` rpar
      let r1ValState' = (1 - gam) * r1ValState + gam * (reward - rhoVal' + epsEnd * ga * rBig)
      return $ Calculation (Just rhoMinimumVal') (Just rhoVal') Nothing Nothing Nothing Nothing r1ValState' Nothing Nothing Nothing Nothing lastRews' episodeEnd

-- TODO maybe integrate learnRandomAbove, etc.:
  -- let borl'
  --       | randomAction && params' ^. exploration <= borl ^. parameters . learnRandomAbove = borl -- multichain ?
  --       | otherwise = set v mv' $ set w mw' $ set rho rhoNew $ set r0 mr0' $ set r1 mr1' borl


actionsIndexed :: BORL s -> s -> [ActionIndexed s]
actionsIndexed borl state = map snd $ filter fst $ zip ((borl ^. actionFilter) state) (borl ^. actionList)


-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
rhoMinimumValue :: (MonadBorl' m, Ord s) => BORL s -> s -> ActionIndex -> m Double
rhoMinimumValue = rhoMinimumValueWith Worker

rhoMinimumValueWith :: (MonadBorl' m, Ord s) => LookupType -> BORL s -> s -> ActionIndex -> m Double
rhoMinimumValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state,a) (borl ^. proxies.rhoMinimum)


-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
rhoValue :: (MonadBorl' m, Ord s) => BORL s -> s -> ActionIndex -> m Double
rhoValue = rhoValueWith Worker

rhoValueWith :: (MonadBorl' m, Ord s) => LookupType -> BORL s -> s -> ActionIndex -> m Double
rhoValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state,a) (borl ^. proxies.rho)


rhoStateValue :: (MonadBorl' m, Ord s) => BORL s -> s -> m Double
rhoStateValue borl state = case borl ^. proxies.rho of
  Scalar r  -> return r
  _ -> maximum <$> mapM (rhoValueWith Target borl state) (map fst $ actionsIndexed borl state)

vValue :: (MonadBorl' m, Ord s) => Bool -> BORL s -> s -> ActionIndex -> m Double
vValue = vValueWith Worker

vValueWith :: (MonadBorl' m, Ord s) => LookupType -> Bool -> BORL s -> s -> ActionIndex -> m Double
vValueWith lkTp addPsiV borl state a = do
  vVal <- P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies . v)
  psiV <-
    if addPsiV
      then P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies . psiV)
      else return 0
  return (vVal + psiV)

vStateValue :: (MonadBorl' m, Ord s) => Bool -> BORL s -> s -> m Double
vStateValue addPsiV borl state = maximum <$> mapM (vValueWith Target addPsiV borl state) (map fst $ actionsIndexed borl state)


wValue :: (MonadBorl' m, Ord s) => BORL s -> s -> ActionIndex -> m Double
wValue = wValueWith Worker

wValueWith :: (MonadBorl' m, Ord s) => LookupType -> BORL s -> s -> ActionIndex -> m Double
wValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies.w)

wStateValue :: (MonadBorl' m, Ord s) => BORL s -> s -> m Double
wStateValue borl state = maximum <$> mapM (wValueWith Target borl state) (map fst $ actionsIndexed borl state)


-- | Calculates the expected discounted value with the provided gamma (small/big).
rValue :: (MonadBorl' m, Ord s) => BORL s -> RSize -> s -> ActionIndex -> m Double
rValue = rValueWith Worker


-- | Calculates the expected discounted value with the provided gamma (small/big).
rValueWith :: (MonadBorl' m, Ord s) => LookupType -> BORL s -> RSize -> s -> ActionIndex -> m Double
rValueWith lkTp borl size state a = P.lookupProxy (borl ^. t) lkTp (state, a) mr
  where
    mr =
      case size of
        RSmall -> borl ^. proxies.r0
        RBig   -> borl ^. proxies.r1

rStateValue :: (MonadBorl' m, Ord s) => BORL s -> RSize -> s -> m Double
rStateValue borl size state = maximum <$> mapM (rValueWith Target borl size state . fst) (actionsIndexed borl state)

-- | Calculates the difference between the expected discounted values.
eValue :: (MonadBorl' m, Ord s) => BORL s -> s -> ActionIndex -> m Double
eValue borl state act = do
  big <- rValueWith Target borl RBig state act
  small <- rValueWith Target borl RSmall state act
  return $ big - small

--  | Calculates the difference between the expected discounted values.
-- eStateValue :: (Ord s) => BORL s -> s -> Double
-- eStateValue borl state = maximum (map (rValueWith Target borl RBig state) as) - reduce (map (rValueWith Target borl RSmall state) as)
--   where as = actionsIndexed borl state
