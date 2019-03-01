{-# LANGUAGE ScopedTypeVariables #-}
module ML.BORL.Step
    ( step
    , steps
    , restoreTensorflowModels
    , saveTensorflowModels
    , stepExecute
    , nextAction
    ) where

import           ML.BORL.Action
import           ML.BORL.Calculation
import           ML.BORL.Fork
import           ML.BORL.NeuralNetwork.Tensorflow (buildTensorflowModel,
                                                   restoreModelWithLastIO,
                                                   saveModelWithLastIO)
import           ML.BORL.Parameters
import           ML.BORL.Properties
import           ML.BORL.Proxy                    as P
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Applicative              ((<|>))
import           Control.DeepSeq                  (NFData, force)
import           Control.Lens
import           Control.Monad
import           Control.Monad.IO.Class           (MonadIO, liftIO)
import           Data.Function                    (on)
import           Data.List                        (find, groupBy, sortBy)
import qualified Data.Map.Strict                  as M
import           Data.Maybe                       (isJust)
import           System.Directory
import           System.IO
import           System.Random

expSmthPsi :: Double
expSmthPsi = 0.03

keepXLastValues :: Int
keepXLastValues = 1000

approxAvg :: Double
approxAvg = fromIntegral (100 :: Int)


-- TF.asyncProdNodes TODO

steps :: (NFData s, Ord s) => BORL s -> Integer -> IO (BORL s)
steps borl nr = runMonadBorl $ do
  restoreTensorflowModels borl
  borl' <- foldM (\b _ -> nextAction b >>= stepExecute) borl [0 .. nr - 1]
  force <$> saveTensorflowModels borl'


step :: (NFData s, Ord s) => BORL s -> IO (BORL s)
step borl = runMonadBorl $ do
  restoreTensorflowModels borl
  borl' <- nextAction borl >>= stepExecute
  force <$> saveTensorflowModels borl'

restoreTensorflowModels :: BORL s -> MonadBorl ()
restoreTensorflowModels borl = do
  buildModels
  mapM_ restoreProxy (allProxies $ borl ^. proxies)
  where
    restoreProxy px =
      case px of
        TensorflowProxy netT netW _ _ _ _ -> restoreModelWithLastIO netT >> restoreModelWithLastIO netW >> return ()
        _ -> return ()
    isTensorflowProxy TensorflowProxy {} = True
    isTensorflowProxy _                  = False
    buildModels =
      case find isTensorflowProxy (allProxies $ borl ^. proxies) of
        Just (TensorflowProxy netT _ _ _ _ _) -> buildTensorflowModel netT
        _                                     -> return ()


saveTensorflowModels :: BORL s -> MonadBorl (BORL s)
saveTensorflowModels borl = do
  mapM_ saveProxy (allProxies $ borl ^. proxies)
  return borl
  where
    saveProxy px =
      case px of
        TensorflowProxy netT netW _ _ _ _ -> saveModelWithLastIO netT >> saveModelWithLastIO netW >> return ()
        _ -> return ()


mkCalculation :: (Ord s) => BORL s -> s -> ActionIndex -> Bool -> Reward -> s -> MonadBorl Calculation
mkCalculation borl state aNr randomAction reward stateNext = do
  let alp = borl ^. parameters . alpha
      bta = borl ^. parameters . beta
      dlt = borl ^. parameters . delta
      gam = borl ^. parameters . gamma
      (ga0, ga1) = borl ^. gammas
      period = borl ^. t
      (psiValRho, psiValV, psiValW) = borl ^. psis -- exponentially smoothed Psis
  let lastRews' = take keepXLastValues $ reward : borl ^. lastRewards
      avgRew = sum lastRews' / fromIntegral (length lastRews')
  let label = (state, aNr)
  rhoMinimumState <- rhoMinimumValue borl state aNr
  vValState <- vValue False borl state aNr
  vValStateNext <- vStateValue False borl stateNext
  rhoVal <- rhoValue borl state aNr
  wValState <- wValue borl state aNr
  wValStateNext <- wStateValue borl state
  r0ValState <- rValue borl RSmall state aNr
  r1ValState <- rValue borl RBig state aNr
  psiVTblVal <- P.lookupProxy period Worker label (borl ^. proxies . psiV)
  psiWTblVal <- P.lookupProxy period Worker label (borl ^. proxies . psiW)
  rhoState <-
    if isUnichain borl
           then -- return (reward + vValStateNext - vValState)
                -- return reward                                              -- Alternative to above (estimating it from actual reward)
                return avgRew
      else do
        rhoStateValNext <- rhoStateValue borl stateNext
        return $ (approxAvg * rhoStateValNext + reward) / (approxAvg + 1) -- approximation
  let rhoVal' = max rhoMinimumState rhoState
                -- ((1 - alp) * rhoVal + alp * rhoState)
  let rhoMinimumVal' | rhoState < rhoMinimumState = rhoMinimumState
                     | otherwise = (1 - expSmthPsi / 50) * rhoMinimumState + expSmthPsi / 50 * rhoState
  let psiRho = rhoVal' - rhoVal -- should converge to 0
  let vValState' = (1 - bta) * vValState + bta * (reward - rhoVal' + vValStateNext)
      psiV = reward + vValStateNext - rhoVal' - vValState' -- should converge towards 0
      lastVs' = take keepXLastValues $ vValState' : borl ^. lastVValues
  let wValState' = (1 - dlt) * wValState + dlt * (-vValState' + wValStateNext)
  let psiW = wValStateNext - vValState' - wValState' -- should converge towards 0
  let psiValRho' = (1 - expSmthPsi) * psiValRho + expSmthPsi * (if randomAction then 0 else abs psiRho)
      psiValV' = (1 - expSmthPsi) * psiValV + expSmthPsi * (if randomAction then 0 else abs psiV)
      psiValW' = (1 - expSmthPsi) * psiValW + expSmthPsi * (if randomAction then 0 else abs psiW)
  let psiVTblVal' = (1 - expSmthPsi) * psiVTblVal + expSmthPsi * psiV
  let psiWTblVal' = (1 - expSmthPsi) * psiWTblVal + expSmthPsi * psiW
  let xiVal = borl ^. parameters . xi
  let vValStateNew -- enforce bias optimality (correction of V(s,a) values)
        --  | randomAction && (psiV > eps || (psiV <= eps && psiV > -eps && psiW > eps)) = vValState' -- interesting action
        --  | randomAction = vValState' -- psiW and psiV should not be 0!
        | abs psiV < abs psiW = (1 - xiVal) * vValState' + xiVal * (vValState' + clip (abs vValState') psiW)
        | otherwise = (1 - xiVal) * vValState' + xiVal * (vValState' + clip (abs vValState') psiV)
      clip minmax val = max (-minmax) $ min minmax val
  -- R0/R1
  rSmall <- rStateValue borl RSmall stateNext
  rBig <- rStateValue borl RBig stateNext
  let r0ValState' = (1 - gam) * r0ValState + gam * (reward + ga0 * rSmall)
  let r1ValState' = (1 - gam) * r1ValState + gam * (reward + ga1 * rBig)
  return $ Calculation rhoMinimumVal' rhoVal' psiVTblVal' psiWTblVal' vValStateNew wValState' r0ValState' r1ValState' psiValRho' psiValV' psiValW' lastVs' lastRews'

stepExecute :: forall s . (NFData s, Ord s) => (BORL s, Bool, ActionIndexed s) -> MonadBorl (BORL s)
stepExecute (borl, randomAction, (aNr, Action action _)) = do
  let state = borl ^. s
      period = borl ^. t
  (reward, stateNext) <- Simple $ action state
  (proxies', calc) <- P.insert period state aNr randomAction reward stateNext (mkCalculation borl) (borl ^. proxies)

  -- File IO Operations
  Simple $ doesFileExist "rhoValues" >>= \exists -> when (exists && period == 0) $ removeFile "rhoValues"
  Simple $ appendFile "rhoValues" (show period ++ "\t" ++ show (getRhoVal' calc) ++ "\t" ++ show (getRhoMinimumVal' calc) ++ "\t" ++ show (sum (getLastVs' calc) / fromIntegral (length (getLastVs' calc))) ++ "\n")

  -- -- forkMv' <- Simple $ doFork $ P.insert period label vValStateNew mv
  -- -- forkMw' <- Simple $ doFork $ runMonadBorl $ P.insert period label wValState' mw
  -- -- forkMr0' <- Simple $ doFork $ runMonadBorl $ P.insert period label r0ValState' mr0
  -- -- forkMr1' <- Simple $ doFork $ runMonadBorl $ P.insert period label r1ValState' mr1
  let params' = (borl ^. decayFunction) (period + 1) (borl ^. psis) (getPsiValRho' calc, getPsiValV' calc, getPsiValW' calc) (borl ^. parameters)
  -- -- mv' <- Simple $ collectForkResult forkMv'
  -- -- mw' <- Simple $ collectForkResult forkMw'
  -- -- mr0' <- Simple $ collectForkResult forkMr0'
  -- -- mr1' <- Simple $ collectForkResult forkMr1'

  -- let borl'
  --       | randomAction && borl ^. parameters . exploration <= borl ^. parameters . learnRandomAbove = borl -- multichain ?
  --       | otherwise = set v mv' $ set w mw' $ set rho rhoNew $ set r0 mr0' $ set r1 mr1' borl
  -- update values
  return $ force $ -- needed to ensure constant memory consumption
    set psis (getPsiValRho' calc, getPsiValV' calc, getPsiValW' calc) $
    set lastVValues (getLastVs' calc) $
    set lastRewards (getLastRews' calc) $
    set proxies proxies' $
    set visits (M.alter (\mV -> ((+ 1) <$> mV) <|> Just 1) state (borl ^. visits)) $
    set s stateNext $
    set t (period + 1) $
    set parameters params' borl


-- | This function chooses the next action from the current state s and all possible actions.
nextAction :: (Ord s) => BORL s -> MonadBorl (BORL s, Bool, ActionIndexed s)
nextAction borl
  | null as = error "Empty action list"
  | length as == 1 = return (borl, False, head as)
  | otherwise = do
    rand <- Simple $ randomRIO (0, 1)
    if rand < explore
      then do
        r <- Simple $ randomRIO (0, length as - 1)
        return (borl, True, as !! r)
      else do
        bestRho <-
          if isUnichain borl
            then return as
            else do
              rhoVals <- mapM (rhoValue borl state) (map fst as)
              return $ map snd $ head $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (zip rhoVals as)
        bestV <-
          do vVals <- mapM (vValue True borl state) (map fst bestRho)
             return $ map snd $ head $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (zip vVals bestRho)
        bestE <-
          do eVals <- mapM (eValue borl state) (map fst bestV)
             return $ map snd $ sortBy (epsCompare compare `on` fst) (zip eVals bestV)
    -- bestR <- sortBy (epsCompare compare `on` rValue borl RBig state) bestV
    -- return (False, head bestR)
        if length bestE > 1
          then do
            r <- Simple $ randomRIO (0, length bestE - 1)
            return (borl, False, bestE !! r)
          else return (borl, False, head bestE)
  where
    eps = borl ^. parameters . epsilon
    epsCompare f x y
      | abs (x - y) <= eps = f 0 0
      | otherwise = y `f` x
    explore = borl ^. parameters . exploration
    state = borl ^. s
    as = actionsIndexed borl state


actionsIndexed :: BORL s -> s -> [ActionIndexed s]
actionsIndexed borl state = map snd $ filter fst $ zip ((borl ^. actionFilter) state) (borl ^. actionList)


-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
rhoMinimumValue :: (Ord s) => BORL s -> s -> ActionIndex -> MonadBorl Double
rhoMinimumValue = rhoMinimumValueWith Worker

rhoMinimumValueWith :: (Ord s) => LookupType -> BORL s -> s -> ActionIndex -> MonadBorl Double
rhoMinimumValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state,a) (borl ^. proxies.rhoMinimum)


-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
rhoValue :: (Ord s) => BORL s -> s -> ActionIndex -> MonadBorl Double
rhoValue = rhoValueWith Worker

rhoValueWith :: (Ord s) => LookupType -> BORL s -> s -> ActionIndex -> MonadBorl Double
rhoValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state,a) (borl ^. proxies.rho)


rhoStateValue :: (Ord s) => BORL s -> s -> MonadBorl Double
rhoStateValue borl state = case borl ^. proxies.rho of
  Scalar r  -> return r
  _ -> maximum <$> mapM (rhoValueWith Target borl state) (map fst $ actionsIndexed borl state)

vValue :: (Ord s) => Bool -> BORL s -> s -> ActionIndex -> MonadBorl Double
vValue = vValueWith Worker

vValueWith :: (Ord s) => LookupType -> Bool -> BORL s -> s -> ActionIndex -> MonadBorl Double
vValueWith lkTp addPsiV borl state a = do
  vVal <- P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies.v)
  psiV <-
    if addPsiV
      then P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies.psiV)
      else return 0
  return (vVal + psiV)

vStateValue :: (Ord s) => Bool -> BORL s -> s -> MonadBorl Double
vStateValue addPsiV borl state = maximum <$> mapM (vValueWith Target addPsiV borl state) (map fst $ actionsIndexed borl state)


wValue :: (Ord s) => BORL s -> s -> ActionIndex -> MonadBorl Double
wValue = wValueWith Worker

wValueWith :: (Ord s) => LookupType -> BORL s -> s -> ActionIndex -> MonadBorl Double
wValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies.w)

wStateValue :: (Ord s) => BORL s -> s -> MonadBorl Double
wStateValue borl state = maximum <$> mapM (wValueWith Target borl state) (map fst $ actionsIndexed borl state)


-- | Used to select a discount factor.
data RSize
  = RSmall
  | RBig


-- | Calculates the expected discounted value with the provided gamma (small/big).
rValue :: (Ord s) => BORL s -> RSize -> s -> ActionIndex -> MonadBorl Double
rValue = rValueWith Worker

-- | Calculates the expected discounted value with the provided gamma (small/big).
rValueWith :: (Ord s) => LookupType -> BORL s -> RSize -> s -> ActionIndex -> MonadBorl Double
rValueWith lkTp borl size state a = P.lookupProxy (borl ^. t) lkTp (state, a) mr
  where
    mr =
      case size of
        RSmall -> borl ^. proxies.r0
        RBig   -> borl ^. proxies.r1

rStateValue :: (Ord s) => BORL s -> RSize -> s -> MonadBorl Double
rStateValue borl size state = maximum <$> mapM (rValueWith Target borl size state . fst) (actionsIndexed borl state)

-- | Calculates the difference between the expected discounted values.
eValue :: (Ord s) => BORL s -> s -> ActionIndex -> MonadBorl Double
eValue borl state act = do
  big <- rValueWith Target borl RBig state act
  small <- rValueWith Target borl RSmall state act
  return $ big - small

--  | Calculates the difference between the expected discounted values.
-- eStateValue :: (Ord s) => BORL s -> s -> Double
-- eStateValue borl state = maximum (map (rValueWith Target borl RBig state) as) - reduce (map (rValueWith Target borl RSmall state) as)
--   where as = actionsIndexed borl state


