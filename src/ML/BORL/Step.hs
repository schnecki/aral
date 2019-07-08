{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ViewPatterns        #-}
module ML.BORL.Step
    ( step
    , steps
    , stepM
    , stepsM
    , restoreTensorflowModels
    , saveTensorflowModels
    , stepExecute
    , nextAction
    , epsCompareWith
    , sortBy
    ) where

import           ML.BORL.Action
import           ML.BORL.Algorithm
import           ML.BORL.Calculation
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

expSmthPsi :: Double
expSmthPsi = 0.03

keepXLastValues :: Int
keepXLastValues = 100

approxAvg :: Double
approxAvg = fromIntegral (100 :: Int)

fileStateValues :: FilePath
fileStateValues = "stateValues"

fileEpisodeLength :: FilePath
fileEpisodeLength = "episodeLength"


steps :: (NFData s, Ord s) => BORL s -> Integer -> IO (BORL s)
steps (force -> borl) nr =
  case find isTensorflow (allProxies $ borl ^. proxies) of
    Nothing -> runMonadBorlIO $ force <$> foldM (\b _ -> nextAction (force b) >>= fmap force . stepExecute) borl [0 .. nr - 1]
    Just _ ->
      runMonadBorlTF $ do
        void $ restoreTensorflowModels True borl
        !borl' <- foldM (\b _ -> nextAction (force b) >>= fmap force . stepExecute) borl [0 .. nr - 1]
        force <$> saveTensorflowModels borl'


step :: (NFData s, Ord s) => BORL s -> IO (BORL s)
step (force -> borl) =
  case find isTensorflow (allProxies $ borl ^. proxies) of
    Nothing -> nextAction borl >>= stepExecute
    Just _ ->
      runMonadBorlTF $ do
        void $ restoreTensorflowModels True borl
        !borl' <- nextAction borl >>= stepExecute
        force <$> saveTensorflowModels borl'

-- | This keeps the Tensorflow session alive. For non-Tensorflow BORL data structures this is equal to step.
stepM :: (MonadBorl' m, NFData s, Ord s) => BORL s -> m (BORL s)
stepM (force -> borl) = nextAction borl >>= fmap force . stepExecute

-- | This keeps the Tensorflow session alive. For non-Tensorflow BORL data structures this is equal to steps, but forces
-- evaluation of the data structure every 1000 periods.
stepsM :: (MonadBorl' m, NFData s, Ord s) => BORL s -> Integer -> m (BORL s)
stepsM (force -> borl) nr = do
  !borl' <- force <$> foldM (\b _ -> nextAction b >>= stepExecute) borl [1 .. min maxNr nr]
  if nr > maxNr
    then stepsM borl' (nr - maxNr)
    else return borl'
  where maxNr = 1000


-- restoreTensorflowModels :: (MonadBorl' m) => BORL s -> m ()
-- restoreTensorflowModels borl = do
--   buildModels
--   mapM_ restoreProxy (allProxies $ borl ^. proxies)
--   where
--     restoreProxy px =
--       case px of
--         TensorflowProxy netT netW _ _ _ _ -> restoreModelWithLastIO netT >> restoreModelWithLastIO netW >> return ()
--         _ -> return ()
--     buildModels =
--       case find isTensorflow (allProxies $ borl ^. proxies) of
--         Just (TensorflowProxy netT _ _ _ _ _) -> buildTensorflowModel netT
--         _                                     -> return ()


-- saveTensorflowModels :: (MonadBorl' m) => BORL s -> m (BORL s)
-- saveTensorflowModels borl = do
--   mapM_ saveProxy (allProxies $ borl ^. proxies)
--   return borl
--   where
--     saveProxy px =
--       case px of
--         TensorflowProxy netT netW _ _ _ _ -> saveModelWithLastIO netT >> saveModelWithLastIO netW >> return ()
--         _ -> return ()


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


stepExecute :: forall m s . (MonadBorl' m, NFData s, Ord s) => (BORL s, Bool, ActionIndexed s) -> m (BORL s)
stepExecute (borl, randomAction, (aNr, Action action _)) = do
  let state = borl ^. s
      period = borl ^. t
  (reward, stateNext, episodeEnd) <- liftSimple $ action state
  (proxies', calc) <- P.insert period state aNr randomAction reward stateNext episodeEnd (mkCalculation borl) (borl ^. proxies)
  let lastVsLst = fromMaybe [0] (getLastVs' calc)
  -- File IO Operations
  when (period == 0) $
    -- liftSimple $ mapM_ (\f -> doesFileExist f >>= \exists -> when exists (removeFile f)) [fileStateValues, fileEpisodeLength]
   do
    liftSimple $ writeFile fileStateValues "Period\tRho\tMinRho\tVAvg\tR0\tR1\n"
    liftSimple $ writeFile fileEpisodeLength "Episode\tEpisodeLength\n"
  let strRho = show (fromMaybe 0 (getRhoVal' calc))
      strMinV = show (fromMaybe 0 (getRhoMinimumVal' calc))
      strVAvg = show (avg lastVsLst)
      strR0 = show $ fromMaybe 0 (getR0ValState' calc)
      strR1 = show $ getR1ValState' calc
      avg xs = sum xs / fromIntegral (length xs)

  liftSimple $ appendFile fileStateValues (show period ++ "\t" ++ strRho ++ "\t" ++ strMinV ++ "\t" ++ strVAvg ++ "\t" ++ strR0 ++ "\t" ++ strR1 ++ "\n")
  let (eNr, eStart) = borl ^. episodeNrStart
      eLength = borl ^. t - eStart
  when (getEpisodeEnd calc) $ liftSimple $ appendFile fileEpisodeLength (show eNr ++ "\t" ++ show eLength ++ "\n")
  let divideAfterGrowth borl =
        case borl ^. algorithm of
          AlgBORL _ _ _ (DivideValuesAfterGrowth nr maxPeriod) _
            | period > maxPeriod -> borl
            | length lastVsLst == nr && endOfIncreasedStateValues -> trace ("multiply in period " ++ show period ++ " by " ++ show val) $
                                                                     foldl (\q f -> over (proxies . f) (multiplyProxy val) q) (set phase SteadyStateValues borl) [psiV, v, w]
            where endOfIncreasedStateValues =
                    borl ^. phase == IncreasingStateValues && 2000 * (avg (take (nr `div` 2) lastVsLst) - avg (drop (nr `div` 2) lastVsLst)) / fromIntegral nr < 0.00
                  val = 0.2 / (sum lastVsLst / fromIntegral nr)
          _ -> borl
      setCurrentPhase borl = case borl ^. algorithm of
        AlgBORL _ _ _ (DivideValuesAfterGrowth nr _) _
          | length lastVsLst == nr && increasingStateValue -> trace ("period: " ++ show period) $ trace (show IncreasingStateValues) $ set phase IncreasingStateValues borl
          where increasingStateValue = borl ^. phase /= IncreasingStateValues && 2000 * (avg (take (nr `div` 2) lastVsLst) - avg (drop (nr `div` 2) lastVsLst)) / fromIntegral nr > 0.20
        _ -> borl
  -- update values
  let setEpisode curEp
        | getEpisodeEnd calc = (eNr + 1, borl ^. t)
        | otherwise = curEp
  return $
    force $ -- needed to ensure constant memory consumption
    setCurrentPhase $ divideAfterGrowth $
    set psis (fromMaybe 0 (getPsiValRho' calc), fromMaybe 0 (getPsiValV' calc), fromMaybe 0 (getPsiValW' calc)) $
    set lastVValues (fromMaybe [] (getLastVs' calc)) $
    set lastRewards (getLastRews' calc) $ set proxies proxies' $ set s stateNext $ set t (period + 1) $
    over episodeNrStart setEpisode
#ifdef DEBUG
    $ set visits (M.alter (\mV -> ((+ 1) <$> mV) <|> Just 1) state (borl ^. visits))
#endif
    borl


-- | This function chooses the next action from the current state s and all possible actions.
nextAction :: (MonadBorl' m, Ord s) => BORL s -> m (BORL s, Bool, ActionIndexed s)
nextAction borl
  | null as = error "Empty action list"
  | length as == 1 = return (borl, False, head as)
  | otherwise = do
    rand <- liftSimple $ randomRIO (0, 1)
    if rand < explore
      then do
        r <- liftSimple $ randomRIO (0, length as - 1)
        return (borl, True, as !! r)
      else case borl ^. algorithm of
             AlgBORL _ _ _ _ decideVPlusPsi -> do
               bestRho <-
                 if isUnichain borl
                   then return as
                   else do
                     rhoVals <- mapM (rhoValue borl state) (map fst as)
                     return $ map snd $ head $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (zip rhoVals as)
               bestV <-
                 do vVals <- mapM (vValue decideVPlusPsi borl state) (map fst bestRho)
                    return $ map snd $ head $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (zip vVals bestRho)
               bestE <-
                 do eVals <- mapM (eValue borl state) (map fst bestV)
                    return $ map snd $ sortBy (epsCompare compare `on` fst) (zip eVals bestV)
               if length bestE > 1
                 then do
                   r <- liftSimple $ randomRIO (0, length bestE - 1)
                   return (borl, False, bestE !! r)
                 else return (borl, False, head bestE)
             AlgDQN {} -> dqnNextAction
             AlgDQNAvgRew {} -> dqnNextAction
  where
    params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
    eps = params' ^. epsilon
    explore = params' ^. exploration
    state = borl ^. s
    as = actionsIndexed borl state
    epsCompare = epsCompareWith eps
    dqnNextAction = do
      rValues <- mapM (rValue borl RBig state . fst) as
      let bestR = sortBy (epsCompare compare `on` fst) (zip rValues as)
               -- liftSimple $ putStrLn ("bestR: " ++ show bestR)
      return (borl, False, snd $ head bestR)

epsCompareWith :: (Ord t, Num t) => t -> (t -> t -> p) -> t -> t -> p
epsCompareWith eps f x y
  | abs (x - y) <= eps = f 0 0
  | otherwise = y `f` x


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


-- | Used to select a discount factor.
data RSize
  = RSmall
  | RBig


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


