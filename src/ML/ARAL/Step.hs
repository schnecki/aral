{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TupleSections       #-}
module ML.ARAL.Step
    ( step
    , steps
    , stepM
    , stepsM
    , stepExecute
    , nextAction
    , epsCompareWith
    , sortBy
    ) where

#ifdef DEBUG
import           Data.List                          (elemIndex)
#endif
import           Control.Applicative                ((<|>))
import           Control.Arrow                      ((&&&), (***))
import           Control.Concurrent.MVar
import           Control.DeepSeq
import           Control.Exception
import           Control.Lens
import           Control.Monad
import           Control.Monad.IO.Class             (MonadIO, liftIO)
import           Control.Parallel.Strategies        hiding (r0)
import           Data.Either                        (isLeft)
import           Data.Function                      (on)
import           Data.List                          (find, foldl', groupBy, intercalate, maximumBy, partition, sortBy, transpose)
import qualified Data.Map.Strict                    as M
import           Data.Maybe                         (fromMaybe, isJust)
import           Data.Ord
import           Data.Serialize
import qualified Data.Text                          as T
import qualified Data.Vector                        as VB
import qualified Data.Vector.Storable               as V
import           EasyLogger
import           GHC.Generics
import           Grenade
import           Say
import           System.Directory
import           System.IO
import           System.IO.Unsafe                   (unsafePerformIO)
import           System.Random
import           Text.Printf

import           ML.ARAL.Action
import           ML.ARAL.Algorithm
import           ML.ARAL.Calculation
import           ML.ARAL.Fork
import           ML.ARAL.NeuralNetwork.NNConfig
import           ML.ARAL.NeuralNetwork.ReplayMemory
import           ML.ARAL.NeuralNetwork.Scaling
import           ML.ARAL.Parameters
import           ML.ARAL.Properties
import           ML.ARAL.Proxy                      as P
import           ML.ARAL.Reward
import           ML.ARAL.Serialisable
import           ML.ARAL.Settings
import           ML.ARAL.Type
import           ML.ARAL.Types
import           ML.ARAL.Workers.Type


import           Debug.Trace

fileDebugStateV :: FilePath
fileDebugStateV = "stateVAllStates"

fileDebugStateVScaled :: FilePath
fileDebugStateVScaled = "stateVAllStates_scaled"

fileDebugStateW :: FilePath
fileDebugStateW = "stateWAllStates"

fileDebugPsiVValues :: FilePath
fileDebugPsiVValues = "statePsiVAllStates"

fileDebugPsiWValues :: FilePath
fileDebugPsiWValues = "statePsiWAllStates"

fileStateValues :: FilePath
fileStateValues = "stateValues"

fileDebugStateValuesNrStates :: FilePath
fileDebugStateValuesNrStates = "stateValuesAllStatesCount"

fileDebugStateValuesAgents :: FilePath
fileDebugStateValuesAgents = "stateValuesAgents"


fileReward :: FilePath
fileReward = "reward"

fileEpisodeLength :: FilePath
fileEpisodeLength = "episodeLength"


steps :: (NFData s, NFData as, Ord s, RewardFuture s, Eq as) => ARAL s as -> Integer -> IO (ARAL s as)
steps !borl nr =
  fmap force $!
  liftIO $ foldM (\b _ -> nextAction b >>= stepExecute b) borl [0 .. nr - 1]


step :: (NFData s, NFData as, Ord s, RewardFuture s, Eq as) => ARAL s as -> IO (ARAL s as)
step !borl =
  fmap force $!
  nextAction borl >>= stepExecute borl

-- | This keeps the MonadIO alive.
stepM :: (MonadIO m, NFData s, NFData as, Ord s, RewardFuture s, Eq as) => ARAL s as -> m (ARAL s as)
stepM !borl = nextAction borl >>= stepExecute borl >>= \(b@ARAL{}) -> return (force b)

-- | This keeps the MonadIO session alive. This is equal to steps, but forces evaluation of the data structure every 100 periods.
stepsM :: (MonadIO m, NFData s, NFData as, Ord s, RewardFuture s, Eq as) => ARAL s as -> Integer -> m (ARAL s as)
stepsM borl nr = do
  borl' <- foldM (\b _ -> stepM b) borl [1 .. min maxNr nr]
  if nr > maxNr
    then stepsM borl' (nr - maxNr)
    else return borl'
  where maxNr = 100

stepExecute :: forall m s as . (MonadIO m, NFData s, NFData as, Ord s, RewardFuture s, Eq as) => ARAL s as -> NextActions -> m (ARAL s as)
stepExecute borl (as, workerActions) = do
  let state = borl ^. s
      period = borl ^. t + length (borl ^. futureRewards)
  -- File IO Operations
  when (period == 0) $ liftIO $ do
    let agents = borl ^. settings . independentAgents
        times txt = concatMap (\nr -> "\t" <> txt <> "-Ag" <> show nr) [1..agents]
    writeFile fileDebugStateValuesAgents (show agents)
    writeFile fileStateValues $ "Period" ++ times "Rho" ++ "\tExpSmthRho" ++ times "RhoOverEstimated" ++ times "MinRho" ++ times "VAvg" ++
      times "R0" ++ times "R1" ++ times "R0_scaled" ++ times "R1_scaled" ++ "\tMinValue\tMaxValue" ++ "\n"
    writeFile fileEpisodeLength "Episode\tEpisodeLength\n"
    writeFile fileReward "Period\tReward\n"
  workerRefs <- liftIO $ runWorkerActions (set t period borl) workerActions
  let action agTp s as = (borl ^. actionFunction) agTp s (VB.toList as)
      actList = borl ^. actionList
      actNrs = VB.map snd as
      acts = VB.map (actList VB.!) actNrs
  (reward, stateNext, episodeEnd) <- liftIO $ action MainAgent state acts
  let borl' = over futureRewards (applyStateToRewardFutureData state . (`VB.snoc` RewardFutureData period state as reward stateNext episodeEnd)) borl
  (dropLen, _, newBorl) <- foldM (stepExecuteMaterialisedFutures MainAgent) (0, False, borl') (borl' ^. futureRewards)
  newWorkers <- liftIO $ collectWorkers workerRefs
  return $ set workers newWorkers $ over futureRewards (VB.drop dropLen) $ set s stateNext newBorl
  where
    collectWorkers (Left xs)      = return xs
    collectWorkers (Right ioRefs) = mapM collectForkResult ioRefs


-- | This functions takes one step for all workers, and returns the new worker replay memories and future reward data
-- lists.
runWorkerActions :: (NFData s) => ARAL s as -> [WorkerActionChoice] -> IO (Either (Workers s) [IORef (ThreadState (WorkerState s))])
runWorkerActions _ [] = return (Left [])
runWorkerActions borl _ | borl ^. settings . disableAllLearning = return (Left $ borl ^. workers)
runWorkerActions borl acts = Right <$> zipWithM (\act worker -> doFork' $ runWorkerAction borl worker act) acts (borl ^. workers)
  where
    doFork'
      | borl ^. settings . useProcessForking = doFork
      | otherwise = doForkFake

-- | Apply the given state to a list of future reward data.
applyStateToRewardFutureData :: State s -> VB.Vector (RewardFutureData s) -> VB.Vector (RewardFutureData s)
applyStateToRewardFutureData state = VB.map (over futureReward applyToReward)
  where
    applyToReward (RewardFuture storage) = applyState storage state
    applyToReward r                      = r

-- | Run one worker.
runWorkerAction :: (NFData s) => ARAL s as -> WorkerState s -> WorkerActionChoice -> IO (WorkerState s)
runWorkerAction borl (WorkerState wNr state replMem oldFutureRewards rew) as = do
  let action agTp s as = (borl ^. actionFunction) agTp s (VB.toList as)
      actList = borl ^. actionList
      actNrs = VB.map snd as
      acts = VB.map (actList VB.!) actNrs
  (reward, stateNext, episodeEnd) <- liftIO $ action (WorkerAgent wNr) state acts
  let newFuturesUndropped = applyStateToRewardFutureData state (oldFutureRewards `VB.snoc` RewardFutureData (borl ^. t) state as reward stateNext episodeEnd)
  let (materialisedFutures, newFutures) = splitMaterialisedFutures newFuturesUndropped
  let addNewRewardToExp currentExpSmthRew (RewardFutureData _ _ _ (Reward rew') _ _) = (1 - expSmthPsi) * currentExpSmthRew + expSmthPsi * rew'
      addNewRewardToExp _ _                                                          = error "unexpected RewardFutureData in runWorkerAction"
  newReplMem <- foldM addExperience replMem materialisedFutures
  return $! force $ WorkerState wNr stateNext newReplMem newFutures (foldl' addNewRewardToExp rew materialisedFutures)
  where
    splitMaterialisedFutures fs =
      let (futures, finished) = VB.partition (isRewardFuture . view futureReward) fs
       in (VB.filter (not . isRewardEmpty . view futureReward) finished, futures)
    addExperience replMem' (RewardFutureData _ state' as' (Reward reward) stateNext episodeEnd) = do
      let (_, stateActs, stateNextActs) = mkStateActs borl state' stateNext
      liftIO $ addToReplayMemories (borl ^. settings . nStep) (stateActs, as', reward, stateNextActs, episodeEnd) replMem'
    addExperience _ _ = error "Unexpected Reward in calcExperience of runWorkerActions!"

-- | This function exectues all materialised rewards until a non-materialised reward is found, i.e. add a new experience
-- to the replay memory and then, select and learn from the experiences of the replay memory.
stepExecuteMaterialisedFutures ::
     forall m s as. (MonadIO m, NFData s, NFData as, Ord s, RewardFuture s, Eq as)
  => AgentType
  -> (Int, Bool, ARAL s as)
  -> RewardFutureData s
  -> m (Int, Bool, ARAL s as)
stepExecuteMaterialisedFutures _ (nr, True, borl) _ = return (nr, True, borl)
stepExecuteMaterialisedFutures agent (nr, _, borl) dt =
  case view futureReward dt of
    RewardEmpty     -> return (nr, False, borl)
    RewardFuture {} -> return (nr, True, borl)
    Reward {}       -> (nr+1, False, ) <$> execute borl agent dt


minMaxStates :: MVar ((Double, (s, AgentActionIndices)), (Double, (s, AgentActionIndices)))
minMaxStates = unsafePerformIO newEmptyMVar
{-# NOINLINE minMaxStates #-}

hasLocked :: String -> IO a -> IO a
hasLocked msg action =
  action `catches`
  [ Handler $ \exc@BlockedIndefinitelyOnMVar -> sayString ("[MVar]: " ++ msg) >> throwIO exc
  , Handler $ \exc@BlockedIndefinitelyOnSTM -> sayString ("[STM]: " ++ msg) >> throwIO exc
  ]

updateMinMax :: ARAL s as -> AgentActionIndices -> Calculation -> IO (Double, Double)
updateMinMax borl as calc = do
  mMinMax <- hasLocked "updateMinMax tryReadMVar" $ tryReadMVar minMaxStates
  let minMax' =
        case mMinMax of
          Nothing                                -> ((V.minimum value, (borl ^. s, as)), (V.maximum value, (borl ^. s, as)))
          Just minMax@((minVal, _), (maxVal, _)) -> bimap (replaceIf V.minimum (V.minimum value < minVal)) (replaceIf V.maximum (V.maximum value > maxVal)) minMax
  empty <- isEmptyMVar minMaxStates
  when (empty || borl ^. t == 0) $ void $ hasLocked "updateMinMax putMVar" $ tryPutMVar minMaxStates minMax'
  when (fmap (bimap fst fst) mMinMax /= Just (bimap fst fst minMax')) $ hasLocked "updateMinMax modifyMVar 1" $ modifyMVar_ minMaxStates (const $ return minMax')
  when (borl ^. t `mod` 1000 == 0) $ do
    let ((_, (minS, minA)), (_, (maxS, maxA))) = minMax'
    AgentValue vMin <- valueFunction minS minA
    AgentValue vMax <- valueFunction maxS maxA
    let res = ((V.minimum vMin, (minS, minA)), (V.maximum vMax, (maxS, maxA)))
    hasLocked "updateMinMax modifyMVar 2" $ modifyMVar_ minMaxStates (const $ return res)
  return $ bimap fst fst minMax'
  where
    replaceIf reduce True _ = (reduce value, (borl ^. s, as))
    replaceIf _ False x     = x
    AgentValue value =
      fromMaybe (error "unexpected empty value in updateMinMax") $
      case borl ^. algorithm of
        AlgDQNAvgRewAdjusted {} -> getR1ValState' calc
        AlgDQN {}               -> getR1ValState' calc
        _                       -> getVValState' calc
    valueFunction =
      case borl ^. algorithm of
        AlgDQNAvgRewAdjusted {} -> rValue borl RBig
        AlgDQN {}               -> rValue borl RBig
        _                       -> vValue borl


-- | Execute the given step, i.e. add a new experience to the replay memory and then, select and learn from the
-- experiences of the replay memory.
execute :: (MonadIO m, NFData s, NFData as, Ord s, RewardFuture s, Eq as) => ARAL s as -> AgentType -> RewardFutureData s -> m (ARAL s as)
execute borl agent (RewardFutureData period state as (Reward reward) stateNext episodeEnd) = do
#ifdef DEBUG
  borl <- if isMainAgent agent
          then do
            when (borl ^. t == 0) $ forM_ [fileDebugStateV, fileDebugStateW, fileDebugPsiWValues, fileDebugPsiVValues, fileDebugPsiWValues, fileDebugStateValuesNrStates] $ \f ->
              liftIO $ doesFileExist f >>= \x -> when x (removeFile f)
            writeDebugFiles borl
          else return borl
#endif
  (proxies', calc) <- P.insert borl agent period state as reward stateNext episodeEnd (mkCalculation borl) (borl ^. proxies)
  let lastVsLst = fromMaybe (VB.singleton $ toValue agents 0) (getLastVs' calc)
      strVAvg = map (show . avg) $ transpose $ VB.toList $ VB.map fromValue lastVsLst
      rhoVal = fromMaybe (toValue agents 0) (getRhoVal' calc)
      strRho = map show (fromValue rhoVal)
      strRhoSmth = show (borl ^. expSmoothedReward)
      strRhoOver = map (show . overEstimateRhoCalc borl) (fromValue rhoVal)
      strMinRho = map show $ fromValue $ fromMaybe (toValue agents 0) (getRhoMinimumVal' calc)
      strR0 = map show $ fromValue $ fromMaybe (toValue agents 0) (getR0ValState' calc)
      mCfg = borl ^? proxies . v . proxyNNConfig <|> borl ^? proxies . r1 . proxyNNConfig
      strR0Scaled = map show $ fromValue $ fromMaybe (toValue agents 0) $ do
         scAlg <- view scaleOutputAlgorithm <$> mCfg
         scParam <- view scaleParameters <$> mCfg
         v <- getR0ValState' calc
         return $ scaleValue scAlg (Just (scParam ^. scaleMinR0Value, scParam ^. scaleMaxR0Value)) v
      strR1 = map show $ fromValue $ fromMaybe (toValue agents 0) (getR1ValState' calc)
      strR1Scaled = map show $ fromValue $ fromMaybe (toValue agents 0) $ do
         scAlg <- view scaleOutputAlgorithm <$> mCfg
         scParam <- view scaleParameters <$> mCfg
         v <- getR1ValState' calc
         return $ scaleValue scAlg (Just (scParam ^. scaleMinR1Value, scParam ^. scaleMaxR1Value)) v
      avg xs = sum xs / fromIntegral (length xs)
      agents = borl ^. settings . independentAgents
      list = concatMap ("\t" <>)
      zero = toValue agents 0
  if isMainAgent agent
  then do
    (minVal, maxVal) <- liftIO $ updateMinMax borl (VB.map snd as) calc
    let minMaxValTxt = "\t" ++ show minVal ++ "\t" ++ show maxVal
    liftIO $ appendFile fileStateValues (show period ++ list strRho ++ "\t" ++ strRhoSmth ++ list strRhoOver ++ list strMinRho ++ list strVAvg ++ list strR0 ++ list strR1 ++ list strR0Scaled ++ list strR1Scaled ++ minMaxValTxt ++ "\n" )
    let (eNr, eStart) = borl ^. episodeNrStart
        eLength = borl ^. t - eStart
    when (getEpisodeEnd calc) $ liftIO $ appendFile fileEpisodeLength (show eNr ++ "\t" ++ show eLength ++ "\n")
    liftIO $ appendFile fileReward (show period ++ "\t" ++ show reward ++ "\n")
    -- update values
    let setEpisode curEp
          | getEpisodeEnd calc = (eNr + 1, borl ^. t)
          | otherwise = curEp
    return $
      set psis (fromMaybe zero (getPsiValRho' calc), fromMaybe zero (getPsiValV' calc), fromMaybe zero (getPsiValW' calc)) $ set expSmoothedReward (getExpSmoothedReward' calc) $
      set lastVValues (fromMaybe mempty (getLastVs' calc)) $ set lastRewards (getLastRews' calc) $ set proxies proxies' $ set t (period + 1) $ over episodeNrStart setEpisode $ maybeFlipDropout borl
  else return $
       set psis (fromMaybe zero (getPsiValRho' calc), fromMaybe zero (getPsiValV' calc), fromMaybe zero (getPsiValW' calc)) $
       set proxies proxies' $ set expSmoothedReward (getExpSmoothedReward' calc) $ set t (period + 1) $ maybeFlipDropout borl

execute _ _ _ = error "Exectue on invalid data structure. This is a bug!"

-- | Flip the dropout active/inactive state.
maybeFlipDropout :: ARAL s as -> ARAL s as
maybeFlipDropout borl =
  case borl ^? proxies . v . proxyNNConfig <|> borl ^? proxies . r1 . proxyNNConfig of
    Just cfg@NNConfig {}
      | borl ^. t == cfg ^. grenadeDropoutOnlyInactiveAfter -> setDropoutValue False borl
      | borl ^. t > cfg ^. grenadeDropoutOnlyInactiveAfter -> borl
      | borl ^. t `mod` cfg ^. grenadeDropoutFlipActivePeriod == 0 ->
        let occurance = borl ^. t `div` cfg ^. grenadeDropoutFlipActivePeriod
            value
              | even occurance = True
              | otherwise = False
         in setDropoutValue value borl
    _ -> borl
  where
    setDropoutValue :: Bool -> ARAL s as -> ARAL s as
    setDropoutValue val =
      overAllProxies
        (filtered isGrenade)
        (\(Grenade tar wor tp cfg act agents) -> Grenade (runSettingsUpdate (NetworkSettings val) tar) (runSettingsUpdate (NetworkSettings val) wor) tp cfg act agents)


#ifdef DEBUG

stateFeatures :: MVar [a]
stateFeatures = unsafePerformIO $ newMVar mempty
{-# NOINLINE stateFeatures #-}

setStateFeatures :: (MonadIO m) => [a] -> m ()
setStateFeatures x = liftIO $ hasLocked "setStateFeatures" $ modifyMVar_ stateFeatures (return . const x)

getStateFeatures :: (MonadIO m) => m [a]
getStateFeatures = liftIO $ hasLocked "getStateFeatures" $ fromMaybe mempty <$> tryReadMVar stateFeatures


writeDebugFiles :: (MonadIO m, NFData s, NFData as, Ord s, Eq as, RewardFuture s) => ARAL s as -> m (ARAL s as)
writeDebugFiles borl = do
  let isDqn = isAlgDqn (borl ^. algorithm) || isAlgDqnAvgRewardAdjusted (borl ^. algorithm)
  let isAnn
        | isDqn = P.isNeuralNetwork (borl ^. proxies . r1)
        | otherwise = P.isNeuralNetwork (borl ^. proxies . v)
  let putStateFeatList borl xs
        | isAnn = borl
        | otherwise = setAllProxies proxyTable xs' borl
        where
          xs' = M.fromList $ zip (map (\xs -> (V.init xs, round (V.last xs))) xs) (repeat 0)
  borl' <-
    if borl ^. t > 0
      then return borl
      else do
        liftIO $ writeFile fileDebugStateV ""
        liftIO $ writeFile fileDebugStateVScaled ""
        liftIO $ writeFile fileDebugStateW ""
        liftIO $ writeFile fileDebugPsiVValues ""
        liftIO $ writeFile fileDebugPsiWValues ""
        borl' <-
          if isAnn
            then return borl
            else stepsM (setAllProxies (proxyNNConfig . replayMemoryMaxSize) 1000 $ set t 1 borl) debugStepsCount -- run steps to fill the table with (hopefully) all states
        let stateFeats
              | isDqn = getStateFeatList (borl' ^. proxies . r1)
              | otherwise = getStateFeatList (borl' ^. proxies . v)
        setStateFeatures stateFeats
        liftIO $ writeFile fileDebugStateValuesNrStates (show $ length stateFeats)
        liftIO $ forM_ [fileDebugStateV, fileDebugStateVScaled, fileDebugStateW, fileDebugPsiVValues, fileDebugPsiWValues] $
          flip writeFile ("Period\t" <> mkListStrAg (shorten . printStateFeat) stateFeats <> "\n")
        if isNeuralNetwork (borl ^. proxies . v)
          then return borl
          else do
            liftIO $ putStrLn $ "[DEBUG INFERRED NUMBER OF STATES]: " <> show (length stateFeats)
            return $ borl -- putStateFeatList borl stateFeats
  stateFeatsLoaded <- getStateFeatures
  let stateFeats
        | not (null stateFeatsLoaded) = stateFeatsLoaded
        | isDqn = getStateFeatList (borl' ^. proxies . r1)
        | otherwise = getStateFeatList (borl' ^. proxies . v)
  when ((borl' ^. t `mod` debugPrintCount) == 0) $ do
    let splitIdx =
          V.length (head stateFeats) - agents
    stateValuesV <- mapM (\xs ->
                            let (st,as) = V.splitAt splitIdx xs in
                            if isDqn then rValueWith Worker borl' RBig st (VB.map round $ V.convert as) else vValueWith Worker borl' st (VB.map round $ V.convert as)) stateFeats
    stateValuesVScaled <- mapM (\xs ->
                                  let (st,as) = V.splitAt splitIdx xs in
                                  if isDqn then rValueNoUnscaleWith Worker borl' RBig st (VB.map round $ V.convert as) else vValueNoUnscaleWith Worker borl' st (VB.map round $ V.convert as)) stateFeats
    stateValuesW <- mapM (\xs -> let (st,as) = V.splitAt splitIdx xs in if isDqn then return 0 else wValueFeat borl' st (VB.map round $ V.convert as)) stateFeats
    liftIO $ appendFile fileDebugStateV (show (borl' ^. t) <> "\t" <> mkListStrV show stateValuesV <> "\n")
    liftIO $ appendFile fileDebugStateVScaled (show (borl' ^. t) <> "\t" <> mkListStrV show stateValuesVScaled <> "\n")
    when (isAlgBorl (borl ^. algorithm)) $ do
      liftIO $ appendFile fileDebugStateW (show (borl' ^. t) <> "\t" <> mkListStrV show stateValuesW <> "\n")
      psiVValues <- mapM (\xs -> let (st,as) = V.splitAt splitIdx xs in psiVFeat borl' st (VB.map round $ V.convert as)) stateFeats
      liftIO $ appendFile fileDebugPsiVValues (show (borl' ^. t) <> "\t" <> mkListStrV show psiVValues <> "\n")
      psiWValues <- mapM (\xs -> let (st,as) = V.splitAt splitIdx xs in psiWFeat borl' st (VB.map round $ V.convert xs)) stateFeats
      liftIO $ appendFile fileDebugPsiWValues (show (borl' ^. t) <> "\t" <> mkListStrV show psiWValues <> "\n")
  return borl'
  where
    getStateFeatList :: Proxy -> [V.Vector Double]
    getStateFeatList Scalar {} = []
    getStateFeatList (Table t _ _) = -- map fst (M.keys t)
      map (\(xs, y) -> xs V.++ V.replicate agents (fromIntegral y)) (M.keys t)
    getStateFeatList nn = concatMap (\xs -> map (\a -> xs V.++ V.replicate agents (fromIntegral $ actIdx a)) acts) (nn ^. proxyNNConfig . prettyPrintElems)
    actIdx a = fromMaybe (-1) (elemIndex a acts)
    acts = VB.toList $ borl ^. actionList
    agents = borl ^. settings . independentAgents
    mkListStrAg :: (a -> String) -> [a] -> String
    mkListStrAg f = intercalate "\t" . concatMap (\x -> map (\nr -> f x <> "-Ag" <> show nr) [1..agents])
    mkListStrV :: (Double -> String) -> [Value] -> String
    mkListStrV f = intercalate "\t" . concatMap (map f . fromValue)
    shorten xs | length xs > 60 = "..." <> drop (length xs - 60) xs
               | otherwise = xs
    printStateFeat :: StateFeatures -> String
    printStateFeat xs = "[" <> intercalate "," (map (printf "%.2f") (V.toList xs)) <> "]"
    psiVFeat borl stateFeat aNr = P.lookupProxy (borl ^. t) Worker (stateFeat, aNr) (borl ^. proxies . psiV)
    psiWFeat borl stateFeat aNr = P.lookupProxy (borl ^. t) Worker (stateFeat, aNr) (borl ^. proxies . psiW)

debugStepsCount :: Integer
debugStepsCount = 8000

debugPrintCount :: Int
debugPrintCount = 100

#endif
