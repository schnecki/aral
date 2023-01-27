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
    , setDropoutValue
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
import           ML.ARAL.NeuralNetwork.Hasktorch
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
steps !aral nr =
  fmap force $!
  liftIO $ foldM (\b _ -> nextAction b >>= stepExecute b) aral [0 .. nr - 1]


step :: (NFData s, NFData as, Ord s, RewardFuture s, Eq as) => ARAL s as -> IO (ARAL s as)
step !aral =
  fmap force $!
  nextAction aral >>= stepExecute aral

-- | This keeps the MonadIO alive and force evaluation of ARAL in every step.
stepM :: (MonadIO m, NFData s, NFData as, Ord s, RewardFuture s, Eq as) => ARAL s as -> m (ARAL s as)
stepM !aral = nextAction aral >>= stepExecute aral >>= \b@ARAL{} -> return (force b)

-- | This keeps the MonadIO session alive. This is equal to steps, but forces evaluation of the data structure every 1000 steps.
stepsM :: (MonadIO m, NFData s, NFData as, Ord s, RewardFuture s, Eq as) => ARAL s as -> Integer -> m (ARAL s as)
stepsM !aral !nr = do
  aral' <- foldM (\b _ -> nextAction b >>= stepExecute b) aral [1 .. min maxNr nr]
  if nr > maxNr
    then stepsM (force aral') (nr - maxNr)
    else return $! force aral'
  where
    maxNr = 1000

stepExecute :: forall m s as . (MonadIO m, NFData s, NFData as, Ord s, RewardFuture s, Eq as) => ARAL s as -> NextActions -> m (ARAL s as)
stepExecute aral (as, workerActions) = do
  let state = aral ^. s
      period = aral ^. t + length (aral ^. futureRewards)
  -- File IO Operations
  when (period == 0) $ liftIO $ do
    let agents = aral ^. settings . independentAgents
        times txt = concatMap (\nr -> "\t" <> txt <> "-Ag" <> show nr) [1..agents]
    writeFile fileDebugStateValuesAgents (show agents)
    writeFile fileStateValues $ "Period" ++ times "Rho" ++ "\tExpSmthRho" ++ times "RhoOverEstimated" ++ times "MinRho" ++ times "VAvg" ++
      times "R0" ++ times "R1" ++ times "R0_scaled" ++ times "R1_scaled" ++ "\tMinValue\tMaxValue" ++ "\n"
    writeFile fileEpisodeLength "Episode\tEpisodeLength\n"
    writeFile fileReward "Period\tReward\n"
  workerRefs <- liftIO $ runWorkerActions (set t period aral) workerActions
  let action agTp s as = (aral ^. actionFunction) aral agTp s (VB.toList as)
      actList = aral ^. actionList
      actNrs = VB.map snd as
      acts = VB.map (actList VB.!) actNrs
  (reward, stateNext, episodeEnd) <- liftIO $ action MainAgent state acts
  let aral' = over futureRewards (applyStateToRewardFutureData state . (`VB.snoc` RewardFutureData period state as reward stateNext episodeEnd)) aral
  (dropLen, _, newAral) <- foldM (stepExecuteMaterialisedFutures MainAgent) (0, False, aral') (aral' ^. futureRewards)
  newWorkers <- liftIO $ collectWorkers workerRefs
  return $ set workers newWorkers $ over futureRewards (VB.drop dropLen) $ set s stateNext newAral
  where
    collectWorkers (Left xs)      = return xs
    collectWorkers (Right ioRefs) = mapM collectForkResult ioRefs


-- | This functions takes one step for all workers, and returns the new worker replay memories and future reward data
-- lists.
runWorkerActions :: (NFData s) => ARAL s as -> [WorkerActionChoice] -> IO (Either (Workers s) [IORef (ThreadState (WorkerState s))])
runWorkerActions _ [] = return (Left [])
runWorkerActions aral _ | aral ^. settings . disableAllLearning = return (Left $ aral ^. workers)
runWorkerActions aral acts = Right <$> zipWithM (\act worker -> doFork' $ runWorkerAction aral worker act) acts (aral ^. workers)
  where
    doFork'
      | aral ^. settings . useProcessForking = doFork
      | otherwise = doForkFake

-- | Apply the given state to a list of future reward data.
applyStateToRewardFutureData :: State s -> VB.Vector (RewardFutureData s) -> VB.Vector (RewardFutureData s)
applyStateToRewardFutureData state = VB.map (over futureReward applyToReward)
  where
    applyToReward (RewardFuture storage) = applyState storage state
    applyToReward r                      = r

-- | Run one worker.
runWorkerAction :: (NFData s) => ARAL s as -> WorkerState s -> WorkerActionChoice -> IO (WorkerState s)
runWorkerAction aral (WorkerState wNr state replMem oldFutureRewards rew) as = do
  let action agTp s as = (aral ^. actionFunction) aral agTp s (VB.toList as)
      actList = aral ^. actionList
      actNrs = VB.map snd as
      acts = VB.map (actList VB.!) actNrs
  (reward, stateNext, episodeEnd) <- liftIO $ action (WorkerAgent wNr) state acts
  let newFuturesUndropped = applyStateToRewardFutureData state (oldFutureRewards `VB.snoc` RewardFutureData (aral ^. t) state as reward stateNext episodeEnd)
  let (materialisedFutures, newFutures) = splitMaterialisedFutures newFuturesUndropped
  let addNewRewardToExp currentExpSmthRew (RewardFutureData _ _ _ (Reward rew') _ _) = (1 - expSmthPsi) * currentExpSmthRew + expSmthPsi * rew'
      addNewRewardToExp _ _                                                          = error "unexpected RewardFutureData in runWorkerAction"
  newReplMem <- foldM addExperience replMem materialisedFutures
  return $! WorkerState wNr stateNext newReplMem newFutures (foldl' addNewRewardToExp rew materialisedFutures)
  where
    splitMaterialisedFutures fs =
      let (futures, finished) = VB.partition (isRewardFuture . view futureReward) fs
       in (VB.filter (not . isRewardEmpty . view futureReward) finished, futures)
    addExperience replMem' (RewardFutureData _ state' as' (Reward reward) stateNext episodeEnd) = do
      let (_, stateActs, stateNextActs) = mkStateActs aral state' stateNext
      liftIO $ addToReplayMemories (aral ^. settings . nStep) (stateActs, as', reward, stateNextActs, episodeEnd) replMem'
    addExperience _ _ = error "Unexpected Reward in calcExperience of runWorkerActions!"

-- | This function exectues all materialised rewards until a non-materialised reward is found, i.e. add a new experience
-- to the replay memory and then, select and learn from the experiences of the replay memory.
stepExecuteMaterialisedFutures ::
     forall m s as. (MonadIO m, NFData s, NFData as, Ord s, RewardFuture s, Eq as)
  => AgentType
  -> (Int, Bool, ARAL s as)
  -> RewardFutureData s
  -> m (Int, Bool, ARAL s as)
stepExecuteMaterialisedFutures _ (nr, True, aral) _ = return (nr, True, aral)
stepExecuteMaterialisedFutures agent (nr, _, aral) dt =
  case view futureReward dt of
    RewardEmpty     -> return (nr + 1, False, t %~ (+ 1) $ aral)
    RewardFuture {} -> return (nr, True, aral)
    Reward {}       -> (nr + 1, False, ) <$> execute agent aral dt


minMaxStates :: MVar ((Double, (s, AgentActionIndices)), (Double, (s, AgentActionIndices)))
minMaxStates = unsafePerformIO newEmptyMVar
{-# NOINLINE minMaxStates #-}

hasLocked :: String -> IO a -> IO a
hasLocked msg action =
  action `catches`
  [ Handler $ \exc@BlockedIndefinitelyOnMVar -> sayString ("[MVar]: " ++ msg) >> throwIO exc
  , Handler $ \exc@BlockedIndefinitelyOnSTM -> sayString ("[STM]: " ++ msg) >> throwIO exc
  ]

updateMinMax :: AgentType -> ARAL s as -> AgentActionIndices -> Calculation -> IO (Double, Double)
updateMinMax agTp aral as calc = do
  mMinMax <- hasLocked "updateMinMax tryReadMVar" $ tryReadMVar minMaxStates
  let minMax' =
        case mMinMax of
          Nothing                                -> ((V.minimum value, (aral ^. s, as)), (V.maximum value, (aral ^. s, as)))
          Just minMax@((minVal, _), (maxVal, _)) -> bimap (replaceIf V.minimum (V.minimum value < minVal)) (replaceIf V.maximum (V.maximum value > maxVal)) minMax
  empty <- isEmptyMVar minMaxStates
  when (empty || aral ^. t == 0) $ void $ hasLocked "updateMinMax putMVar" $ tryPutMVar minMaxStates minMax'
  when (fmap (bimap fst fst) mMinMax /= Just (bimap fst fst minMax')) $ hasLocked "updateMinMax modifyMVar 1" $ modifyMVar_ minMaxStates (const $ return minMax')
  when (aral ^. t `mod` 1000 == 0) $ do
    let ((_, (minS, minA)), (_, (maxS, maxA))) = minMax'
    AgentValue vMin <- valueFunction minS minA
    AgentValue vMax <- valueFunction maxS maxA
    let res = ((V.minimum vMin, (minS, minA)), (V.maximum vMax, (maxS, maxA)))
    hasLocked "updateMinMax modifyMVar 2" $ modifyMVar_ minMaxStates (const $ return res)
  return $ bimap fst fst minMax'
  where
    replaceIf reduce True _ = (reduce value, (aral ^. s, as))
    replaceIf _ False x     = x
    AgentValue value =
      fromMaybe (error "unexpected empty value in updateMinMax") $
      case aral ^. algorithm of
        AlgARAL {} -> getR1ValState' calc
        AlgDQN {}  -> getR1ValState' calc
        _          -> getVValState' calc
    valueFunction =
      case aral ^. algorithm of
        AlgARAL {} -> rValue agTp aral RBig
        AlgDQN {}  -> rValue agTp aral RBig
        _          -> vValue agTp aral


-- | Execute the given step, i.e. add a new experience to the replay memory and then, select and learn from the
-- experiences of the replay memory.
execute :: (MonadIO m, NFData s, NFData as, Ord s, RewardFuture s, Eq as) => AgentType -> ARAL s as -> RewardFutureData s -> m (ARAL s as)
execute agTp aral (RewardFutureData period state as (Reward reward) stateNext episodeEnd) = do
#ifdef DEBUG
  aral <- if isMainAgent agTp
          then do
            when (aral ^. t == 0) $ forM_ [fileDebugStateV, fileDebugStateW, fileDebugPsiWValues, fileDebugPsiVValues, fileDebugPsiWValues, fileDebugStateValuesNrStates] $ \f ->
              liftIO $ doesFileExist f >>= \x -> when x (removeFile f)
            writeDebugFiles aral
          else return aral
#endif
  (proxies', calc) <- P.insert aral agTp period state as reward stateNext episodeEnd (mkCalculation agTp aral) (aral ^. proxies)
  let lastVsLst = fromMaybe (VB.singleton $ toValue agents 0) (getLastVs' calc)
      strVAvg = map (show . avg) $ transpose $ VB.toList $ VB.map fromValue lastVsLst
      rhoVal = fromMaybe (toValue agents 0) (getRhoVal' calc)
      strRho = map show (fromValue rhoVal)
      strRhoSmth = show (aral ^. expSmoothedReward)
      strRhoOver = map (show . overEstimateRhoCalc aral) (fromValue rhoVal)
      strMinRho = map show $ fromValue $ fromMaybe (toValue agents 0) (getRhoMinimumVal' calc)
      strR0 = map show $ fromValue $ fromMaybe (toValue agents 0) (getR0ValState' calc)
      mCfg = aral ^? proxies . v . proxyNNConfig <|> aral ^? proxies . r1 . proxyNNConfig
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
      agents = aral ^. settings . independentAgents
      list = concatMap ("\t" <>)
      zero = toValue agents 0
  if isMainAgent agTp
  then do
    (minVal, maxVal) <- liftIO $ updateMinMax agTp aral (VB.map snd as) calc
    let minMaxValTxt = "\t" ++ show minVal ++ "\t" ++ show maxVal
    liftIO $ unless (period < 10) $
      appendFile fileStateValues (show period ++ list strRho ++ "\t" ++ strRhoSmth ++ list strRhoOver ++ list strMinRho ++ list strVAvg ++ list strR0 ++ list strR1 ++ list strR0Scaled ++ list strR1Scaled ++ minMaxValTxt ++ "\n" )
    let (eNr, eStart) = aral ^. episodeNrStart
        eLength = aral ^. t - eStart
    when (getEpisodeEnd calc) $ liftIO $ appendFile fileEpisodeLength (show eNr ++ "\t" ++ show eLength ++ "\n")
    liftIO $ appendFile fileReward (show period ++ "\t" ++ show reward ++ "\n")
    -- update values
    let setEpisode curEp
          | getEpisodeEnd calc = (eNr + 1, aral ^. t)
          | otherwise = curEp
    return $
      set psis (fromMaybe zero (getPsiValRho' calc), fromMaybe zero (getPsiValV' calc), fromMaybe zero (getPsiValW' calc)) $ set expSmoothedReward (getExpSmoothedReward' calc) $
      set lastVValues (fromMaybe mempty (getLastVs' calc)) $ set lastRewards (getLastRews' calc) $ set proxies proxies' $ set t (period + 1) $ over episodeNrStart setEpisode $ maybeFlipDropout aral
  else return $
       set psis (fromMaybe zero (getPsiValRho' calc), fromMaybe zero (getPsiValV' calc), fromMaybe zero (getPsiValW' calc)) $
       set proxies proxies' $ set expSmoothedReward (getExpSmoothedReward' calc) $ set t (period + 1) $ maybeFlipDropout aral

execute _ _ _ = error "Exectue on invalid data structure. This is a bug!"

-- | Flip the dropout active/inactive state.
maybeFlipDropout :: ARAL s as -> ARAL s as
maybeFlipDropout aral =
  case aral ^? proxies . v . proxyNNConfig <|> aral ^? proxies . r1 . proxyNNConfig of
    Just cfg@NNConfig {}
      | cfg ^. grenadeDropoutOnlyInactiveAfter <= 0 -> aral
      | aral ^. t == cfg ^. grenadeDropoutOnlyInactiveAfter -> setDropoutValue False aral
      | aral ^. t > cfg ^. grenadeDropoutOnlyInactiveAfter -> aral
      | aral ^. t `mod` cfg ^. grenadeDropoutFlipActivePeriod == 0 ->
        let occurance = aral ^. t `div` cfg ^. grenadeDropoutFlipActivePeriod
            value
              | even occurance = True
              | otherwise = False
         in setDropoutValue value aral
    _ -> aral
  where

setDropoutValue :: Bool -> ARAL s as -> ARAL s as
setDropoutValue val = overAllProxies (filtered (\p -> isGrenade p || isHasktorch p)) setDropout
  where
    setDropout (Grenade tar wor tp cfg act agents wel)                  = Grenade (runSettingsUpdate (NetworkSettings val) tar) (runSettingsUpdate (NetworkSettings val) wor) tp cfg act agents wel
    setDropout (Hasktorch tar wo tp cfg nrAct nrAg adam mlp wel nnActs) = Hasktorch (setDropoutMLP tar) (setDropoutMLP wo) tp cfg nrAct nrAg adam (setDropoutMLPSpec mlp) wel nnActs
    setDropoutMLPSpec x@MLPSpec {}                                              = x
    setDropoutMLPSpec (MLPSpecWDropoutLSTM mLoss lin act mDrI mDr mLSTM outAct) = MLPSpecWDropoutLSTM mLoss lin act ((val, ) . snd <$> mDrI) ((val, ) . snd <$> mDr) mLSTM outAct
    setDropoutMLP (MLP lays hAct hSpecAct mInpDrpOut mHidDrpOut mLSTM mOutAct mLossFun) = MLP lays hAct hSpecAct ((val, ) . snd <$> mInpDrpOut) ((val, ) . snd <$> mHidDrpOut) mLSTM mOutAct mLossFun


#ifdef DEBUG

stateFeatures :: MVar [a]
stateFeatures = unsafePerformIO $ newMVar mempty
{-# NOINLINE stateFeatures #-}

setStateFeatures :: (MonadIO m) => [a] -> m ()
setStateFeatures x = liftIO $ hasLocked "setStateFeatures" $ modifyMVar_ stateFeatures (return . const x)

getStateFeatures :: (MonadIO m) => m [a]
getStateFeatures = liftIO $ hasLocked "getStateFeatures" $ fromMaybe mempty <$> tryReadMVar stateFeatures


writeDebugFiles :: (MonadIO m, NFData s, NFData as, Ord s, Eq as, RewardFuture s) => ARAL s as -> m (ARAL s as)
writeDebugFiles aral = do
  let isDqn = isAlgDqn (aral ^. algorithm) || isAlgDqnAvgRewardAdjusted (aral ^. algorithm)
  let isAnn
        | isDqn = P.isNeuralNetwork (aral ^. proxies . r1)
        | otherwise = P.isNeuralNetwork (aral ^. proxies . v)
  let putStateFeatList aral xs
        | isAnn = aral
        | otherwise = setAllProxies proxyTable xs' aral
        where
          xs' = M.fromList $ zip (map (\xs -> (V.init xs, round (V.last xs))) xs) (repeat 0)
  aral' <-
    if aral ^. t > 0
      then return aral
      else do
        liftIO $ writeFile fileDebugStateV ""
        liftIO $ writeFile fileDebugStateVScaled ""
        liftIO $ writeFile fileDebugStateW ""
        liftIO $ writeFile fileDebugPsiVValues ""
        liftIO $ writeFile fileDebugPsiWValues ""
        aral' <-
          if isAnn
            then return aral
            else stepsM (setAllProxies (proxyNNConfig . replayMemoryMaxSize) 1000 $ set t 1 aral) debugStepsCount -- run steps to fill the table with (hopefully) all states
        let stateFeats
              | isDqn = getStateFeatList (aral' ^. proxies . r1)
              | otherwise = getStateFeatList (aral' ^. proxies . v)
        setStateFeatures stateFeats
        liftIO $ writeFile fileDebugStateValuesNrStates (show $ length stateFeats)
        liftIO $ forM_ [fileDebugStateV, fileDebugStateVScaled, fileDebugStateW, fileDebugPsiVValues, fileDebugPsiWValues] $
          flip writeFile ("Period\t" <> mkListStrAg (shorten . printStateFeat) stateFeats <> "\n")
        if isNeuralNetwork (aral ^. proxies . v)
          then return aral
          else do
            liftIO $ putStrLn $ "[DEBUG INFERRED NUMBER OF STATES]: " <> show (length stateFeats)
            return $ aral -- putStateFeatList aral stateFeats
  stateFeatsLoaded <- getStateFeatures
  let stateFeats
        | not (null stateFeatsLoaded) = stateFeatsLoaded
        | isDqn = getStateFeatList (aral' ^. proxies . r1)
        | otherwise = getStateFeatList (aral' ^. proxies . v)
  when ((aral' ^. t `mod` debugPrintCount) == 0) $ do
    let splitIdx =
          V.length (head stateFeats) - agents
    stateValuesV <- mapM (\xs ->
                            let (st,as) = V.splitAt splitIdx xs in
                            if isDqn then rValueWith Worker aral' RBig st (VB.map round $ V.convert as) else vValueWith Worker aral' st (VB.map round $ V.convert as)) stateFeats
    stateValuesVScaled <- mapM (\xs ->
                                  let (st,as) = V.splitAt splitIdx xs in
                                  if isDqn then rValueNoUnscaleWith Worker aral' RBig st (VB.map round $ V.convert as) else vValueNoUnscaleWith Worker aral' st (VB.map round $ V.convert as)) stateFeats
    stateValuesW <- mapM (\xs -> let (st,as) = V.splitAt splitIdx xs in if isDqn then return 0 else wValueFeat aral' st (VB.map round $ V.convert as)) stateFeats
    liftIO $ appendFile fileDebugStateV (show (aral' ^. t) <> "\t" <> mkListStrV show stateValuesV <> "\n")
    liftIO $ appendFile fileDebugStateVScaled (show (aral' ^. t) <> "\t" <> mkListStrV show stateValuesVScaled <> "\n")
    when (isAlgAral (aral ^. algorithm)) $ do
      liftIO $ appendFile fileDebugStateW (show (aral' ^. t) <> "\t" <> mkListStrV show stateValuesW <> "\n")
      psiVValues <- mapM (\xs -> let (st,as) = V.splitAt splitIdx xs in psiVFeat aral' st (VB.map round $ V.convert as)) stateFeats
      liftIO $ appendFile fileDebugPsiVValues (show (aral' ^. t) <> "\t" <> mkListStrV show psiVValues <> "\n")
      psiWValues <- mapM (\xs -> let (st,as) = V.splitAt splitIdx xs in psiWFeat aral' st (VB.map round $ V.convert xs)) stateFeats
      liftIO $ appendFile fileDebugPsiWValues (show (aral' ^. t) <> "\t" <> mkListStrV show psiWValues <> "\n")
  return aral'
  where
    getStateFeatList :: Proxy -> [V.Vector Double]
    getStateFeatList Scalar {} = []
    getStateFeatList (Table t _ _) = -- map fst (M.keys t)
      map (\(xs, y) -> xs V.++ V.replicate agents (fromIntegral y)) (M.keys t)
    getStateFeatList nn = concatMap (\xs -> map (\a -> xs V.++ V.replicate agents (fromIntegral $ actIdx a)) acts) (nn ^. proxyNNConfig . prettyPrintElems)
    actIdx a = fromMaybe (-1) (elemIndex a acts)
    acts = VB.toList $ aral ^. actionList
    agents = aral ^. settings . independentAgents
    mkListStrAg :: (a -> String) -> [a] -> String
    mkListStrAg f = intercalate "\t" . concatMap (\x -> map (\nr -> f x <> "-Ag" <> show nr) [1..agents])
    mkListStrV :: (Double -> String) -> [Value] -> String
    mkListStrV f = intercalate "\t" . concatMap (map f . fromValue)
    shorten xs | length xs > 60 = "..." <> drop (length xs - 60) xs
               | otherwise = xs
    printStateFeat :: StateFeatures -> String
    printStateFeat xs = "[" <> intercalate "," (map (printf "%.2f") (V.toList xs)) <> "]"
    psiVFeat aral stateFeat aNr = P.lookupProxy (aral ^. t) Worker (stateFeat, aNr) (aral ^. proxies . psiV)
    psiWFeat aral stateFeat aNr = P.lookupProxy (aral ^. t) Worker (stateFeat, aNr) (aral ^. proxies . psiW)

debugStepsCount :: Integer
debugStepsCount = 8000

debugPrintCount :: Int
debugPrintCount = 100

#endif
