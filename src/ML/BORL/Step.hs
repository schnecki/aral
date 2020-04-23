{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Strict              #-}
{-# LANGUAGE TupleSections       #-}
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

#ifdef DEBUG
import           Control.Concurrent.MVar
import           System.IO.Unsafe                   (unsafePerformIO)
#endif
import           Control.Applicative                ((<|>))
import           Control.Arrow                      ((&&&), (***))
import           Control.DeepSeq
import           Control.Lens
import           Control.Monad
import           Control.Monad.IO.Class             (liftIO)
import           Control.Monad.IO.Class             (MonadIO, liftIO)
import           Control.Parallel.Strategies        hiding (r0)
import           Data.Either                        (isLeft)
import           Data.Function                      (on)
import           Data.List                          (find, groupBy, intercalate, maximumBy,
                                                     partition, sortBy, transpose)
import qualified Data.Map.Strict                    as M
import           Data.Maybe                         (fromMaybe, isJust)
import           Data.Ord
import           Data.Serialize
import qualified Data.Vector                        as VB
import qualified Data.Vector.Storable               as V
import           GHC.Generics
import           System.Directory
import           System.IO
import           System.Random
import           Text.Printf

import           ML.BORL.Action
import           ML.BORL.Algorithm
import           ML.BORL.Calculation
import           ML.BORL.Fork
import           ML.BORL.NeuralNetwork.NNConfig
import           ML.BORL.NeuralNetwork.ReplayMemory
import           ML.BORL.NeuralNetwork.Scaling
import           ML.BORL.Parameters
import           ML.BORL.Properties
import           ML.BORL.Proxy                      as P
import           ML.BORL.Reward
import           ML.BORL.SaveRestore
import           ML.BORL.Serialisable
import           ML.BORL.Settings
import           ML.BORL.Type
import           ML.BORL.Types
import           ML.BORL.Workers.Type
import           ML.BORL.Workers.Type


import           Debug.Trace

fileDebugStateV :: FilePath
fileDebugStateV = "stateVAllStates"

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

fileReward :: FilePath
fileReward = "reward"

fileEpisodeLength :: FilePath
fileEpisodeLength = "episodeLength"


steps :: (NFData s, Ord s, RewardFuture s) => BORL s -> Integer -> IO (BORL s)
steps !borl nr =
  fmap force $!
  case find isTensorflow (allProxies $ borl ^. proxies) of
    Nothing -> runMonadBorlIO $ foldM (\b _ -> nextAction b >>= stepExecute b) borl [0 .. nr - 1]
    Just _ ->
      runMonadBorlTF $ do
        void $ restoreTensorflowModels True borl
        borl' <- foldM (\b _ -> nextAction b >>= stepExecute b) borl [0 .. nr - 1]
        saveTensorflowModels borl'


step :: (NFData s, Ord s, RewardFuture s) => BORL s -> IO (BORL s)
step !borl =
  fmap force $!
  case find isTensorflow (allProxies $ borl ^. proxies) of
    Nothing -> nextAction borl >>= stepExecute borl
    Just _ ->
      runMonadBorlTF $ do
        void $ restoreTensorflowModels True borl
        borl' <- nextAction borl >>= stepExecute borl
        saveTensorflowModels borl'

-- | This keeps the Tensorflow session alive. For non-Tensorflow BORL data structures this is equal to step.
stepM :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> m (BORL s)
stepM !borl = nextAction borl >>= stepExecute borl >>= \(b@BORL{}) -> return (force b)

-- | This keeps the Tensorflow session alive. For non-Tensorflow BORL data structures this is equal to steps, but forces
-- evaluation of the data structure every 1000 periods.
stepsM :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> Integer -> m (BORL s)
stepsM borl nr = do
  borl' <- foldM (\b _ -> stepM b) borl [1 .. min maxNr nr]
  if nr > maxNr
    then stepsM borl' (nr - maxNr)
    else return borl'
  where maxNr = 1000

stepExecute :: forall m s . (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> NextActions s -> m (BORL s)
stepExecute borl ((randomAction, (aNr, Action action _)), workerActions) = do
  let state = borl ^. s
      period = borl ^. t + length (borl ^. futureRewards)
  -- File IO Operations
  when (period == 0) $ do
    liftIO $ writeFile fileStateValues "Period\tRho\tMinRho\tVAvg\tR0\tR1\n"
    liftIO $ writeFile fileEpisodeLength "Episode\tEpisodeLength\n"
    liftIO $ writeFile fileReward "Period\tReward\n"
  workerRefs <- liftIO $ runWorkerActions (set t period borl) workerActions
  (reward, stateNext, episodeEnd) <- liftIO $ action MainAgent state
  let borl' = over futureRewards (applyStateToRewardFutureData state . (++ [RewardFutureData period state aNr randomAction reward stateNext episodeEnd])) borl
  (dropLen, _, borlNew) <- foldM (stepExecuteMaterialisedFutures MainAgent) (0, False, borl') (borl' ^. futureRewards)
  newWorkerStates <- liftIO $ collectWorkers workerRefs
  let updateFromWorkers = (borl ^. t) `mod` borl ^. settings . workersUpdateInterval == 0
  let allWorkerProxies = newWorkerStates ^.. traversed . workerProxies . _Right
      allWorkerExpSmoothedReward = newWorkerStates ^.. traversed . workerExpSmthReward
      avgExpSmthReward = avg $ (borl ^. psis . _1) : allWorkerExpSmoothedReward
      avg xs = sum xs / fromIntegral (length xs)
      newProxies
        | updateFromWorkers && (null newWorkerStates || isLeft (head newWorkerStates ^. workerProxies)) = borlNew ^. proxies
        | updateFromWorkers =
          -- choose workers by exponentially smoothed average reward (only those within 10% of best, except if their minimum exploration rate is reached)
          mergeProxiesInto (borlNew ^. algorithm) (borlNew ^. proxies) $ map snd $ take
          (max 1 $ round $ fromIntegral (length allWorkerProxies) / (4 :: Float)) $ sortBy (comparing (Down . fst)) $ zip allWorkerExpSmoothedReward allWorkerProxies
          -- mergeProxiesInto (borlNew ^. algorithm) (borlNew ^. proxies) allWorkerProxies
        | otherwise = borlNew ^. proxies
      setNewWorkerProxies (Right oldProxies) = Right $ set replayMemory (oldProxies ^. replayMemory) newProxies
      setNewWorkerProxies (Left experience) = Left experience
      newWorkers
        | updateFromWorkers = map (set workerExpSmthReward avgExpSmthReward . over workerProxies setNewWorkerProxies) newWorkerStates
        | otherwise = newWorkerStates
  return $ set workers newWorkers $ set proxies newProxies $ over futureRewards (drop dropLen) $ set s stateNext borlNew
  where
    collectWorkers (Left xs)      = return xs
    collectWorkers (Right ioRefs) = mapM collectForkResult ioRefs


-- | This functions takes one step for all workers, and returns the new worker replay memories and future reward data
-- lists.
runWorkerActions :: (NFData s, Ord s, RewardFuture s) => BORL s -> [WorkerActionChoice s] -> IO (Either (Workers s) [IORef (ThreadState (WorkerState s))])
runWorkerActions _ [] = return (Left [])
runWorkerActions borl _ | borl ^. settings . disableAllLearning = return (Left $ borl ^. workers)
runWorkerActions borl acts = Right <$> zipWithM (\act worker -> doFork' $ runWorkerAction borl worker act) acts (borl ^. workers)
  where
    doFork'
      | borl ^. settings . useForking = doFork
      | otherwise = doForkFake

-- | Apply the given state to a list of future reward data.
applyStateToRewardFutureData :: State s -> [RewardFutureData s] -> [RewardFutureData s]
applyStateToRewardFutureData state = map (over futureReward applyToReward)
  where
    applyToReward (RewardFuture storage) = applyState storage state
    applyToReward r                      = r

-- | Run one worker.
runWorkerAction :: (NFData s, RewardFuture s, Ord s) => BORL s -> WorkerState s -> WorkerActionChoice s -> IO (WorkerState s)
runWorkerAction borl (WorkerState wNr state (Right oldPx) oldFutureRewards rew) (randomAction, (aNr, Action action _)) = do
  let workerType = WorkerAgent wNr
  (reward, stateNext, episodeEnd) <- liftIO $ action workerType state
  let newFuturesUndropped = applyStateToRewardFutureData state (oldFutureRewards ++ [RewardFutureData (borl ^. t) state aNr randomAction reward stateNext episodeEnd])
  let borlAdapted = set proxies oldPx $ set workers [] $ set s state $ set futureRewards [] $ set (psis . _1) rew borl
  (dropLen, _, borlNew) <- foldM (stepExecuteMaterialisedFutures workerType) (0, False, borlAdapted) newFuturesUndropped
  return $ force $ WorkerState wNr stateNext (Right $ borlNew ^. proxies) (drop dropLen $ borlNew ^. futureRewards) (borlNew ^. psis . _1)
runWorkerAction borl (WorkerState wNr state (Left replMem) oldFutureRewards rew) (randomAction, (aNr, Action action _)) = do
  (reward, stateNext, episodeEnd) <- liftIO $ action (WorkerAgent wNr) state
  let newFuturesUndropped = applyStateToRewardFutureData state (oldFutureRewards ++ [RewardFutureData (borl ^. t) state aNr randomAction reward stateNext episodeEnd])
  let (materialisedFutures, newFutures) = splitMaterialisedFutures newFuturesUndropped
  newReplMem <- foldM addExperience replMem materialisedFutures
  return $ force $ WorkerState wNr stateNext (Left newReplMem) newFutures rew
  where
    splitMaterialisedFutures fs =
      let xs = takeWhile (not . isRewardFuture . view futureReward) fs
       in (filter (not . isRewardEmpty . view futureReward) xs, drop (length xs) fs)
    addExperience replMem (RewardFutureData _ state aNr randomAction (Reward reward) stateNext episodeEnd) = do
      let (_, stateActs, stateNextActs) = mkStateActs borl state stateNext
      liftIO $ addToReplayMemories (stateActs, aNr, randomAction, reward, stateNextActs, episodeEnd) replMem
    addExperience _ _ = error "Unexpected Reward in calcExperience of runWorkerActions!"

-- | This function exectues all materialised rewards until a non-materialised reward is found, i.e. add a new experience
-- to the replay memory and then, select and learn from the experiences of the replay memory.
stepExecuteMaterialisedFutures ::
     forall m s. (MonadBorl' m, NFData s, Ord s, RewardFuture s)
  => AgentType
  -> (Int, Bool, BORL s)
  -> RewardFutureData s
  -> m (Int, Bool, BORL s)
stepExecuteMaterialisedFutures _ (nr, True, borl) _ = return (nr, True, borl)
stepExecuteMaterialisedFutures agent (nr, _, borl) dt =
  case view futureReward dt of
    RewardEmpty     -> return (nr, False, borl)
    RewardFuture {} -> return (nr, True, borl)
    Reward {}       -> (nr+1, False, ) <$> execute borl agent dt

-- | Execute the given step, i.e. add a new experience to the replay memory and then, select and learn from the
-- experiences of the replay memory.
execute :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> AgentType -> RewardFutureData s -> m (BORL s)
execute borl agent (RewardFutureData period state aNr randomAction (Reward reward) stateNext episodeEnd) = do
#ifdef DEBUG
  borl <- if isMainAgent agent
          then do
            when (borl ^. t == 0) $ forM_ [fileDebugStateV, fileDebugStateW, fileDebugPsiWValues, fileDebugPsiVValues, fileDebugPsiWValues, fileDebugStateValuesNrStates] $ \f ->
              liftIO $ doesFileExist f >>= \x -> when x (removeFile f)
            writeDebugFiles borl
          else return borl
#endif
  (proxies', calc) <- P.insert borl agent period state aNr randomAction reward stateNext episodeEnd (mkCalculation borl) (borl ^. proxies)
  let lastVsLst = fromMaybe [0] (getLastVs' calc)
  let strRho = show (fromMaybe 0 (getRhoVal' calc))
      strMinV = show (fromMaybe 0 (getRhoMinimumVal' calc))
      strVAvg = show (avg lastVsLst)
      strR0 = show $ fromMaybe 0 (getR0ValState' calc)
      strR1 = show $ fromMaybe 0 (getR1ValState' calc)
      avg xs = sum xs / fromIntegral (length xs)
  if isMainAgent agent
  then do
    liftIO $ appendFile fileStateValues (show period ++ "\t" ++ strRho ++ "\t" ++ strMinV ++ "\t" ++ strVAvg ++ "\t" ++ strR0 ++ "\t" ++ strR1 ++ "\n")
    let (eNr, eStart) = borl ^. episodeNrStart
        eLength = borl ^. t - eStart
    when (getEpisodeEnd calc) $ liftIO $ appendFile fileEpisodeLength (show eNr ++ "\t" ++ show eLength ++ "\n")
    liftIO $ appendFile fileReward (show period ++ "\t" ++ show reward ++ "\n")
    -- update values
    let setEpisode curEp
          | getEpisodeEnd calc = (eNr + 1, borl ^. t)
          | otherwise = curEp
    return $
      set psis (fromMaybe 0 (getPsiValRho' calc), fromMaybe 0 (getPsiValV' calc), fromMaybe 0 (getPsiValW' calc)) $
      set lastVValues (fromMaybe [] (getLastVs' calc)) $ set lastRewards (getLastRews' calc) $ set proxies proxies' $ set t (period + 1) $ over episodeNrStart setEpisode borl
  else return $
       set psis (fromMaybe 0 (getPsiValRho' calc), fromMaybe 0 (getPsiValV' calc), fromMaybe 0 (getPsiValW' calc)) $
       set proxies proxies' $ set t (period + 1) borl

execute _ _ _ = error "Exectue on invalid data structure. This is a bug!"


#ifdef DEBUG

stateFeatures :: MVar [a]
stateFeatures = unsafePerformIO $ newMVar mempty
{-# NOINLINE stateFeatures #-}

setStateFeatures :: (MonadIO m) => [a] -> m ()
setStateFeatures x = liftIO $ modifyMVar_ stateFeatures (return . const x)

getStateFeatures :: (MonadIO m) => m [a]
getStateFeatures = liftIO $ fromMaybe mempty <$> tryReadMVar stateFeatures


writeDebugFiles :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> m (BORL s)
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
        liftIO $ forM_ [fileDebugStateV, fileDebugStateW, fileDebugPsiVValues, fileDebugPsiWValues] $ flip writeFile ("Period\t" <> mkListStr (shorten . printStateFeat) stateFeats <> "\n")
        if isNeuralNetwork (borl ^. proxies . v)
          then return borl
          else do
            liftIO $ putStrLn $ "[DEBUG INFERRED NUMBER OF STATES]: " <> show (length stateFeats)
            return $ putStateFeatList borl stateFeats
  let stateFeats
        | isDqn = getStateFeatList (borl' ^. proxies . r1)
        | otherwise = getStateFeatList (borl' ^. proxies . v)
      isTf
        | isDqn && isTensorflow (borl' ^. proxies . r1) = True
        | isTensorflow (borl' ^. proxies . v) = True
        | otherwise = False
  stateFeats <- getStateFeatures
  when ((borl' ^. t `mod` debugPrintCount) == 0) $ do
    stateValuesV <- mapM (\xs -> if isDqn then rValueWith Worker borl' RBig (V.init xs) (round $ V.last xs) else vValueWith Worker borl' (V.init xs) (round $ V.last xs)) stateFeats
    stateValuesW <- mapM (\xs -> if isDqn then return 0 else wValueFeat borl' (V.init xs) (round $ V.last xs)) stateFeats
    liftIO $ appendFile fileDebugStateV (show (borl' ^. t) <> "\t" <> mkListStr show stateValuesV <> "\n")
    when (isAlgBorl (borl ^. algorithm)) $ do
      liftIO $ appendFile fileDebugStateW (show (borl' ^. t) <> "\t" <> mkListStr show stateValuesW <> "\n")
      psiVValues <- mapM (\xs -> psiVFeat borl' (V.init xs) (round $ V.last xs)) stateFeats
      liftIO $ appendFile fileDebugPsiVValues (show (borl' ^. t) <> "\t" <> mkListStr show psiVValues <> "\n")
      psiWValues <- mapM (\xs -> psiWFeat borl' (V.init xs) (round $ V.last xs)) stateFeats
      liftIO $ appendFile fileDebugPsiWValues (show (borl' ^. t) <> "\t" <> mkListStr show psiWValues <> "\n")
  return borl'
  where
    getStateFeatList Scalar {} = []
    getStateFeatList (Table t _) = map (\(xs, y) -> V.snoc xs (fromIntegral y)) (M.keys t)
    getStateFeatList nn = concatMap (\xs -> map (\(idx, _) -> V.snoc xs (fromIntegral idx)) acts) (nn ^. proxyNNConfig . prettyPrintElems)
    acts = VB.toList $ borl ^. actionList
    mkListStr :: (a -> String) -> [a] -> String
    mkListStr f = intercalate "\t" . map f
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
