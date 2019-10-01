{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE ScopedTypeVariables #-}
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

import           ML.BORL.Action
import           ML.BORL.Algorithm
import           ML.BORL.Calculation
import           ML.BORL.Fork
import           ML.BORL.NeuralNetwork.NNConfig
import           ML.BORL.NeuralNetwork.Tensorflow (buildTensorflowModel,
                                                   restoreModelWithLastIO,
                                                   saveModelWithLastIO)
import           ML.BORL.Parameters
import           ML.BORL.Properties
import           ML.BORL.Proxy                    as P
import           ML.BORL.Reward
import           ML.BORL.SaveRestore
import           ML.BORL.Serialisable
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Applicative              ((<|>))
import           Control.Arrow                    ((&&&), (***))
import           Control.DeepSeq                  (NFData, force)
import           Control.Lens
import           Control.Monad
import           Control.Monad.IO.Class           (MonadIO, liftIO)
import           Control.Parallel.Strategies      hiding (r0)
import           Data.Function                    (on)
import           Data.List                        (find, groupBy, intercalate, partition,
                                                   sortBy)
import qualified Data.Map.Strict                  as M
import           Data.Maybe                       (fromMaybe, isJust)
import           System.Directory
import           System.IO
import           System.Random
import           Text.Printf

import           Debug.Trace

fileDebugStateValues :: FilePath
fileDebugStateValues = "stateValuesAllStates"

fileDebugPsiVValues :: FilePath
fileDebugPsiVValues = "statePsiVAllStates"

fileDebugPsiWValues :: FilePath
fileDebugPsiWValues = "statePsiWAllStates"


fileDebugStateValuesNrStates :: FilePath
fileDebugStateValuesNrStates = "stateValuesAllStatesCount"


fileStateValues :: FilePath
fileStateValues = "stateValues"

fileReward :: FilePath
fileReward = "reward"

fileEpisodeLength :: FilePath
fileEpisodeLength = "episodeLength"


steps :: (NFData s, Ord s, RewardFuture s) => BORL s -> Integer -> IO (BORL s)
steps (force -> borl) nr =
  case find isTensorflow (allProxies $ borl ^. proxies) of
    Nothing -> runMonadBorlIO $ force <$> foldM (\b _ -> nextAction (force b) >>= fmap force . stepExecute) borl [0 .. nr - 1]
    Just _ ->
      runMonadBorlTF $ do
        void $ restoreTensorflowModels True borl
        !borl' <- foldM (\b _ -> nextAction (force b) >>= fmap force . stepExecute) borl [0 .. nr - 1]
        force <$> saveTensorflowModels borl'


step :: (NFData s, Ord s, RewardFuture s) => BORL s -> IO (BORL s)
step (force -> borl) =
  case find isTensorflow (allProxies $ borl ^. proxies) of
    Nothing -> nextAction borl >>= stepExecute
    Just _ ->
      runMonadBorlTF $ do
        void $ restoreTensorflowModels True borl
        !borl' <- nextAction borl >>= stepExecute
        force <$> saveTensorflowModels borl'

-- | This keeps the Tensorflow session alive. For non-Tensorflow BORL data structures this is equal to step.
stepM :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> m (BORL s)
stepM (force -> borl) = nextAction borl >>= fmap force . stepExecute

-- | This keeps the Tensorflow session alive. For non-Tensorflow BORL data structures this is equal to steps, but forces
-- evaluation of the data structure every 1000 periods.
stepsM :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> Integer -> m (BORL s)
stepsM (force -> borl) nr = do
  !borl' <- force <$> foldM (\b _ -> nextAction b >>= stepExecute) borl [1 .. min maxNr nr]
  if nr > maxNr
    then stepsM borl' (nr - maxNr)
    else return borl'
  where maxNr = 1000


-- | This function chooses the next action from the current state s and all possible actions.
nextAction :: (MonadBorl' m) => BORL s -> m (BORL s, Bool, ActionIndexed s)
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
             AlgBORL _ _ _ _ decideVPlusPsi _ -> do
               bestRho <-
                 if isUnichain borl
                   then return as
                   else do
                     rhoVals <- mapM (rhoValue borl state . fst) as
                     return $ map snd $ headRho $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (zip rhoVals as)
               bestV <-
                 do vVals <- mapM (vValue decideVPlusPsi borl state . fst) bestRho
                    return $ map snd $ headV $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (zip vVals bestRho)
               bestE <-
                 do eVals <- mapM (eValue borl state . fst) bestV
                    rhoVal <- rhoValue borl state (fst $ head bestRho)
                    vVal <- vValue decideVPlusPsi borl state (fst $ head bestV) -- all a have the same V(s,a) value!
                    r0Values <- mapM (rValue borl RSmall state . fst) bestV
                    let rhoPlusV = rhoVal / (1-gamma0) + vVal
                        (posErr,negErr) = (map snd *** map snd) $ partition ((rhoPlusV<) . fst) (zip r0Values (zip eVals bestV))
                    return $ map snd $ head $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (if null posErr then negErr else posErr)
               if length bestV == 1
                 then return (borl, False, head bestV)
                 else if length bestE > 1
                        then do
                          r <- liftSimple $ randomRIO (0, length bestE - 1)
                          return (borl, False, bestE !! r)
                        else return (borl, False, headE bestE)
             AlgBORLVOnly {} -> singleValueNextAction (vValue False borl state . fst)
             AlgDQN {} -> singleValueNextAction (rValue borl RBig state . fst)
             AlgDQNAvgRewardFree {} -> singleValueNextAction (rValue borl RBig state . fst)
  where
    headRho []    = error "head: empty input data in nextAction on Rho value"
    headRho (x:_) = x
    headV []    = error "head: empty input data in nextAction on V value"
    headV (x:_) = x
    headE []    = error "head: empty input data in nextAction on E Value"
    headE (x:_) = x
    headDqn []    = error "head: empty input data in nextAction on Dqn Value"
    headDqn (x:_) = x
    gamma0 = case borl ^. algorithm of
      AlgBORL g0 _ _ _ _ _     -> g0
      AlgDQN g0                -> g0
      AlgDQNAvgRewardFree g0 _ -> g0
      AlgBORLVOnly _ _         -> 1
    params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
    eps = params' ^. epsilon
    explore = params' ^. exploration
    state = borl ^. s
    as = actionsIndexed borl state
    epsCompare = epsCompareWith eps
    singleValueNextAction f = do
      rValues <- mapM f as
      let bestR = sortBy (epsCompare compare `on` fst) (zip rValues as)
               -- liftSimple $ putStrLn ("bestR: " ++ show bestR)
      return (borl, False, snd $ headDqn bestR)

epsCompareWith :: (Ord t, Num t) => t -> (t -> t -> p) -> t -> t -> p
epsCompareWith eps f x y
  | abs (x - y) <= eps = f 0 0
  | otherwise = y `f` x


stepExecute :: forall m s . (MonadBorl' m, NFData s, Ord s, RewardFuture s) => (BORL s, Bool, ActionIndexed s) -> m (BORL s)
stepExecute (borl, randomAction, (aNr, Action action _)) = do
  let state = borl ^. s
      period = borl ^. t + length (borl ^. futureRewards)
  (reward, stateNext, episodeEnd) <- liftSimple $ action state
  let applyToReward r@(RewardFuture storage) = applyState storage state
      applyToReward r                        = r
      updateFutures = map (over futureReward applyToReward)
  let borl' = over futureRewards (updateFutures . (++ [RewardFutureData period state aNr randomAction reward stateNext (episodeEnd)])) borl
  (dropLen, _, borlNew) <- foldM stepExecuteMaterialisedFutures (0, False, borl') (borl' ^. futureRewards)
  return $ force $ over futureRewards (drop dropLen) $ set s stateNext borlNew


stepExecuteMaterialisedFutures ::
     forall m s. (MonadBorl' m, NFData s, Ord s, RewardFuture s)
  => (Int, Bool, BORL s)
  -> RewardFutureData s
  -> m (Int, Bool, BORL s)
stepExecuteMaterialisedFutures (nr, True, borl) _ = return (nr, True, borl)
stepExecuteMaterialisedFutures (nr, _, borl) dt =
  case view futureReward dt of
    RewardEmpty     -> return (nr, False, borl)
    RewardFuture {} -> return (nr, True, borl)
    Reward {}       -> (nr+1, False, ) <$> execute borl dt


execute :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> RewardFutureData s -> m (BORL s)
execute borl (RewardFutureData period state aNr randomAction (Reward reward) stateNext episodeEnd) = do
#ifdef DEBUG
  when (borl ^. t == 0) $ forM_ [fileDebugPsiWValues, fileDebugPsiVValues, fileDebugPsiWValues, fileDebugStateValuesNrStates] $ \f ->
    liftSimple $ doesFileExist f >>= \x -> when x (removeFile f)
  borl <- liftSimple $ writeDebugFiles borl
#endif
  (proxies', calc) <- P.insert borl period state aNr randomAction reward stateNext episodeEnd (mkCalculation borl) (borl ^. proxies)
  let lastVsLst = fromMaybe [0] (getLastVs' calc)
  -- File IO Operations
  when (period == 0) $ do
    liftSimple $ writeFile fileStateValues "Period\tRho\tMinRho\tVAvg\tR0\tR1\n"
    liftSimple $ writeFile fileEpisodeLength "Episode\tEpisodeLength\n"
    liftSimple $ writeFile fileReward "Period\tReward\n"
  let strRho = show (fromMaybe 0 (getRhoVal' calc))
      strMinV = show (fromMaybe 0 (getRhoMinimumVal' calc))
      strVAvg = show (avg lastVsLst)
      strR0 = show $ fromMaybe 0 (getR0ValState' calc)
      strR1 = show $ getR1ValState' calc
      divideAfterGrowth :: BORL s -> BORL s
      divideAfterGrowth borl =
        case borl ^. algorithm of
          AlgBORL _ _ _ (DivideValuesAfterGrowth nr maxPeriod) _ _
            | period > maxPeriod -> borl
            | length lastVsLst == nr && endOfIncreasedStateValues ->
              trace ("multiply in period " ++ show period ++ " by " ++ show val) $
              foldl (\q f -> over (proxies . f) (multiplyProxy val) q) (set phase SteadyStateValues borl) [psiV, v, w]
            where endOfIncreasedStateValues =
                    borl ^. phase == IncreasingStateValues && 2000 * (avg (take (nr `div` 2) lastVsLst) - avg (drop (nr `div` 2) lastVsLst)) / fromIntegral nr < 0.00
                  val = 0.2 / (sum lastVsLst / fromIntegral nr)
          _ -> borl
      setCurrentPhase :: BORL s -> BORL s
      setCurrentPhase borl =
        case borl ^. algorithm of
          AlgBORL _ _ _ (DivideValuesAfterGrowth nr _) _ _
            | length lastVsLst == nr && increasingStateValue -> trace ("period: " ++ show period) $ trace (show IncreasingStateValues) $ set phase IncreasingStateValues borl
            where increasingStateValue =
                    borl ^. phase /= IncreasingStateValues && 2000 * (avg (take (nr `div` 2) lastVsLst) - avg (drop (nr `div` 2) lastVsLst)) / fromIntegral nr > 0.20
          _ -> borl
      avg xs = sum xs / fromIntegral (length xs)
  liftSimple $ appendFile fileStateValues (show period ++ "\t" ++ strRho ++ "\t" ++ strMinV ++ "\t" ++ strVAvg ++ "\t" ++ strR0 ++ "\t" ++ strR1 ++ "\n")
  let (eNr, eStart) = borl ^. episodeNrStart
      eLength = borl ^. t - eStart
  when (getEpisodeEnd calc) $ liftSimple $ appendFile fileEpisodeLength (show eNr ++ "\t" ++ show eLength ++ "\n")
  liftSimple $ appendFile fileReward (show period ++ "\t" ++ show reward ++ "\n")
  -- update values
  let setEpisode curEp
        | getEpisodeEnd calc = (eNr + 1, borl ^. t)
        | otherwise = curEp
  return $
    setCurrentPhase $
    divideAfterGrowth $
    set psis (fromMaybe 0 (getPsiValRho' calc), fromMaybe 0 (getPsiValV' calc), fromMaybe 0 (getPsiValW' calc)) $
    set lastVValues (fromMaybe [] (getLastVs' calc)) $ set lastRewards (getLastRews' calc) $ set proxies proxies' $ set t (period + 1) $ over episodeNrStart setEpisode borl
execute _ _ = error "Exectue on invalid data structure. This is a bug!"

writeDebugFiles :: (NFData s, Ord s, RewardFuture s) => BORL s -> IO (BORL s)
writeDebugFiles borl = do
  borl' <-
    if borl ^. t > 0
      then return borl
      else do
        writeFile fileDebugStateValues ""
        writeFile fileDebugPsiVValues ""
        writeFile fileDebugPsiWValues ""
        writeFile fileDebugStateValuesNrStates "-1"
        borl' <-
          if isNeuralNetwork (borl ^. proxies . v)
            then return borl
            else steps (set t 1 borl) debugStepsCount -- run steps to fill the table with (hopefully) all states
        let stateFeats = getStateFeatList (borl' ^. proxies . v)
        forM_ [fileDebugStateValues, fileDebugPsiVValues, fileDebugPsiWValues] $ flip writeFile ("Period\t" <> mkListStr show stateFeats <> "\n")
        writeFile fileDebugStateValuesNrStates (show $ length stateFeats)
        if isNeuralNetwork (borl ^. proxies . v)
          then return borl
          else do
            putStrLn $ "[DEBUG INFERRED NUMBER OF STATES]: " <> show (length stateFeats)
            return $ putStateFeatList borl stateFeats
  let stateFeats = getStateFeatList (borl' ^. proxies . v)
  len <- read <$> readFile fileDebugStateValuesNrStates
  when (len >= 0 && len /= length stateFeats) $ error $ "Number of states to write to debug file changed from " <> show len <> " to " <> show (length stateFeats) <>
    ". Increase debugStepsCount count in Step.hs!"
  stateValues <- liftSimple $ mapM (\xs -> vValueFeat False borl' (init xs) (round $ last xs)) stateFeats
  psiVValues <- liftSimple $ mapM (\xs -> psiVFeat borl' (init xs) (round $ last xs)) stateFeats
  psiWValues <- liftSimple $ mapM (\xs -> psiWFeat borl' (init xs) (round $ last xs)) stateFeats
  appendFile fileDebugStateValues (show (borl' ^. t) <> "\t" <> mkListStr show stateValues <> "\n")
  appendFile fileDebugPsiVValues (show (borl' ^. t) <> "\t" <> mkListStr show psiVValues <> "\n")
  appendFile fileDebugPsiWValues (show (borl' ^. t) <> "\t" <> mkListStr show psiWValues <> "\n")
  return borl'
  where
    getStateFeatList Scalar {}   = []
    getStateFeatList (Table t _) = map (\(xs, y) -> xs ++ [fromIntegral y]) (M.keys t)
    getStateFeatList nn          = nn ^. proxyNNConfig . prettyPrintElems
    mkListStr :: (a -> String) -> [a] -> String
    mkListStr f = intercalate "\t" . map f -- (map show) -- (printf "%.2f"))
    putStateFeatList borl xs = setAllProxies proxyTable xs' borl
      where
        xs' = M.fromList $ zip (map (\xs -> (init xs, round (last xs))) xs) [0 ..]
    psiVFeat borl stateFeat aNr = P.lookupProxy (borl ^. t) Worker (stateFeat, aNr) (borl ^. proxies . psiV)
    psiWFeat borl stateFeat aNr = P.lookupProxy (borl ^. t) Worker (stateFeat, aNr) (borl ^. proxies . psiW)

debugStepsCount :: Integer
debugStepsCount = 4000
