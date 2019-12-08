{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
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
import           ML.BORL.NeuralNetwork.Scaling
import           ML.BORL.Parameters
import           ML.BORL.Properties
import           ML.BORL.Proxy                  as P
import           ML.BORL.Reward
import           ML.BORL.SaveRestore
import           ML.BORL.Serialisable
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Applicative            ((<|>))
import           Control.Arrow                  ((&&&), (***))
import           Control.DeepSeq
import           Control.DeepSeq                (NFData, force)
import           Control.Lens
import           Control.Monad
import           Control.Monad.IO.Class         (liftIO)
import           Control.Monad.IO.Class         (MonadIO, liftIO)
import           Control.Parallel.Strategies    hiding (r0)
import           Data.Function                  (on)
import           Data.List                      (find, groupBy, intercalate, partition,
                                                 sortBy)
import qualified Data.Map.Strict                as M
import           Data.Maybe                     (fromMaybe, isJust)
import           Data.Serialize
import           GHC.Generics
import           System.Directory
import           System.IO
import           System.Random
import           Text.Printf

import           Debug.Trace

fileDebugStateV :: FilePath
fileDebugStateV = "stateVAllStates"

fileDebugStateW :: FilePath
fileDebugStateW = "stateWAllStates"

fileDebugStateW2 :: FilePath
fileDebugStateW2 = "stateW2AllStates"


fileDebugPsiVValues :: FilePath
fileDebugPsiVValues = "statePsiVAllStates"

fileDebugPsiWValues :: FilePath
fileDebugPsiWValues = "statePsiWAllStates"

fileDebugPsiW2Values :: FilePath
fileDebugPsiW2Values = "statePsiW2AllStates"

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
    Nothing -> runMonadBorlIO $ force <$> foldM (\b _ -> nextAction (force b) >>= stepExecute) borl [0 .. nr - 1]
    Just _ ->
      runMonadBorlTF $ do
        void $ restoreTensorflowModels True borl
        !borl' <- foldM (\b _ -> nextAction (force b) >>= stepExecute) borl [0 .. nr - 1]
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
stepM (force -> borl) = nextAction (force borl) >>= stepExecute

-- | This keeps the Tensorflow session alive. For non-Tensorflow BORL data structures this is equal to steps, but forces
-- evaluation of the data structure every 1000 periods.
stepsM :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> Integer -> m (BORL s)
stepsM (force -> borl) nr = do
  !borl' <- force <$> foldM (\b _ -> nextAction (force b) >>= stepExecute) borl [1 .. min maxNr nr]
  if nr > maxNr
    then stepsM borl' (nr - maxNr)
    else return borl'
  where maxNr = 1000

data Decision = Random | MaxRho | MaxV | MaxE
  deriving (Show, Read, Eq, Ord, Generic, NFData, Serialize)

-- | This function chooses the next action from the current state s and all possible actions.
nextAction :: (MonadBorl' m) => BORL s -> m (BORL s, Bool, ActionIndexed s)
nextAction borl
  | null as = error "Empty action list"
  | length as == 1 = return (borl, False, head as)
  | otherwise = do
    rand <- liftIO $ randomRIO (0, 1)
    if rand < explore
      then do
        r <- liftIO $ randomRIO (0, length as - 1)
        return (borl, True, as !! r)
      else case borl ^. algorithm of
             AlgBORL _ _ _ decideVPlusPsi _ -> do
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
                          r <- liftIO $ randomRIO (0, length bestE - 1)
                          return (borl, False, bestE !! r)
                        else return (borl, False, headE bestE)
             AlgBORLVOnly {} -> singleValueNextAction (vValue False borl state . fst)
             AlgDQN {} -> singleValueNextAction (rValue borl RBig state . fst)
             AlgDQNAvgRewardFree {} -> do
               r1Values <- mapM (rValue borl RBig state . fst) as
               let bestR1ValueActions = headV $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (zip r1Values as)
                   bestR1 = map snd bestR1ValueActions
               r0Values <- mapM (rValue borl RSmall state . fst) bestR1
               let r1Value = fst $ headR1 bestR1ValueActions
                   group = groupBy (epsCompare (==) `on` fst) . sortBy (epsCompare compare `on` fst)
                   (posErr,negErr) = (group *** group) $ partition ((r1Value<) . fst) (zip r0Values bestR1)
               let bestR0 = map snd $ head $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (headR0 $ if null posErr then negErr else posErr)
               -- trace ("bestR1: " ++ show bestR1) $
               --  trace ("bestR0: " ++ show bestR0) $
               if length bestR1 == 1
                 then return (borl, False, head bestR1)
                 else if length bestR0 > 1
                        then do
                          r <- liftIO $ randomRIO (0, length bestR0 - 1)
                          return (borl, False, bestR0 !! r)
                        else return (borl, False, headDqnAvgRewFree bestR0)

               -- singleValueNextAction
  where
    headRho []    = error "head: empty input data in nextAction on Rho value"
    headRho (x:_) = x
    headV []    = error "head: empty input data in nextAction on V value"
    headV (x:_) = x
    headE []    = error "head: empty input data in nextAction on E Value"
    headE (x:_) = x
    headR0 []    = error "head: empty input data in nextAction on R0 Value"
    headR0 (x:_) = x
    headR1 []    = error "head: empty input data in nextAction on R1 Value"
    headR1 (x:_) = x
    headDqn []    = error "head: empty input data in nextAction on Dqn Value"
    headDqn (x:_) = x
    headDqnAvgRewFree []    = error "head: empty input data in nextAction on DqnAvgRewFree Value"
    headDqnAvgRewFree (x:_) = x
    gamma0 = case borl ^. algorithm of
      AlgBORL g0 _ _ _ _         -> g0
      AlgDQN g0                  -> g0
      AlgDQNAvgRewardFree g0 _ _ -> g0
      AlgBORLVOnly _ _           -> 1
    params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
    eps = params' ^. epsilon
    explore = params' ^. exploration
    state = borl ^. s
    as = actionsIndexed borl state
    epsCompare = epsCompareWith eps
    singleValueNextAction f = do
      rValues <- mapM f as
      let bestR = sortBy (epsCompare compare `on` fst) (zip rValues as)
      return (borl, False, snd $ headDqn bestR)

epsCompareWith :: (Ord t, Num t) => t -> (t -> t -> p) -> t -> t -> p
epsCompareWith eps f x y
  | abs (x - y) <= eps = f 0 0
  | otherwise = y `f` x


stepExecute :: forall m s . (MonadBorl' m, NFData s, Ord s, RewardFuture s) => (BORL s, Bool, ActionIndexed s) -> m (BORL s)
stepExecute (borl, randomAction, (aNr, Action action _)) = do
  let state = borl ^. s
      period = borl ^. t + length (borl ^. futureRewards)
  (reward, stateNext, episodeEnd) <- liftIO $ action state
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
  when (borl ^. t == 0) $ forM_ [fileDebugStateV, fileDebugStateW, fileDebugStateW2, fileDebugPsiWValues, fileDebugPsiVValues, fileDebugPsiWValues, fileDebugPsiW2Values, fileDebugStateValuesNrStates] $ \f ->
    liftIO $ doesFileExist f >>= \x -> when x (removeFile f)
  borl <- writeDebugFiles borl
#endif
  (proxies', calc) <- P.insert borl period state aNr randomAction reward stateNext episodeEnd (mkCalculation borl) (borl ^. proxies)
  let lastVsLst = fromMaybe [0] (getLastVs' calc)
  -- File IO Operations
  when (period == 0) $ do
    liftIO $ writeFile fileStateValues "Period\tRho\tMinRho\tVAvg\tR0\tR1\n"
    liftIO $ writeFile fileEpisodeLength "Episode\tEpisodeLength\n"
    liftIO $ writeFile fileReward "Period\tReward\n"
  let strRho = show (fromMaybe 0 (getRhoVal' calc))
      strMinV = show (fromMaybe 0 (getRhoMinimumVal' calc))
      strVAvg = show (avg lastVsLst)
      strR0 = show $ fromMaybe 0 (getR0ValState' calc)
      strR1 = show $ fromMaybe 0 (getR1ValState' calc)
      avg xs = sum xs / fromIntegral (length xs)
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
    set psis (fromMaybe 0 (getPsiValRho' calc), fromMaybe 0 (getPsiValV' calc), fromMaybe 0 (getPsiValW' calc), fromMaybe 0 (getPsiValW2' calc)) $
    set lastVValues (fromMaybe [] (getLastVs' calc)) $ set lastRewards (getLastRews' calc) $ set proxies proxies' $ set t (period + 1) $ over episodeNrStart setEpisode borl
execute _ _ = error "Exectue on invalid data structure. This is a bug!"


#ifdef DEBUG
writeDebugFiles :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> m (BORL s)
writeDebugFiles borl = do
  let isDqn = isAlgDqn (borl ^. algorithm) || isAlgDqnAvgRewardFree (borl ^. algorithm)
  let isAnn
        | isDqn = P.isNeuralNetwork (borl ^. proxies . r1)
        | otherwise = P.isNeuralNetwork (borl ^. proxies . v)
  let putStateFeatList borl xs
        | isAnn = borl
        | otherwise = setAllProxies proxyTable xs' borl
        where
          xs' = M.fromList $ zip (map (\xs -> (init xs, round (last xs))) xs) (repeat 0)
  borl' <-
    if borl ^. t > 0
      then return borl
      else do
        liftIO $ writeFile fileDebugStateV ""
        liftIO $ writeFile fileDebugStateW ""
        liftIO $ writeFile fileDebugStateW2 ""
        liftIO $ writeFile fileDebugPsiVValues ""
        liftIO $ writeFile fileDebugPsiWValues ""
        liftIO $ writeFile fileDebugPsiW2Values ""
        liftIO $ writeFile fileDebugStateValuesNrStates "-1"
        borl' <-
          if isAnn
            then return borl
            else stepsM
                   (setAllProxies (proxyNNConfig . trainMSEMax) Nothing $ setAllProxies (proxyNNConfig . replayMemoryMaxSize) 1000 $ set t 1 borl)
                   debugStepsCount -- run steps to fill the table with (hopefully) all states
        let stateFeats
              | isDqn = getStateFeatList (borl' ^. proxies . r1)
              | otherwise = getStateFeatList (borl' ^. proxies . v)
        liftIO $ forM_ [fileDebugStateV, fileDebugStateW, fileDebugStateW2, fileDebugPsiVValues, fileDebugPsiWValues, fileDebugPsiW2Values] $ flip writeFile ("Period\t" <> mkListStr show stateFeats <> "\n")
        liftIO $ writeFile fileDebugStateValuesNrStates (show $ length stateFeats)
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
  len <- liftIO $ read <$> readFile fileDebugStateValuesNrStates
  when (len >= 0 && len /= length stateFeats) $ error $ "Number of states to write to debug file changed from " <> show len <> " to " <> show (length stateFeats) <>
    ". Increase debugStepsCount count in Step.hs!"
  when ((borl' ^. t `mod` debugPrintCount) == 0) $ do
    stateValuesV <- mapM (\xs -> if isDqn then rValueFeat borl' RBig (init xs) (round $ last xs) else vValueFeat False borl' (init xs) (round $ last xs)) stateFeats
    stateValuesW <- mapM (\xs -> if isDqn then return 0 else wValueFeat borl' (init xs) (round $ last xs)) stateFeats
    stateValuesW2 <- mapM (\xs -> if isDqn then return 0 else w2ValueFeat borl' (init xs) (round $ last xs)) stateFeats
    liftIO $ appendFile fileDebugStateV (show (borl' ^. t) <> "\t" <> mkListStr show stateValuesV <> "\n")
    when (isAlgBorl (borl ^. algorithm)) $ do
      liftIO $ appendFile fileDebugStateW (show (borl' ^. t) <> "\t" <> mkListStr show stateValuesW <> "\n")
      liftIO $ appendFile fileDebugStateW2 (show (borl' ^. t) <> "\t" <> mkListStr show stateValuesW2 <> "\n")
      psiVValues <- mapM (\xs -> psiVFeat borl' (init xs) (round $ last xs)) stateFeats
      liftIO $ appendFile fileDebugPsiVValues (show (borl' ^. t) <> "\t" <> mkListStr show psiVValues <> "\n")
      psiWValues <- mapM (\xs -> psiWFeat borl' (init xs) (round $ last xs)) stateFeats
      liftIO $ appendFile fileDebugPsiWValues (show (borl' ^. t) <> "\t" <> mkListStr show psiWValues <> "\n")
      psiW2Values <- mapM (\xs -> psiW2Feat borl' (init xs) (round $ last xs)) stateFeats
      liftIO $ appendFile fileDebugPsiW2Values (show (borl' ^. t) <> "\t" <> mkListStr show psiW2Values <> "\n")
  return borl'
  where
    getStateFeatList Scalar {} = []
    getStateFeatList (Table t _) = map (\(xs, y) -> xs ++ [fromIntegral y]) (M.keys t)
    getStateFeatList nn = concatMap (\xs -> map (\(idx, _) -> xs ++ [fromIntegral idx]) acts) (nn ^. proxyNNConfig . prettyPrintElems)
    acts = borl ^. actionList
    mkListStr :: (a -> String) -> [a] -> String
    mkListStr f = intercalate "\t" . map f
    psiVFeat borl stateFeat aNr = P.lookupProxy (borl ^. t) Worker (stateFeat, aNr) (borl ^. proxies . psiV)
    psiWFeat borl stateFeat aNr = P.lookupProxy (borl ^. t) Worker (stateFeat, aNr) (borl ^. proxies . psiW)
    psiW2Feat borl stateFeat aNr = P.lookupProxy (borl ^. t) Worker (stateFeat, aNr) (borl ^. proxies . psiW2)

debugStepsCount :: Integer
debugStepsCount = 8000

debugPrintCount :: Int
debugPrintCount = 100

#endif
