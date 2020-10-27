{-# LANGUAGE BangPatterns              #-}
{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE Rank2Types                #-}
{-# LANGUAGE RankNTypes                #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE Strict                    #-}
{-# LANGUAGE StrictData                #-}
{-# LANGUAGE TupleSections             #-}
{-# LANGUAGE UndecidableInstances      #-}

module ML.BORL.Proxy.Ops
    ( insert
    , lookupProxy
    , lookupProxyAgent
    , lookupProxyNoUnscale
    , lookupState
    , lookupNeuralNetwork
    , lookupNeuralNetworkUnscaled
    , lookupActionsNeuralNetwork
    , lookupActionsNeuralNetworkUnscaled
    , mkNNList
    , getMinMaxVal
    , mkStateActs
    , StateFeatures
    , StateNextFeatures
    , LookupType (..)
    , AgentNumber
    ) where

import           Control.Applicative         ((<|>))
import           Control.Arrow
import           Control.Concurrent.MVar
import           Control.DeepSeq
import           Control.Lens
import           Control.Monad
import           Control.Monad.IO.Class
import           Control.Parallel.Strategies hiding (r0)
import           Data.Function               (on)
import           Data.List                   (foldl', sortBy, transpose)
import qualified Data.Map.Strict             as M
import           Data.Maybe                  (fromMaybe, isNothing)
import qualified Data.Vector                 as VB
import qualified Data.Vector.Storable        as V
import           Grenade
import           System.IO.Unsafe            (unsafePerformIO)

import           ML.BORL.Calculation.Type
import           ML.BORL.Decay
import           ML.BORL.NeuralNetwork
import           ML.BORL.Proxy.Proxies
import           ML.BORL.Proxy.Type
import           ML.BORL.Settings
import           ML.BORL.Type
import           ML.BORL.Types               as T
import           ML.BORL.Workers.Type

import           Debug.Trace

-- ^ Lookup Type for neural networks.
data LookupType = Target | Worker
  deriving (Eq, Ord, Show, Read)

-- data Output =
--   SingleAgent (V.Vector Float)
--   | MultiAgent (V.Vector (V.Vector Float))
--   deriving (Eq, Ord, Show, Read)


mkStateActs :: BORL s as -> s -> s -> (StateFeatures, (StateFeatures, FilteredActionIndices), (StateNextFeatures, FilteredActionIndices))
mkStateActs borl state stateNext = (stateFeat, stateActs, stateNextActs)
  where
    !sActIdxes = actionIndicesFiltered borl state
    !sNextActIdxes = actionIndicesFiltered borl stateNext
    !stateFeat = (borl ^. featureExtractor) state
    !stateNextFeat = (borl ^. featureExtractor) stateNext
    !stateActs = (stateFeat, sActIdxes)
    !stateNextActs = (stateNextFeat, sNextActIdxes)

------------------------------ inserts ------------------------------


-- | Insert (or update) a value.
insert ::
     forall m s as. (MonadIO m)
  => BORL s as                     -- ^ Latest BORL
  -> AgentType
  -> Period                     -- ^ Period when action was taken
  -> State s                    -- ^ State when action was taken
  -> ActionChoice               -- ^ RandomAction & ActionIndex for each agent
  -> RewardValue
  -> StateNext s
  -> EpisodeEnd
  -> ReplMemFun m s
  -> Proxies
  -> m (Proxies, Calculation)
insert borl agent _ state aNrs rew stateNext episodeEnd getCalc pxs
  | borl ^. settings . disableAllLearning = (pxs, ) . fst <$> getCalc stateActs aNrs rew stateNextActs episodeEnd emptyExpectedValuationNext
  where
    (_, stateActs, stateNextActs) = mkStateActs borl state stateNext
insert !borl !agent !period !state !as !rew !stateNext !episodeEnd !getCalc !pxs@(Proxies !pRhoMin !pRho !pPsiV !pV !pPsiW !pW !pR0 !pR1 !Nothing) = do
  (calc, _) <- getCalc stateActs as rew stateNextActs episodeEnd emptyExpectedValuationNext
  -- forkMv' <- liftIO $ doFork $ P.insert period label vValStateNew mv
  -- mv' <- liftIO $ collectForkResult forkMv'
  let aNr = VB.map snd as
  let mInsertProxy mVal px = maybe (return px) (\val ->  insertProxy agent (borl ^. settings) period stateFeat aNr val px) mVal
  pRhoMin' <- mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
  pRho' <-  mInsertProxy   (getRhoVal' calc) pRho             `using` rpar
  pV' <-    mInsertProxy   (getVValState' calc) pV            `using` rpar
  pW' <-    mInsertProxy   (getWValState' calc) pW            `using` rpar
  pPsiV' <- mInsertProxy   (getPsiVValState' calc) pPsiV      `using` rpar
  pPsiW' <- mInsertProxy   (getPsiWValState' calc) pPsiW      `using` rpar
  pR0' <-   mInsertProxy   (getR0ValState' calc) pR0          `using` rpar
  pR1' <-   mInsertProxy   (getR1ValState' calc) pR1          `using` rpar
  return (Proxies pRhoMin' pRho' pPsiV' pV' pPsiW' pW' pR0' pR1' Nothing, calc)
  where
    (stateFeat, stateActs, stateNextActs) = mkStateActs borl state stateNext
insert !borl !agent !period !state !as !rew !stateNext !episodeEnd !getCalc !pxs@(Proxies !pRhoMin !pRho !pPsiV !pV !pPsiW !pW !pR0 !pR1 (Just !replMems))
  | (1 + period) `mod` (borl ^. settings . nStep) /= 0 || period <= fromIntegral (replayMemoriesSubSize replMems) - 1 = do
    replMem' <- liftIO $ addToReplayMemories (borl ^. settings . nStep) (stateActs, as, rew, stateNextActs, episodeEnd) replMems
    (calc, _) <- getCalc stateActs as rew stateNextActs episodeEnd emptyExpectedValuationNext
    return (replayMemory ?~ replMem' $ pxs, calc)
  | otherwise = do
    !replMems' <- liftIO $ addToReplayMemories (borl ^. settings . nStep) (stateActs, as, rew, stateNextActs, episodeEnd) replMems
    (~calc, _) <- getCalc stateActs as rew stateNextActs episodeEnd emptyExpectedValuationNext
    let !config = fromMaybe (error "Neither v nor r1 holds a ANN proxy, but we got a replay memory...") $ pV ^? proxyNNConfig <|> borl ^? proxies.r1.proxyNNConfig  --  ## TODO why not r1
    let !workerReplMems = borl ^.. workers.traversed.workerReplayMemory
    !mems <- liftIO $ getRandomReplayMemoriesElements (borl ^. settings.nStep) (config ^. trainBatchSize) replMems'
    !workerMems <- liftIO $ mapM (getRandomReplayMemoriesElements (borl ^. settings.nStep) (config ^. trainBatchSize)) workerReplMems
    let mkCalc (s, idx, rew, s', epiEnd) = getCalc s idx rew s' epiEnd
    !calcs <- parMap rdeepseq force <$> mapM (executeAndCombineCalculations mkCalc) (mems ++ concat workerMems)
    let aNr = VB.map snd as
    let mInsertProxy mVal px = maybe (return px) (\val ->  insertProxy agent (borl ^. settings) period stateFeat aNr val px) mVal
    let mTrainBatch !accessor !calcs !px =
          maybe (return px) (\xs -> insertProxyMany agent (borl ^. settings) period xs px) (mapM (mapM (\c -> let (inp, mOut) = second accessor c in mOut >>= \out -> Just (inp, out))) calcs)
    !pRhoMin' <-
      if isNeuralNetwork pRhoMin
        then mTrainBatch getRhoMinimumVal' calcs pRhoMin `using` rpar
        else mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
    !pRho' <-
      if isNeuralNetwork pRho
        then mTrainBatch getRhoVal' calcs pRho `using` rpar
        else mInsertProxy (getRhoVal' calc) pRho `using` rpar
    !pV' <-     mTrainBatch getVValState' calcs pV `using` rpar
    !pW' <-     mTrainBatch getWValState' calcs pW `using` rpar
    !pPsiV' <-  mTrainBatch getPsiVValState' calcs pPsiV `using` rpar
    !pPsiW' <-  mTrainBatch getPsiWValState' calcs pPsiW `using` rpar
    !pR0' <-    mTrainBatch getR0ValState' calcs pR0 `using` rpar
    !pR1' <-    mTrainBatch getR1ValState' calcs pR1 `using` rpar
    return (Proxies pRhoMin' pRho' pPsiV' pV' pPsiW' pW' pR0' pR1' (Just replMems'), calc)
  where
    (!stateFeat, !stateActs, !stateNextActs) = mkStateActs borl state stateNext
insert !borl !agent !period !state !as !rew !stateNext !episodeEnd !getCalc !pxs@(ProxiesCombinedUnichain !pRhoMin !pRho !proxy Nothing) = do
  (calc, _) <- getCalc stateActs as rew stateNextActs episodeEnd emptyExpectedValuationNext
  let aNr = VB.map snd as
  let mInsertProxy !mVal !px = maybe (return (px, False)) (\val -> (, True) <$> insertProxy agent (borl ^. settings) period stateFeat aNr val px) mVal
  (!pRhoMin', _) <-        mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
  (!pRho', _) <-           mInsertProxy (getRhoVal' calc) pRho `using` rpar
  (!pV', vActive) <-       mInsertProxy (getVValState' calc) (pxs ^. v) `using` rpar
  (!pW', wActive) <-       mInsertProxy (getWValState' calc) (pxs ^. w) `using` rpar
  (!pPsiV', psiVActive) <- mInsertProxy (getPsiVValState' calc) (pxs ^. psiV) `using` rpar
  (!pPsiW', psiWActive) <- mInsertProxy (getPsiWValState' calc) (pxs ^. psiW) `using` rpar
  (!pR0', r0Active) <-     mInsertProxy (getR0ValState' calc) (pxs ^. r0) `using` rpar
  (!pR1', r1Active) <-     mInsertProxy (getR1ValState' calc) (pxs ^. r1) `using` rpar
  let combinedProxies = [pR0' | r0Active] ++ [pR1' | r1Active] ++ [pPsiV' | psiVActive] ++ [pV' | vActive] ++ [pPsiW' | psiWActive] ++  [pW' | wActive]
  !proxy' <- insertCombinedProxies agent (borl ^. settings) period combinedProxies
  return (ProxiesCombinedUnichain pRhoMin' pRho' proxy' Nothing, calc)
  where
    (stateFeat, stateActs, stateNextActs) = mkStateActs borl state stateNext
insert !borl !agent !period !state !as !rew !stateNext !episodeEnd !getCalc !pxs@(ProxiesCombinedUnichain !pRhoMin !pRho !proxy (Just !replMems))
  | (1 + period) `mod` (borl ^. settings.nStep) /= 0 || period <= fromIntegral (replayMemoriesSubSize replMems) - 1 = do
    !replMems' <- liftIO $ addToReplayMemories (borl ^. settings . nStep) (stateActs, as, rew, stateNextActs, episodeEnd) replMems
    (calc, _) <- getCalc stateActs as rew stateNextActs episodeEnd emptyExpectedValuationNext
    return (replayMemory ?~ replMems' $ pxs, calc)
  | otherwise = do
    !replMems' <- liftIO $ addToReplayMemories (borl ^. settings . nStep) (stateActs, as, rew, stateNextActs, episodeEnd) replMems
    (~calc, _) <- getCalc stateActs as rew stateNextActs episodeEnd emptyExpectedValuationNext
    let !config = proxy ^?! proxyNNConfig
    let !workerReplMems = borl ^.. workers.traversed.workerReplayMemory
    !mems <- liftIO $ getRandomReplayMemoriesElements (borl ^. settings . nStep) (config ^. trainBatchSize) replMems'
    !workerMems <- liftIO $ mapM (getRandomReplayMemoriesElements (borl ^. settings.nStep) (config ^. trainBatchSize)) workerReplMems
    let mkCalc (!sas, !idx, !sarew, !sas', !epiEnd) = getCalc sas idx sarew sas' epiEnd
    !calcs <- parMap rdeepseq force <$> mapM (executeAndCombineCalculations mkCalc) (mems ++ concat workerMems)
    let aNr = VB.map snd as
    let mInsertProxy !mVal !px = maybe (return (px, False)) (\val -> (,True) <$> insertProxy agent (borl ^. settings) period stateFeat aNr val px) mVal
    let mTrainBatch !accessor !calculations !px =
          maybe (return (px, False)) (\xs -> (,True) <$> insertProxyMany agent (borl ^. settings) period xs px) (mapM (mapM (\c -> let (inp, mOut) = second accessor c in mOut >>= \out -> Just (inp, out))) calculations)
    (!pRhoMin', _) <-
      if isNeuralNetwork pRhoMin
        then mTrainBatch getRhoMinimumVal' calcs pRhoMin `using` rpar
        else mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
    (!pRho', _) <-
      if isNeuralNetwork pRho
        then mTrainBatch getRhoVal' calcs pRho `using` rpar
        else mInsertProxy (getRhoVal' calc) pRho `using` rpar
    (!pV', vActive) <-       mTrainBatch getVValState' calcs (pxs ^. v) `using` rpar
    (!pW', wActive) <-       mTrainBatch getWValState' calcs (pxs ^. w) `using` rpar
    (!pPsiV', psiVActive) <- mTrainBatch getPsiVValState' calcs (pxs ^. psiV) `using` rpar
    (!pPsiW', psiWActive) <- mTrainBatch getPsiWValState' calcs (pxs ^. psiW) `using` rpar
    (!pR0', r0Active) <-     mTrainBatch getR0ValState' calcs (pxs ^. r0) `using` rpar
    (!pR1', r1Active) <-     mTrainBatch getR1ValState' calcs (pxs ^. r1) `using` rpar
    let combinedProxies = [pR0' | r0Active] ++ [pR1' | r1Active] ++ [pPsiV' | psiVActive] ++ [pV' | vActive] ++ [pPsiW' | psiWActive] ++  [pW' | wActive]
    !proxy' <- insertCombinedProxies agent (borl ^. settings) period combinedProxies
    return (ProxiesCombinedUnichain pRhoMin' pRho' proxy' (Just replMems'), calc)
  where
    (!stateFeat, !stateActs, !stateNextActs) = mkStateActs borl state stateNext


-- | Takes a list of calculations of consecutive periods, where the latest period is at the end.
executeAndCombineCalculations :: (MonadIO m) => (Experience -> ExpectedValuationNext -> m (Calculation, ExpectedValuationNext)) -> [Experience] -> m [((StateFeatures, AgentActionIndices), Calculation)]
executeAndCombineCalculations _ [] = error "Empty experiences in executeAndCombineCalculations"
executeAndCombineCalculations calcFun experiences = fst <$> foldM eval ([], emptyExpectedValuationNext) (reverse experiences)
  where
    eval (res, lastExpVal) experience@((state, _), idx, _, _, _) = do
      (calc, newExpVal) <- calcFun experience lastExpVal
      return (((state, VB.map snd idx), calc) : res, newExpVal)


-- | Insert a new (single) value to the proxy. For neural networks this will add the value to the startup table. See
-- `trainBatch` to train the neural networks.
insertProxy :: (MonadIO m) => AgentType -> Settings -> Period -> StateFeatures -> AgentActionIndices -> Value -> Proxy -> m Proxy
insertProxy !agent !setts !p !st !aNr !val = insertProxyMany agent setts p [[((st, aNr), val)]]

-- | Insert a new (single) value to the proxy. For neural networks this will add the value to the startup table. See
-- `trainBatch` to train the neural networks.
insertProxyMany :: (MonadIO m) => AgentType -> Settings -> Period -> [[((StateFeatures, AgentActionIndices), Value)]] -> Proxy -> m Proxy
insertProxyMany _ _ p [] px = trace ("\n\nEmpty input in insertProxyMany. Period: " ++ show p ++ "\n\n") return px
insertProxyMany _ _ _ !xs (Scalar _) = return $ Scalar (V.fromList $ fromValue $ snd $ last $ last xs)
insertProxyMany _ _ _ !xs (Table !m !def acts) = return $ Table m' def acts
  where
    trunc x = fromInteger (round $ x * (10 ^ n)) / (10.0 ^^ n)
    n = 3 :: Int
    m' = foldl' (\m' ((st, as), AgentValue vs) -> update m' st as vs) m (concat xs)
    update :: M.Map (StateFeatures, ActionIndex) (V.Vector Float) -> StateFeatures -> AgentActionIndices -> V.Vector Float -> M.Map (StateFeatures, ActionIndex) (V.Vector Float)
    update m st as vs = foldl' (\m' (idx, aNr, v) -> M.alter (\mOld -> Just $ fromMaybe def mOld V.// [(idx, v)]) (V.map trunc st, aNr) m') m (zip3 [0 ..V.length vs - 1] (VB.toList as) (V.toList vs))
insertProxyMany _ setts !period !xs px@(CombinedProxy !subPx !col !vs) -- only accumulate data if an update will follow
  | (1 + period) `mod` (setts ^. nStep) == 0 = return $ CombinedProxy subPx col (vs <> xs)
  | otherwise = return px
insertProxyMany agent setts !period _ !px | (1+period) `mod` (setts ^. nStep) /= 0 = updateNNTargetNet agent setts False period px -- skip ANN learning if not nStep or terminal
insertProxyMany agent setts !period !xs !px = emptyCache >> trainBatch period xs px >>= updateNNTargetNet agent setts False period


insertCombinedProxies :: (MonadIO m) => AgentType -> Settings -> Period -> [Proxy] -> m Proxy
insertCombinedProxies !agent !setts !period !pxs = set proxyType (head pxs ^?! proxyType) <$!> insertProxyMany agent setts period combineProxyExpectedOuts pxLearn
  where
    pxLearn = set proxyType (NoScaling (head pxs ^?! proxyType) mMinMaxs) $ head pxs ^?! proxySub
    combineProxyExpectedOuts :: [[((StateFeatures, VB.Vector ActionIndex), Value)]]
    combineProxyExpectedOuts = concatMap getAndScaleExpectedOutput (sortBy (compare `on` (^?! proxyOutCol)) pxs)
    len =  (head pxs ^?! proxyNrActions) * setts ^. independentAgents
    mMinMaxs = mapM getMinMaxVal pxs
    scaleAlg = pxLearn ^?! proxyNNConfig . scaleOutputAlgorithm
    getAndScaleExpectedOutput px@(CombinedProxy _ idx outs) =
      -- trace ("idx: " ++ show (idx, len))
      map (map (\((ft, curIdx), out) ->
                  -- trace ("curIdx: "++ show curIdx)
                  ((ft, VB.map (idx * len +) curIdx), scaleValue scaleAlg (getMinMaxVal px) out))) outs
    getAndScaleExpectedOutput px = error $ "unexpected proxy in insertCombinedProxies" ++ show px


-- | Copy the worker net to the target.
updateNNTargetNet :: (MonadIO m) => AgentType -> Settings -> Bool -> Period -> Proxy -> m Proxy
updateNNTargetNet _ _ _ _ px | not (isNeuralNetwork px) = error "updateNNTargetNet called on non-neural network proxy"
updateNNTargetNet agent setts forceReset period px
  | config ^. updateTargetInterval <= 1 || currentUpdateInterval <= 1 = return px
  | forceReset = copyValues
  | period <= memSubSize = return px
  | nStepUpdate || isGrenade px = copyValues -- updating 2 steps offset to round numbers to ensure we see the difference in the values
  | otherwise = return px
  where
    nStepUpdate
      | setts ^. nStep == 1 = (period - memSubSize - 1) `mod` currentUpdateInterval == 0
      | otherwise = any ((== 0) . (`mod` currentUpdateInterval)) [period - memSubSize - 1 - setts ^. nStep .. period - memSubSize - 1]
    memSubSize = px ^?! proxyNNConfig . replayMemoryMaxSize
    config = px ^?! proxyNNConfig
    currentUpdateInterval = max 1 $ round $ decaySetup (config ^. updateTargetIntervalDecay) period (fromIntegral $ config ^. updateTargetInterval)
    copyValues =
      case px of
        (Grenade netT' netW' tp' config' nrActs agents)
          | smoothUpd > 0 -> return $! Grenade (((1 - smoothUpd) |* netT') |+ (smoothUpd |* netW') `using` rdeepseq) netW' tp' config' nrActs agents
          | nStepUpdate -> return $ Grenade netW' netW' tp' config' nrActs agents
          | otherwise -> return px
          where smoothUpd = config ^. grenadeSmoothTargetUpdate
        CombinedProxy {} -> error "Combined proxy in updateNNTargetNet. Should not happen!"
        Table {} -> error "not possible"
        Scalar {} -> error "not possible"


-- | Train the neural network from a given batch. The training instances are Unscaled, that is in the range [-1, 1] or similar.
trainBatch :: forall m . (MonadIO m) => Period -> [[((StateFeatures, AgentActionIndices), Value)]] -> Proxy -> m Proxy
trainBatch !period !trainingInstances px@(Grenade !netT !netW !tp !config !nrActs !agents) = do
  let netW' = trainGrenade opt config minMaxVal netW (concatMap (map convertTrainingInstances) trainingInstances')
  return $! Grenade netT netW' tp config nrActs agents
  where
    minMaxVal =
      case px ^?! proxyType of
        NoScaling _ (Just minMaxVals) -> Just (minV, maxV)
          where minV = minimum $ map fst minMaxVals
                maxV = maximum $ map snd minMaxVals
        _ -> getMinMaxVal px
    convertTrainingInstances :: ((StateFeatures, VB.Vector ActionIndex), Value) -> [((StateFeatures, ActionIndex), Float)]
    convertTrainingInstances ((ft, as), AgentValue vs) = zipWith3 (\agNr aIdx val -> ((ft, agNr * nrActs + aIdx), val)) [0..VB.length as - 1] (VB.toList as) (V.toList vs)
    trainingInstances' =
      case px ^?! proxyType of
        NoScaling {} -> trainingInstances
        _ -> map (map (second $ scaleValue scaleAlg minMaxVal)) trainingInstances
    lRate = getLearningRate (config ^. grenadeLearningParams)
    scaleAlg = config ^. scaleOutputAlgorithm
    dec = decaySetup (config ^. learningParamsDecay) period
    opt = setLearningRate (realToFrac $ dec $ realToFrac lRate) (config ^. grenadeLearningParams)
trainBatch _ _ _ = error "called trainBatch on non-neural network proxy (programming error)"


------------------------------ lookup ------------------------------


-- -- | Retrieve a value.
-- lookupProxy :: (MonadIO m) => Period -> LookupType -> (StateFeatures, ActionIndex) -> Proxy -> m Value
-- lookupProxy _ _ _ (Scalar x)    = return $ AgentValue $ V.toList x
-- lookupProxy _ _ k (Table m def acts) = return $ AgentValue $ V.toList $ M.findWithDefault def k m
-- lookupProxy _ lkType k px       = lookupNeuralNetwork lkType k px

lookupProxyAgent :: (MonadIO m) => Period -> LookupType -> AgentNumber -> (StateFeatures, ActionIndex) -> Proxy -> m Float
lookupProxyAgent _ _ agNr _ (Scalar x)    = return $ x V.! agNr
lookupProxyAgent _ _ agNr (k, a) (Table m def _) = return $ M.findWithDefault def (k, a) m V.! agNr
lookupProxyAgent _ lkType agNr (k, a) px = selectIndex agNr <$> lookupNeuralNetwork lkType (k, VB.replicate agents a) px
  where
    agents = px ^?! proxyNrAgents


-- | Retrieve a value.
lookupProxy :: (MonadIO m) => Period -> LookupType -> (StateFeatures, AgentActionIndices) -> Proxy -> m Value
lookupProxy _ _ _ (Scalar x)    = return $ AgentValue x
lookupProxy _ _ (k, ass) (Table m def _) = return $ AgentValue $ V.convert $ VB.zipWith (\a agNr -> M.findWithDefault def (k, a) m V.! agNr) ass (VB.generate (VB.length ass) id)
lookupProxy _ lkType k px       = lookupNeuralNetwork lkType k px


-- | Retrieve a value, but do not unscale! For DEBUGGING only!
lookupProxyNoUnscale :: (MonadIO m) => Period -> LookupType -> (StateFeatures, AgentActionIndices) -> Proxy -> m Value
lookupProxyNoUnscale _ _ _ (Scalar x)    = return $ AgentValue x
lookupProxyNoUnscale _ _ (k,ass) (Table m def _) = return $ AgentValue $ V.convert $ VB.zipWith (\a agNr -> M.findWithDefault def (k,a) m V.! agNr) ass (VB.generate (VB.length ass) id)
lookupProxyNoUnscale _ lkType k px       = lookupNeuralNetworkUnscaled lkType k px


-- -- | Retrieves all action values for the state but filters to the provided actions.
-- lookupState :: (MonadIO m) => LookupType -> (StateFeatures, V.Vector ActionIndex) -> Proxy -> m Values
-- lookupState _ (_, as) (Scalar x) = return $ AgentValues $ replicate (V.length as) x
-- lookupState _ (k, as) (Table m def acts) = return $ AgentValues $ map V.fromList $ transpose $ map (\a -> V.toList $ M.findWithDefault def (k,a) m) (V.toList as)
-- lookupState tp (k, as) px = do
--   unfiltered@(AgentValues vals) <- lookupActionsNeuralNetwork tp k px
--   if V.length (head vals) == V.length as
--     then return unfiltered -- no need to filter
--     else return $ mapValues (\vs -> V.map (vs V.!) as) unfiltered

-- | Retrieves all action values for the state but filters to the provided actions.
lookupState :: (MonadIO m) => LookupType -> (StateFeatures, FilteredActionIndices) -> Proxy -> m Values
lookupState _ (_, ass) (Scalar x) = return $ AgentValues $ VB.zipWith (\agentValue as -> V.replicate (V.length as) agentValue) (V.convert x) ass
lookupState _ (k, ass) (Table m def _) = return $ AgentValues $ VB.zipWith (\as agNr -> V.map (\a -> M.findWithDefault def (k,a) m V.! agNr) as) ass (VB.fromList [0..VB.length ass-1])
lookupState tp (k, ass) px = do
  AgentValues vals <- lookupActionsNeuralNetwork tp k px
  return $ AgentValues $ VB.zipWith filterActions ass vals
  where
    filterActions :: V.Vector ActionIndex -> V.Vector Float -> V.Vector Float
    filterActions as vs =
      if V.length vs == V.length as
        then vs -- no need to filter
        else V.map (vs V.!) as


-- | Retrieve a value from a neural network proxy. The output is scaled to the original range. For other proxies an
-- error is thrown. The returned value is up-scaled to the original interval before returned.
lookupNeuralNetwork :: (MonadIO m) => LookupType -> (StateFeatures, AgentActionIndices) -> Proxy -> m Value
lookupNeuralNetwork !tp !k !px = unscaleVal <$> lookupNeuralNetworkUnscaled tp k px
  where scaleAlg = px ^?! proxyNNConfig . scaleOutputAlgorithm
        unscaleVal = unscaleValue scaleAlg (getMinMaxVal px)

-- | Retrieve all values of one feature from a neural network proxy. The output is scaled to the original range. For
-- other proxies an error is thrown. The returned value is up-scaled to the original interval before returned.
lookupActionsNeuralNetwork :: (MonadIO m) => LookupType -> StateFeatures -> Proxy -> m Values
lookupActionsNeuralNetwork !tp !k !px = unscaleVal <$> lookupActionsNeuralNetworkUnscaled tp k px
  where scaleAlg = px ^?! proxyNNConfig . scaleOutputAlgorithm
        unscaleVal = unscaleValues scaleAlg (getMinMaxVal px)

-- | Retrieve a value from a neural network proxy. The output is *not* scaled to the original range. For other proxies an error is thrown.
lookupNeuralNetworkUnscaled :: (MonadIO m) => LookupType -> (StateFeatures, AgentActionIndices) -> Proxy -> m Value
lookupNeuralNetworkUnscaled !tp (!st, !actIdx) px@Grenade{} = selectIndices actIdx <$> lookupActionsNeuralNetworkUnscaled tp st px
lookupNeuralNetworkUnscaled !tp (!st, !actIdx) (CombinedProxy px _ _) = lookupNeuralNetworkUnscaled tp (st, actIdx) px
lookupNeuralNetworkUnscaled _ _ _ = error "lookupNeuralNetworkUnscaled called on non-neural network proxy"


-- | Retrieve all action values of a state from a neural network proxy. For other proxies an error is thrown.
lookupActionsNeuralNetworkUnscaled :: (MonadIO m) => LookupType -> StateFeatures -> Proxy -> m Values
lookupActionsNeuralNetworkUnscaled tp st px@Grenade{} = head <$> lookupActionsNeuralNetworkUnscaledFull tp st px
lookupActionsNeuralNetworkUnscaled tp st (CombinedProxy px nr _) = (!! nr) <$> lookupActionsNeuralNetworkUnscaledFull tp st px
lookupActionsNeuralNetworkUnscaled _ _ _ = error "lookupNeuralNetworkUnscaled called on non-neural network proxy"


-- | Retrieve all action values of a state from a neural network proxy. For other proxies an error is thrown.
lookupActionsNeuralNetworkUnscaledFull :: (MonadIO m) => LookupType -> StateFeatures -> Proxy -> m [Values]
lookupActionsNeuralNetworkUnscaledFull Worker st (Grenade _ netW tp _ _ agents) =
  cached (Worker, tp, st) (return $ runGrenade netW agents st)
lookupActionsNeuralNetworkUnscaledFull Target st px@(Grenade netT _ tp config _ agents)
  | config ^. updateTargetInterval <= 1 = lookupActionsNeuralNetworkUnscaledFull Worker st px
  | otherwise = cached (Target, tp, st) (return $ runGrenade netT agents st)
lookupActionsNeuralNetworkUnscaledFull _ _ CombinedProxy{} = error "lookupActionsNeuralNetworkUnscaledFull called on CombinedProxy"
lookupActionsNeuralNetworkUnscaledFull _ _ _ = error "lookupActionsNeuralNetworkUnscaledFull called on a non-neural network proxy"


-- headLookupActions :: [p] -> p
-- headLookupActions []    = error "head: empty input data in lookupActionsNeuralNetworkUnscaled"
-- headLookupActions (x:_) = x


------------------------------ Helpers ------------------------------

-- | Caching of results
type CacheKey = (LookupType, ProxyType, StateFeatures)

cacheMVar :: MVar (M.Map CacheKey [Values])
cacheMVar = unsafePerformIO $ newMVar mempty
{-# NOINLINE cacheMVar #-}

emptyCache :: MonadIO m => m ()
emptyCache = liftIO $ modifyMVar_ cacheMVar (const mempty)

addCache :: (MonadIO m) => CacheKey -> [Values] -> m ()
addCache k val = liftIO $ modifyMVar_ cacheMVar (return . M.insert k val)

lookupCache :: (MonadIO m) => CacheKey -> m (Maybe [Values])
lookupCache k = liftIO $ (M.lookup k =<<) <$> tryReadMVar cacheMVar


-- | Get output of function f, if possible from cache according to key (st).
cached :: (MonadIO m) => (LookupType, ProxyType, StateFeatures) -> m [Values] -> m [Values]
cached st ~f = do
  c <- lookupCache st
  case c of
    Nothing -> do
      res <- f
      res `seq` addCache st res
      return res
    Just res -> return res


-- | Finds the correct value for scaling.
getMinMaxVal :: Proxy -> Maybe (MinValue Float, MaxValue Float)
getMinMaxVal Table{} = error "getMinMaxVal called for Table"
getMinMaxVal !p =
  case unCombine (p ^?! proxyType) of
    VTable -> Just (p ^?! proxyNNConfig . scaleParameters . scaleMinVValue, p ^?! proxyNNConfig . scaleParameters . scaleMaxVValue)
    WTable -> Just (p ^?! proxyNNConfig . scaleParameters . scaleMinWValue, p ^?! proxyNNConfig . scaleParameters . scaleMaxWValue)
    R0Table -> Just (p ^?! proxyNNConfig . scaleParameters . scaleMinR0Value, p ^?! proxyNNConfig . scaleParameters . scaleMaxR0Value)
    R1Table -> Just (p ^?! proxyNNConfig . scaleParameters . scaleMinR1Value, p ^?! proxyNNConfig . scaleParameters . scaleMaxR1Value)
    PsiVTable -> Just (1.0 * p ^?! proxyNNConfig . scaleParameters . scaleMinVValue, 1.0 * p ^?! proxyNNConfig . scaleParameters . scaleMaxVValue)
    PsiWTable -> Just (1.0 * p ^?! proxyNNConfig . scaleParameters . scaleMinVValue, 1.0 * p ^?! proxyNNConfig . scaleParameters . scaleMaxVValue)
    NoScaling {} -> Nothing
    CombinedUnichain -> error "should not happend"
    -- CombinedUnichainScaleAs {} -> error "should not happend"
  where
    unCombine CombinedUnichain
      | isCombinedProxy p = fromCombinedIndex (p ^?! proxyOutCol)
    -- unCombine (CombinedUnichainScaleAs x)
    --   | isCombinedProxy p = x
    unCombine x = x


-- | This function retrieves the data and builds a table like return value.
mkNNList :: (MonadIO m) => BORL s as -> Bool -> Proxy -> m [(NetInputWoAction, ([(ActionIndex, Value)], [(ActionIndex, Value)]))]
mkNNList !borl !scaled !pr =
  mapM
    (\st -> do
       target <-
         if scaled
           then lookupActionsNeuralNetwork Target st pr
           else lookupActionsNeuralNetworkUnscaled Target st pr
       worker <-
         if scaled
           then lookupActionsNeuralNetwork Worker st pr
           else lookupActionsNeuralNetworkUnscaled Worker st pr
       return (st, (zip actIdxs (toActionValue target), zip actIdxs (toActionValue worker))))
       -- return (st, (replicate agents actIdxs))
    (conf ^. prettyPrintElems)
  where
    conf = pr ^?! proxyNNConfig
    agents = pr ^?! proxyNrAgents
    actIdxs = [0 .. (nrActs - 1)]
    nrActs = pr ^?! proxyNrActions
