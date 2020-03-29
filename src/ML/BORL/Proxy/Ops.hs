{-# LANGUAGE BangPatterns              #-}
{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE ExplicitForAll            #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE Rank2Types                #-}
{-# LANGUAGE RankNTypes                #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TemplateHaskell           #-}
{-# LANGUAGE TupleSections             #-}
{-# LANGUAGE UndecidableInstances      #-}

module ML.BORL.Proxy.Ops
    ( insert
    , lookupProxy
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
    ) where

import           Control.Arrow
import           Control.Concurrent.MVar
import           Control.DeepSeq
import           Control.Lens
import           Control.Lens
import           Control.Monad
import           Control.Monad.IO.Class       (liftIO)
import           Control.Parallel.Strategies  hiding (r0)
import           Data.Function                (on)
import           Data.List                    (find, foldl', sortBy, transpose)
import qualified Data.Map.Strict              as M
import qualified Data.Map.Strict              as M
import           Data.Maybe                   (fromJust, isJust, isNothing)
import qualified Data.Set                     as S
import           Data.Singletons.Prelude.List
import qualified Data.Vector                  as VB
import qualified Data.Vector.Storable         as V
import           GHC.Generics
import           GHC.TypeLits
import           Grenade
import           System.IO.Unsafe             (unsafePerformIO)
import           System.Random.Shuffle

import           ML.BORL.Calculation.Type
import           ML.BORL.Decay
import           ML.BORL.Fork
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import           ML.BORL.Proxy.Proxies
import           ML.BORL.Proxy.Type
import           ML.BORL.Reward
import           ML.BORL.Type
import           ML.BORL.Types                as T
import           ML.BORL.Workers.Type

import           Debug.Trace

-- ^ Lookup Type for neural networks.
data LookupType = Target | Worker
  deriving (Eq, Ord)


mkStateActs :: BORL s -> s -> s -> (StateFeatures, (StateFeatures, FilteredActionIndices), (StateNextFeatures, FilteredActionIndices))
mkStateActs borl state stateNext = (stateFeat, stateActs, stateNextActs)
    where
    !sActIdxes = V.convert $ VB.map fst $ actionsIndexed borl state
    !sNextActIdxes = V.convert $ VB.map fst $ actionsIndexed borl stateNext
    !stateFeat = (borl ^. featureExtractor) state
    !stateNextFeat = (borl ^. featureExtractor) stateNext
    !stateActs = (stateFeat, sActIdxes)
    !stateNextActs = (stateNextFeat, sNextActIdxes)


-- | Insert (or update) a value.
insert ::
     forall m s. (NFData s, Ord s, MonadBorl' m)
  => BORL s                     -- ^ Latest BORL
  -> Period                     -- ^ Period when action was taken
  -> State s                    -- ^ State when action was taken
  -> ActionIndex
  -> IsRandomAction
  -> RewardValue
  -> StateNext s
  -> EpisodeEnd
  -> ReplMemFun s
  -> Proxies
  -> m (Proxies, Calculation)
insert borl _ state aNr randAct rew stateNext episodeEnd getCalc pxs
  | borl ^. parameters . disableAllLearning = (pxs, ) <$> getCalc stateActs aNr randAct rew stateNextActs episodeEnd
  where
    (_, stateActs, stateNextActs) = mkStateActs borl state stateNext
insert !borl !period !state !aNr !randAct !rew !stateNext !episodeEnd !getCalc !pxs@(Proxies !pRhoMin !pRho !pPsiV !pV !pPsiW !pW !pR0 !pR1 !Nothing) = do
  calc <- getCalc stateActs aNr randAct rew stateNextActs episodeEnd
  -- forkMv' <- liftIO $ doFork $ P.insert period label vValStateNew mv
  -- mv' <- liftIO $ collectForkResult forkMv'
  let mInsertProxy mVal px = maybe (return px) (\val -> insertProxy period stateFeat aNr val px) mVal
  pRhoMin' <- mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
  pRho' <- mInsertProxy (getRhoVal' calc) pRho `using` rpar
  pV' <- mInsertProxy (getVValState' calc) pV `using` rpar
  pW' <- mInsertProxy (getWValState' calc) pW `using` rpar
  pPsiV' <- mInsertProxy (getPsiVValState' calc) pPsiV `using` rpar
  pPsiW' <- mInsertProxy (getPsiWValState' calc) pPsiW `using` rpar
  pR0' <- mInsertProxy (getR0ValState' calc) pR0 `using` rpar
  pR1' <- mInsertProxy (getR1ValState' calc) pR1 `using` rpar
  return $! force (Proxies pRhoMin' pRho' pPsiV' pV' pPsiW' pW' pR0' pR1' Nothing, calc)
  where
    (stateFeat, stateActs, stateNextActs) = mkStateActs borl state stateNext
insert !borl !period !state !aNr !randAct !rew !stateNext !episodeEnd !getCalc !pxs@(Proxies !pRhoMin !pRho !pPsiV !pV !pPsiW !pW !pR0 !pR1 (Just !replMems))
  | pV ^?! proxyNNConfig . replayMemoryMaxSize <= 1 = insert borl period state aNr randAct rew stateNext episodeEnd getCalc (Proxies pRhoMin pRho pPsiV pV pPsiW pW pR0 pR1 Nothing)
  | period <= fromIntegral (replMems ^. replayMemories idxStart . replayMemorySize) - 1 = do
    replMem' <- liftIO $ addToReplayMemories (stateActs, aNr, randAct, rew, stateNextActs, episodeEnd) replMems
    (pxs', calc) <- insert borl period state aNr randAct rew stateNext episodeEnd getCalc (replayMemory .~ Nothing $ pxs)
    return $! force (replayMemory ?~ replMem' $ pxs', calc)
  | otherwise = do
    !replMems' <- liftIO $ addToReplayMemories (stateActs, aNr, randAct, rew, stateNextActs, episodeEnd) replMems
    !calc <- getCalc stateActs aNr randAct rew stateNextActs episodeEnd
    let !config = pV ^?! proxyNNConfig --  ## TODO why not r1
    let !workerReplMems = borl ^. workers.traversed.workersReplayMemories
    !mems <- liftIO $ getRandomReplayMemoriesElements (config ^. trainBatchSize) replMems'
    !workerMems <- liftIO $ mapM (getRandomReplayMemoriesElements (max 1 $ config ^. trainBatchSize `div` length workerReplMems)) workerReplMems
    let mkCalc (s, idx, rand, rew, s', epiEnd) = getCalc s idx rand rew s' epiEnd
    !calcs <- parMap rdeepseq force <$> mapM (\m@((s, _), idx, _, _, _, _) -> mkCalc m >>= \v -> return ((s, idx), v)) (mems ++ concat workerMems)
    let mInsertProxy mVal px = maybe (return px) (\val -> insertProxy period stateFeat aNr val px) mVal
    let mTrainBatch !accessor !calcs !px =
          maybe
            (return px)
            (\xs -> insertProxyMany period xs px)
            (mapM
               (\c ->
                  let (inp, mOut) = second accessor c
                   in mOut >>= \out -> Just (inp, out))
               calcs)
    !pRhoMin' <-
      if isNeuralNetwork pRhoMin
        then mTrainBatch getRhoMinimumVal' calcs pRhoMin `using` rpar
        else mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
    !pRho' <-
      if isNeuralNetwork pRho
        then mTrainBatch getRhoVal' calcs pRho `using` rpar
        else mInsertProxy (getRhoVal' calc) pRho `using` rpar
    !pV' <- mTrainBatch getVValState' calcs pV `using` rpar
    !pW' <- mTrainBatch getWValState' calcs pW `using` rpar
    !pPsiV' <- mTrainBatch getPsiVValState' calcs pPsiV `using` rpar
    !pPsiW' <- mTrainBatch getPsiWValState' calcs pPsiW `using` rpar
    !pR0' <- mTrainBatch getR0ValState' calcs pR0 `using` rpar
    !pR1' <- mTrainBatch getR1ValState' calcs pR1 `using` rpar
    return $! force (Proxies pRhoMin' pRho' pPsiV' pV' pPsiW' pW' pR0' pR1' (Just replMems'), calc)
  where
    (!stateFeat, !stateActs, !stateNextActs) = mkStateActs borl state stateNext
insert !borl !period !state !aNr !randAct !rew !stateNext !episodeEnd !getCalc !pxs@(ProxiesCombinedUnichain !pRhoMin !pRho !proxy !Nothing) = do
  !calc <- getCalc stateActs aNr randAct rew stateNextActs episodeEnd
  let mInsertProxy !mVal !px = maybe (return px) (\val -> insertProxy period stateFeat aNr val px) mVal
  !pRhoMin' <- mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
  !pRho' <- mInsertProxy (getRhoVal' calc) pRho `using` rpar
  !pV' <- mInsertProxy (getVValState' calc) (pxs ^. v) `using` rpar
  !pW' <- mInsertProxy (getWValState' calc) (pxs ^. w) `using` rpar
  !pPsiV' <- mInsertProxy (getPsiVValState' calc) (pxs ^. psiV) `using` rpar
  !pPsiW' <- mInsertProxy (getPsiWValState' calc) (pxs ^. psiW) `using` rpar
  !pR0' <- mInsertProxy (getR0ValState' calc) (pxs ^. r0) `using` rpar
  !pR1' <- mInsertProxy (getR1ValState' calc) (pxs ^. r1) `using` rpar
  !proxy' <- insertCombinedProxies period [pR0', pR1', pPsiV', pV', pPsiW', pW']
  return $! force (ProxiesCombinedUnichain pRhoMin' pRho' proxy' Nothing, calc)
  where
    (stateFeat, stateActs, stateNextActs) = mkStateActs borl state stateNext
insert !borl !period !state !aNr !randAct !rew !stateNext !episodeEnd !getCalc !pxs@(ProxiesCombinedUnichain !pRhoMin !pRho !proxy !(Just !replMems))
  | proxy ^?! proxyNNConfig . replayMemoryMaxSize <= 1 = insert borl period state aNr randAct rew stateNext episodeEnd getCalc (ProxiesCombinedUnichain pRhoMin pRho proxy Nothing)
  | period <= fromIntegral (replMems ^. replayMemories idxStart . replayMemorySize) - 1 = do
    !replMems' <- liftIO $ addToReplayMemories (stateActs, aNr, randAct, rew, stateNextActs, episodeEnd) replMems
    (!pxs', !calc) <- insert borl period state aNr randAct rew stateNext episodeEnd getCalc (replayMemory .~ Nothing $ pxs)
    return $! force (replayMemory ?~ replMems' $ pxs', calc)
  | otherwise = do
    !replMems' <- liftIO $ addToReplayMemories (stateActs, aNr, randAct, rew, stateNextActs, episodeEnd) replMems
    !calc <- getCalc stateActs aNr randAct rew stateNextActs episodeEnd
    let !config = proxy ^?! proxyNNConfig
    let !workerReplMems = borl ^. workers.traversed.workersReplayMemories
    !mems <- liftIO $ getRandomReplayMemoriesElements (config ^. trainBatchSize) replMems'
    !workerMems <- liftIO $ mapM (getRandomReplayMemoriesElements (max 1 $ config ^. trainBatchSize `div` length workerReplMems)) workerReplMems
    let mkCalc (!s, !idx, !rand, !rew, !s', !epiEnd) = getCalc s idx rand rew s' epiEnd
    !calcs <- parMap rdeepseq force <$> mapM (\m@((s, _), idx, _, _, _, _) -> mkCalc m >>= \v -> return ((s, idx), v)) (mems ++ concat workerMems)
    let mInsertProxy !mVal !px = maybe (return px) (\val -> insertProxy period stateFeat aNr val px) mVal
    let mTrainBatch !accessor !calcs !px =
          maybe
            (return px)
            (\xs -> insertProxyMany period xs px)
            (mapM
               (\c ->
                  let (inp, mOut) = second accessor c
                   in mOut >>= \out -> Just (inp, out))
               calcs)
    !pRhoMin' <-
      if isNeuralNetwork pRhoMin
        then mTrainBatch getRhoMinimumVal' calcs pRhoMin `using` rpar
        else mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
    !pRho' <-
      if isNeuralNetwork pRho
        then mTrainBatch getRhoVal' calcs pRho `using` rpar
        else mInsertProxy (getRhoVal' calc) pRho `using` rpar
    !pV' <- mTrainBatch getVValState' calcs (pxs ^. v) `using` rpar
    !pW' <- mTrainBatch getWValState' calcs (pxs ^. w) `using` rpar
    !pPsiV' <- mTrainBatch getPsiVValState' calcs (pxs ^. psiV) `using` rpar
    !pPsiW' <- mTrainBatch getPsiWValState' calcs (pxs ^. psiW) `using` rpar
    !pR0' <- mTrainBatch getR0ValState' calcs (pxs ^. r0) `using` rpar
    !pR1' <- mTrainBatch getR1ValState' calcs (pxs ^. r1) `using` rpar
    !proxy' <- insertCombinedProxies period [pR0', pR1', pPsiV', pV', pPsiW', pW']
    return $! force (ProxiesCombinedUnichain pRhoMin' pRho' proxy' (Just replMems'), calc)
  where
    (!stateFeat, !stateActs, !stateNextActs) = mkStateActs borl state stateNext

-- | Caching of results
type CacheKey = (LookupType, ProxyType, StateFeatures)

cacheMVar :: MVar (M.Map CacheKey NetOutput)
cacheMVar = unsafePerformIO $ newMVar mempty
{-# NOINLINE cacheMVar #-}

emptyCache :: MonadBorl' m => m ()
emptyCache = liftIO $ modifyMVar_ cacheMVar (const mempty)

addCache :: (MonadBorl' m) => CacheKey -> NetOutput -> m ()
addCache k v = liftIO $ modifyMVar_ cacheMVar (return . M.insert k v)

lookupCache :: (MonadBorl' m) => CacheKey -> m (Maybe NetOutput)
lookupCache k = liftIO $ (M.lookup k =<<) <$> tryReadMVar cacheMVar


-- | Insert a new (single) value to the proxy. For neural networks this will add the value to the startup table. See
-- `trainBatch` to train the neural networks.
insertProxy :: (MonadBorl' m) => Period -> StateFeatures -> ActionIndex -> Float -> Proxy -> m Proxy
insertProxy !p !st !aNr !val = insertProxyMany p [((st, aNr), val)]

-- | Insert a new (single) value to the proxy. For neural networks this will add the value to the startup table. See
-- `trainBatch` to train the neural networks.
insertProxyMany :: (MonadBorl' m) => Period -> [((StateFeatures, ActionIndex), Float)] -> Proxy -> m Proxy
insertProxyMany _ !xs !(Scalar _) = return $ Scalar (snd $ last xs)
insertProxyMany _ !xs !(Table !m !def) = return $! force $! Table (foldl' (\m' ((st,aNr),v') -> M.insert (V.map trunc st, aNr) v' m') m xs) def
  where trunc x = fromInteger (round $ x * (10^n)) / (10.0^^n)
        n = 3
insertProxyMany !_ !xs !(CombinedProxy !subPx !col !vs) = return $ CombinedProxy subPx col (vs <> xs)
insertProxyMany !period !xs !px
  | period < memSize - 1 && isNothing (px ^?! proxyNNConfig . trainMSEMax) = return $ proxyNNStartup .~ foldl' (\m ((st, aNr), v) -> M.insert (st, aNr) 0 m) tab xs $ px
  | period < memSize - 1 = return px
  | period == memSize - 1 && (isNothing (px ^?! proxyNNConfig . trainMSEMax) || px ^?! proxyNNConfig . replayMemoryMaxSize == 1) = emptyCache >> updateNNTargetNet True period px
  | period == memSize - 1 = liftIO (putStrLn $ "Initializing artificial neural networks: " ++ show (px ^? proxyType)) >> emptyCache >> netInit px >>= updateNNTargetNet True period
  | otherwise = emptyCache >> trainBatch period xs px >>= updateNNTargetNet False period
  where
    netInit px = trainMSE (Just 0) (M.toList tab) (config ^. grenadeLearningParams) px -- no decay needed
    config = px ^?! proxyNNConfig
    tab = px ^?! proxyNNStartup
    memSize = fromIntegral (px ^?! proxyNNConfig . replayMemoryMaxSize)


insertCombinedProxies :: (MonadBorl' m) => Period -> [Proxy] -> m Proxy
insertCombinedProxies !period !pxs = scaleTab unscaleValue . set proxyType (head pxs ^?! proxyType) <$!> insertProxyMany period combineProxyExpectedOuts pxLearn
  where
    scaleTab f px
      | period == memSize - 1 = proxyNNStartup .~ M.mapWithKey (\(_, idx) -> scaleIndex f idx) (px ^?! proxyNNStartup) $ px
      | otherwise = px
    scaleIndex f idx val = maybe (error $ "could not find proxy for idx: " ++ show idx) (\px -> f (getMinMaxVal px) val) (find ((== idx `div` len) . (^?! proxyOutCol)) pxs)
    pxLearn = scaleTab scaleValue $ set proxyType (NoScaling $ head pxs ^?! proxyType) $ head pxs ^?! proxySub
    combineProxyExpectedOuts =
      concatMap
        (\px@(CombinedProxy _ idx outs) -> map (\((ft, curIdx), out) -> ((ft, idx * len + curIdx), scaleValue' (getMinMaxVal px) out)) outs)
        (sortBy (compare `on` (^?! proxyOutCol)) pxs)
    len = head pxs ^?! proxyNrActions
    scaleValue' val
      | period < memSize - 1 = id
      | otherwise = scaleValue val
    memSize = fromIntegral (head pxs ^?! proxyNNConfig . replayMemoryMaxSize)


-- | Copy the worker net to the target.
updateNNTargetNet :: (MonadBorl' m) => Bool -> Period -> Proxy -> m Proxy
updateNNTargetNet _ _ px | not (isNeuralNetwork px) = error "updateNNTargetNet called on non-neural network proxy"
updateNNTargetNet !forceReset !period !px
  | config ^. updateTargetInterval <= 1 || currentUpdateInterval <= 1 = return px
  | forceReset = copyValues
  | period <= memSize = return px
  | ((period - memSize - 1) `mod` currentUpdateInterval) == 0 = copyValues -- updating 2 steps offset to round numbers to ensure we see the difference in the values
  | otherwise = return px
  where
    memSize = px ^?! proxyNNConfig . replayMemoryMaxSize
    config = px ^?! proxyNNConfig
    currentUpdateInterval = max 1 $ round $ decaySetup (config ^. updateTargetIntervalDecay) period (fromIntegral $ config ^. updateTargetInterval)
    copyValues =
      case px of
        (Grenade _ netW' tab' tp' config' nrActs) -> return $! Grenade netW' netW' tab' tp' config' nrActs
        (TensorflowProxy netT' netW' tab' tp' config' nrActs) -> do
          copyValuesFromTo netW' netT'
          return $! TensorflowProxy netT' netW' tab' tp' config' nrActs
        CombinedProxy {} -> error "Combined proxy in updateNNTargetNet. Should not happen!"
        Table {} -> error "not possible"
        Scalar {} -> error "not possible"


-- | Train the neural network from a given batch. The training instances are Unscaled, that is in the range [-1, 1] or similar.
trainBatch :: forall m . (MonadBorl' m) => Period -> [((StateFeatures, ActionIndex), Float)] -> Proxy -> m Proxy
trainBatch !period !trainingInstances !px@(Grenade !netT !netW !tab !tp !config !nrActs) = do
  let netW' = foldl' (trainGrenade lp) netW (map return trainingInstances')
  return $! Grenade netT netW' tab tp config nrActs
  where
    trainingInstances' = map (second $ scaleValue (getMinMaxVal px)) trainingInstances
    LearningParameters lRate momentum l2 = config ^. grenadeLearningParams
    dec = decaySetup (config ^. learningParamsDecay) period
    lp = LearningParameters (realToFrac $ dec $ realToFrac lRate) momentum l2
trainBatch !period !trainingInstances !px@(TensorflowProxy !netT !netW !tab !tp !config !nrActs) = do
  backwardRunRepMemData netW trainingInstances'
  if period == px ^?! proxyNNConfig . replayMemoryMaxSize
    then do
      lrs <- getLearningRates netW
      when (null lrs) $ error "Could not get the Tensorflow learning rate in Proxy.Ops"
      when (length lrs > 1) $ error "Cannot handle multiple Tensorflow optimizers (multiple learning rates) in Proxy.Ops"
      return $! TensorflowProxy netT netW tab tp (grenadeLearningParams .~ LearningParameters (realToFrac $ head lrs) 0 0 $ config) nrActs
    else do
      when (period `mod` 1000 == 0 && dec lRate /= lRate) $
        setLearningRates [dec lRate] netW -- this seems to be an expensive operation!
      -- when (period `mod` 100 == 0) $
      --   getLearningRates netW >>= liftIO . print
      return $! TensorflowProxy netT netW tab tp config nrActs
  where
    trainingInstances' = map (second $ scaleValue (getMinMaxVal px)) trainingInstances
    dec = decaySetup (config ^. learningParamsDecay) period
    LearningParameters lRateDb _ _ = config ^. grenadeLearningParams
    lRate = realToFrac lRateDb
trainBatch _ _ _ = error "called trainBatch on non-neural network proxy (programming error)"


-- | Train until MSE hits the value given in NNConfig.
trainMSE :: (MonadBorl' m) => Maybe Int -> [((StateFeatures, ActionIndex), Float)] -> LearningParameters -> Proxy -> m Proxy
trainMSE _ _ _ px@Table{} = return px
trainMSE !mIteration !dataset !lp !px@(Grenade !_ !netW !tab !tp !config !nrActs)
  | isNothing (config ^. trainMSEMax) = return px
  | mIteration > Just 200 = do
      liftIO $ putStrLn "Giving up after 200 Iterartions :-(   Think about changing your network topography and/or scaling settings!"
      return px
  | mse < mseMax = do
    liftIO $ putStrLn $ "Final MSE for " ++ show tp ++ ": " ++ show mse
    return px
  | otherwise = do
    datasetShuffled <- liftIO $ shuffleM dataset
    let net' = foldl' (trainGrenade lp) netW (zipWith (curry return) (map fst datasetShuffled) (map snd datasetShuffled))
    when (maybe False ((== 0) . (`mod` 5)) mIteration) $ liftIO $ putStrLn $ "Current MSE for " ++ show tp ++ ": " ++ show mse
    fmap force <$> trainMSE ((+ 1) <$> mIteration) dataset lp $ Grenade net' net' tab tp config nrActs
  where
    mseMax = fromJust (config ^. trainMSEMax)
    -- net' = trainGrenade lp netSGD (zip kScaled vScaled)
    vScaled = map (scaleValue (getMinMaxVal px) . snd) dataset
    -- vUnscaled = map snd dataset
    -- kScaled = map fst dataset
    -- (minV,maxV) = getMinMaxVal px
    getValue k =
       -- unscaleValue (getMinMaxVal px) $
     (\x -> x V.! snd k) $ snd $ fromLastShapes netW $ runNetwork netW ((toHeadShapes netW . fst) k)
    mse = 1 / fromIntegral (length dataset) * sum (zipWith (\k v -> (v - getValue k) ** 2) (map fst dataset) (map (min 1 . max (-1)) vScaled)) -- scaled or unscaled ones?
trainMSE !mIteration !dataset !lp !px@(TensorflowProxy !netT !netW !tab !tp !config !nrActs)
  | isNothing (config ^. trainMSEMax) = return px
  | mIteration > Just 200 = do
      liftIO $ putStrLn "Giving up after 200 Iterartions :-(   Think about changing your network topography and/or scaling settings!"
      return px
  | otherwise = do
    datasetShuffled <- liftIO $ shuffleM dataset
    let mseMax = fromJust (config ^. trainMSEMax)
        kFullScaled = map fst datasetShuffled :: [(StateFeatures, ActionIndex)]
        kScaled = map fst kFullScaled
        actIdxs = map snd kFullScaled
        -- (minV,maxV) = getMinMaxVal px
        vScaledDbl = map (scaleValue (getMinMaxVal px) . snd) datasetShuffled
        vScaled = map realToFrac vScaledDbl
        -- vUnscaled = map (realToFrac . snd) dataset
        -- datasetRepMem = map (first (first (map realToFrac))) datasetShuffled
    current <- forwardRun netW kScaled
    zipWithM_ (backwardRun netW) (map return kScaled) (map return $ zipWith (V.//) current (map return $ zip actIdxs vScaled))
    -- backwardRunRepMemData netW datasetRepMem
    let forward k = realToFrac <$> lookupNeuralNetworkUnscaled Worker k px -- lookupNeuralNetworkUnscaled Worker k px -- scaled or unscaled ones?
    mse <- ((1 / fromIntegral (length datasetShuffled)) *) . sum <$> zipWithM (\k v -> (** 2) . (v-) <$> forward k) (map fst datasetShuffled) (map (min 1 . max (-1)) vScaledDbl) -- scaled or unscaled ones?
    if realToFrac mse < mseMax
      then do
      liftIO $ putStrLn ("Final MSE for " ++ show tp ++ ": " ++ show mse)
      void $ saveModelWithLastIO netW -- Save model to ensure correct values when reading from another session
      return px
      else do
      when (maybe False ((== 0) . (`mod` 5)) mIteration) $ liftIO $ putStrLn $ "Current MSE for " ++ show tp ++ ": " ++ show mse
      trainMSE ((+ 1) <$!> mIteration) dataset lp (TensorflowProxy netT netW tab tp config nrActs)
trainMSE _ _ _ _ = error "trainMSE should not have been callable with this type of proxy. programming error!"

-- | Retrieve a value.
lookupProxy :: (MonadBorl' m) => Period -> LookupType -> (StateFeatures, ActionIndex) -> Proxy -> m Float
lookupProxy !_ !_ !_ !(Scalar !x) = return x
lookupProxy !_ !_ !k !(Table !m !def) = return $! M.findWithDefault def k m
lookupProxy !period !lkType !k@(feat, !actIdx) !px
  | period <= fromIntegral (config ^. replayMemoryMaxSize) && (config ^. trainBatchSize) /= 1 = return $! M.findWithDefault 0 k' tab
  | otherwise = lookupNeuralNetwork lkType k px
  where
    config = px ^?! proxyNNConfig
    tab = px ^?! proxyNNStartup
    k' = case px of
      CombinedProxy _ nr _ -> (feat, nr * px ^?! proxyNrActions + actIdx)
      _                    -> k


-- | Retrieve a value from a neural network proxy. The output is sclaed to the original range. For other proxies an
-- error is thrown. The returned value is up-scaled to the original interval before returned.
lookupNeuralNetwork :: (MonadBorl' m) => LookupType -> (StateFeatures, ActionIndex) -> Proxy -> m Float
lookupNeuralNetwork !tp !k !px = unscaleValue (getMinMaxVal px) <$> lookupNeuralNetworkUnscaled tp k px

-- | Retrieve all values of one feature from a neural network proxy. The output is sclaed to the original range. For
-- other proxies an error is thrown. The returned value is up-scaled to the original interval before returned.
lookupActionsNeuralNetwork :: (MonadBorl' m) => LookupType -> StateFeatures -> Proxy -> m NetOutput
lookupActionsNeuralNetwork !tp !k !px = V.map (unscaleValue (getMinMaxVal px)) <$> lookupActionsNeuralNetworkUnscaled tp k px

-- | Retrieve a value from a neural network proxy. The output is *not* scaled to the original range. For other proxies
-- an error is thrown.
lookupNeuralNetworkUnscaled :: (MonadBorl' m) => LookupType -> (StateFeatures, ActionIndex) -> Proxy -> m Float
lookupNeuralNetworkUnscaled !tp !(!st, !actIdx) px@Grenade{} = (V.! actIdx) <$> lookupActionsNeuralNetworkUnscaled tp st px
lookupNeuralNetworkUnscaled !tp !(!st, !actIdx) px@TensorflowProxy {} = (V.! actIdx) <$> lookupActionsNeuralNetworkUnscaled tp st px
lookupNeuralNetworkUnscaled !tp !(!st, !actIdx) p@(CombinedProxy px nr _) = lookupNeuralNetworkUnscaled tp (st, nr * px ^?! proxyNrActions + actIdx) px
lookupNeuralNetworkUnscaled _ _ _ = error "lookupNeuralNetworkUnscaled called on non-neural network proxy"

headLookupActions :: [p] -> p
headLookupActions []    = error "head: empty input data in lookupActionsNeuralNetworkUnscaled"
headLookupActions (x:_) = x

-- | Get output of function f, if possible from cache according to key (st).
cached :: (MonadBorl' m) => (LookupType, ProxyType, StateFeatures) -> m NetOutput -> m NetOutput
cached st f = do
  c <- lookupCache st
  case c of
    Nothing -> do
      res <- f
      addCache st res
      return res
    Just res -> return res

-- | Retrieve all action values of a state from a neural network proxy. For other proxies an error is thrown.
lookupActionsNeuralNetworkUnscaled :: (MonadBorl' m) => LookupType -> StateFeatures -> Proxy -> m NetOutput
lookupActionsNeuralNetworkUnscaled Worker !st (Grenade !_ !netW !_ !tp !_ !_) = cached (Worker, tp, st) (return $ snd $ fromLastShapes netW $ runNetwork netW (toHeadShapes netW st))
lookupActionsNeuralNetworkUnscaled Target !st !px@(Grenade !netT !_ !_ !tp !config !_)
  | config ^. updateTargetInterval <= 1 = lookupActionsNeuralNetworkUnscaled Worker st px
  | otherwise = cached (Target, tp, st) (return $ snd $ fromLastShapes netT $ runNetwork netT (toHeadShapes netT st))
lookupActionsNeuralNetworkUnscaled Worker !st !(TensorflowProxy !_ !netW !_ !tp !_ !_) = cached (Worker, tp, st) (headLookupActions <$> forwardRun netW [st])
lookupActionsNeuralNetworkUnscaled Target !st !px@(TensorflowProxy !netT !_ !_ !tp !config !_)
  | config ^. updateTargetInterval <= 1 = lookupActionsNeuralNetworkUnscaled Worker st px
  | otherwise = cached (Target, tp, st) (headLookupActions <$> forwardRun netT [st])
lookupActionsNeuralNetworkUnscaled !tp !st !(CombinedProxy !px !nr !_) = V.slice (nr*nrActs) nrActs <$> cached (tp, CombinedUnichain, st) (lookupActionsNeuralNetworkUnscaled tp st px)
  where nrActs = px ^?! proxyNrActions
lookupActionsNeuralNetworkUnscaled _ _ _ = error "lookupNeuralNetworkUnscaled called on non-neural network proxy"


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
mkNNList :: (MonadBorl' m) => BORL k -> Bool -> Proxy -> m [(NetInputWoAction, ([(ActionIndex, Float)], [(ActionIndex, Float)]))]
mkNNList !borl !scaled !pr =
  mapM
    (\st -> do
       target <-
         if useTable
           then return $ lookupTable scaled st
           else if scaled
                  then V.toList <$> lookupActionsNeuralNetwork Target st pr
                  else V.toList <$> lookupActionsNeuralNetworkUnscaled Target st pr
       worker <-
         if scaled
           then V.toList <$> lookupActionsNeuralNetwork Worker st pr
           else V.toList <$> lookupActionsNeuralNetworkUnscaled Worker st pr
       return (st, (zip actIdxs target, zip actIdxs worker)))
    (conf ^. prettyPrintElems)
  where
    conf = pr ^?! proxyNNConfig
    actIdxs = [0 .. (pr ^?! proxyNrActions - 1)]
    useTable = borl ^. t == fromIntegral (pr ^?! proxyNNConfig . replayMemoryMaxSize) && (pr ^?! proxyNNConfig . trainBatchSize) /= 1
    lookupTable :: Bool -> NetInputWoAction -> [Float]
    lookupTable scale st
      | scale = val -- values are being unscaled, thus let table value be unscaled
      | otherwise = map (scaleValue (getMinMaxVal pr)) val
      where
        val = map (\actNr -> M.findWithDefault 0 (st, actNr) (pr ^?! proxyNNStartup)) [0 .. (pr ^?! proxyNrActions - 1)]
