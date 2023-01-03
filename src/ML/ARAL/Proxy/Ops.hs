{-# LANGUAGE BangPatterns              #-}
{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE Rank2Types                #-}
{-# LANGUAGE RankNTypes                #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TemplateHaskell           #-}
{-# LANGUAGE TupleSections             #-}
{-# LANGUAGE UndecidableInstances      #-}

module ML.ARAL.Proxy.Ops
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

import           Control.Applicative                         ((<|>))
import           Control.Arrow
import           Control.Concurrent.MVar
import           Control.DeepSeq
import           Control.Exception
import           Control.Lens
import           Control.Monad
import           Control.Monad.IO.Class
import           Control.Parallel.Strategies                 hiding (r0)
import           Data.Function                               (on)
import           Data.List                                   (foldl', sortBy, transpose)
import qualified Data.Map.Strict                             as M
import           Data.Maybe                                  (fromMaybe, isNothing)
import qualified Data.Text                                   as T
import qualified Data.Vector                                 as VB
import qualified Data.Vector.Storable                        as V
import           EasyLogger
import           Grenade
import           Say
import           Statistics.Sample.WelfordOnlineMeanVariance
import           System.IO.Unsafe                            (unsafePerformIO)
import           System.Random
import qualified Torch                                       as Torch
import qualified Torch.Optim                                 as Torch

import           RegNet
-- import           ML.ARAL.Proxy.Regression.RegressionLayer


import           ML.ARAL.Calculation.Type
import           ML.ARAL.Decay
import           ML.ARAL.NeuralNetwork
import           ML.ARAL.NeuralNetwork.Hasktorch
import           ML.ARAL.NeuralNetwork.Normalisation
import           ML.ARAL.Proxy.Proxies
import           ML.ARAL.Proxy.Type
import           ML.ARAL.Settings
import           ML.ARAL.Type
import           ML.ARAL.Types                               as T
import           ML.ARAL.Workers.Type

import           Debug.Trace

-- ^ Lookup Type for neural networks.
data LookupType = Target | Worker
  deriving (Eq, Ord, Show, Read)

-- data Output =
--   SingleAgent (V.Vector Double)
--   | MultiAgent (V.Vector (V.Vector Double))
--   deriving (Eq, Ord, Show, Read)

mkStateActs :: ARAL s as -> s -> s -> (StateFeatures, (StateFeatures, DisallowedActionIndicies), (StateNextFeatures, DisallowedActionIndicies))
mkStateActs borl state stateNext = (stateFeat, stateActs, stateNextActs)
  where
    -- !sActIdxes = actionIndicesFiltered borl state
    !sActIdxes = actionIndicesDisallowed borl state
    -- !sNextActIdxes = actionIndicesFiltered borl stateNext
    !sNextActIdxes = actionIndicesDisallowed borl stateNext
    !stateFeat = (borl ^. featureExtractor) state
    !stateNextFeat = (borl ^. featureExtractor) stateNext
    !stateActs = (stateFeat, sActIdxes)
    !stateNextActs = (stateNextFeat, sNextActIdxes)

------------------------------ inserts ------------------------------


-- | Insert (or update) a value.
insert ::
     forall m s as. (MonadIO m)
  => ARAL s as                     -- ^ Latest ARAL
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
  let aRand = or $ VB.map fst as
  let mInsertProxy mVal px = maybe (return px) (\val -> insertProxy agent (borl ^. settings) period stateFeat aNr rew aRand val px) mVal
  pRhoMin' <- mInsertProxy (getRhoMinimumVal' calc) pRhoMin   `using` rpar
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
  | (1 + period) `mod` (borl ^. settings . nStep) /= 0 || period <= fromIntegral (replayMemoriesSize replMems) - 1 = do
    replMem' <- liftIO $ addToReplayMemories (borl ^. settings . nStep) (stateActs, as, rew, stateNextActs, episodeEnd) replMems
    (calc, _) <- getCalc stateActs as rew stateNextActs episodeEnd emptyExpectedValuationNext
    let aNr = VB.map snd as
    let aRand = or $ VB.map fst as
    let mInsertProxy mVal px = maybe (return px) (\val -> insertProxy agent (borl ^. settings) period stateFeat aNr rew aRand val px) mVal
        mInsertWelford mVal px = maybe (return px) (\val -> return $ addWelford period [[((stateFeat, aNr, rew, aRand), val)]] px) mVal
    !pRhoMin' <- mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
    !pRho'    <- mInsertProxy (getRhoVal' calc) pRho `using` rpar
    !pV'      <- mInsertWelford (getVValState' calc) (pxs ^. v) `using` rpar
    !pW'      <- mInsertWelford (getWValState' calc) (pxs ^. w) `using` rpar
    !pPsiV'   <- mInsertWelford (getPsiVValState' calc) (pxs ^. psiV) `using` rpar
    !pPsiW'   <- mInsertWelford (getPsiWValState' calc) (pxs ^. psiW) `using` rpar
    !pR0'     <- mInsertWelford (getR0ValState' calc) (pxs ^. r0) `using` rpar
    !pR1'     <- mInsertWelford (getR1ValState' calc) (pxs ^. r1) `using` rpar
    when (period == fromIntegral (replayMemoriesSize replMems) - 1) $ $(logPrintInfoText) (T.pack $ "Starting to learn. Period: " <> show period)
    emptyCache
    return $ (Proxies pRhoMin' pRho' pPsiV' pV' pPsiW' pW' pR0' pR1' (Just replMem'), calc)
  | otherwise = do
    !replMems' <- liftIO $ addToReplayMemories (borl ^. settings . nStep) (stateActs, as, rew, stateNextActs, episodeEnd) replMems
    (~calc, _) <- getCalc stateActs as rew stateNextActs episodeEnd emptyExpectedValuationNext
    let !config = fromMaybe (error "Neither v nor r1 holds a ANN proxy, but we got a replay memory...") $ pV ^? proxyNNConfig <|> borl ^? proxies.r1.proxyNNConfig
    let !workerReplMems = borl ^.. workers.traversed.workerReplayMemory
    !mems <- liftIO $ getRandomReplayMemoriesElements (borl ^. settings.nStep) (config ^. trainBatchSize) replMems'
    !workerMems <- liftIO $ mapM (getRandomReplayMemoriesElements (borl ^. settings.nStep) (config ^. trainBatchSize)) workerReplMems
    let mkCalc (s, idx, rew, s', epiEnd) = getCalc s idx rew s' epiEnd
    !calcs <- parMap rdeepseq force <$> mapM (executeAndCombineCalculations mkCalc) (mems ++ concat workerMems)
    let mInsertProxy mVals px = maybe (return px) (\val -> insertProxyManyScalar agent (borl ^. settings) period val px) mVals
    -- let mInsertProxy mVal px = maybe (return px) (\val ->  insertProxy agent (borl ^. settings) period stateFeat aNr val px) mVal
    --     aNr = VB.map snd as
    let mTrainBatch !accessor !calcs !px =
          maybe (return px) (\xs -> insertProxyMany agent (borl ^. settings) period xs px) (mapM (mapM (\c -> let (inp, mOut) = second accessor c in mOut >>= \out -> Just (inp, out))) calcs)
    !pRhoMin' <-
      if isNeuralNetwork pRhoMin
        then mTrainBatch getRhoMinimumVal' calcs pRhoMin `using` rpar
        else mInsertProxy (traverse (traverse (getRhoMinimumVal' . snd)) calcs) pRhoMin `using` rpar
        -- else mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
    !pRho' <-
      if isNeuralNetwork pRho
        then mTrainBatch getRhoVal' calcs pRho `using` rpar
        else mInsertProxy (traverse (traverse (getRhoVal' . snd))  calcs) pRho `using` rpar
        -- else mInsertProxy (getRhoVal' calc) pRho `using` rpar
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
  let aRand = or $ VB.map fst as
  let mInsertProxy !mVal !px = maybe (return (px, False)) (\val -> (, True) <$> insertProxy agent (borl ^. settings) period stateFeat aNr rew aRand val px) mVal
  (!pRhoMin', _)        <- mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
  (!pRho', _)           <- mInsertProxy (getRhoVal' calc) pRho `using` rpar
  (!pV', vActive)       <- mInsertProxy (getVValState' calc) (pxs ^. v) `using` rpar
  (!pW', wActive)       <- mInsertProxy (getWValState' calc) (pxs ^. w) `using` rpar
  (!pPsiV', psiVActive) <- mInsertProxy (getPsiVValState' calc) (pxs ^. psiV) `using` rpar
  (!pPsiW', psiWActive) <- mInsertProxy (getPsiWValState' calc) (pxs ^. psiW) `using` rpar
  (!pR0', r0Active)     <- mInsertProxy (getR0ValState' calc) (pxs ^. r0) `using` rpar
  (!pR1', r1Active)     <- mInsertProxy (getR1ValState' calc) (pxs ^. r1) `using` rpar
  let combinedProxies = [pR0' | r0Active] ++ [pR1' | r1Active] ++ [pPsiV' | psiVActive] ++ [pV' | vActive] ++ [pPsiW' | psiWActive] ++  [pW' | wActive]
  !proxy' <- insertCombinedProxies agent (borl ^. settings) period combinedProxies
  return (ProxiesCombinedUnichain pRhoMin' pRho' proxy' Nothing, calc)
  where
    (stateFeat, stateActs, stateNextActs) = mkStateActs borl state stateNext
insert !borl !agent !period !state !as !rew !stateNext !episodeEnd !getCalc !pxs@(ProxiesCombinedUnichain !pRhoMin !pRho !proxy (Just !replMems))
  | (1 + period) `mod` (borl ^. settings.nStep) /= 0 || period <= fromIntegral (replayMemoriesSize replMems) - 1 = do
    !replMem' <- liftIO $ addToReplayMemories (borl ^. settings . nStep) (stateActs, as, rew, stateNextActs, episodeEnd) replMems
    (calc, _) <- getCalc stateActs as rew stateNextActs episodeEnd emptyExpectedValuationNext
    let aNr = VB.map snd as
    let aRand = or $ VB.map fst as
    let mInsertProxy mVal px = maybe (return px) (\val -> insertProxy agent (borl ^. settings) period stateFeat aNr rew aRand val px) mVal
    !pRhoMin' <- mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
    !pRho' <- mInsertProxy (getRhoVal' calc) pRho `using` rpar
    when (period == fromIntegral (replayMemoriesSize replMems) - 1) $ $(logPrintInfoText) (T.pack $ "Starting to learn. Period: " <>  show period)
    emptyCache
    return (replayMemory ?~ replMem' $ pxs { _rhoMinimum = pRhoMin', _rho = pRho' }, calc)
  | otherwise = do
    !replMems' <- liftIO $ addToReplayMemories (borl ^. settings . nStep) (stateActs, as, rew, stateNextActs, episodeEnd) replMems
    (~calc, _) <- getCalc stateActs as rew stateNextActs episodeEnd emptyExpectedValuationNext
    let !config = proxy ^?! proxyNNConfig
    let !workerReplMems = borl ^.. workers.traversed.workerReplayMemory
    !mems <- liftIO $ getRandomReplayMemoriesElements (borl ^. settings . nStep) (config ^. trainBatchSize) replMems'
    !workerMems <- liftIO $ mapM (getRandomReplayMemoriesElements (borl ^. settings.nStep) (config ^. trainBatchSize)) workerReplMems
    let mkCalc (!sas, !idx, !sarew, !sas', !epiEnd) = getCalc sas idx sarew sas' epiEnd
    !calcs <- parMap rdeepseq force <$> mapM (executeAndCombineCalculations mkCalc) (mems ++ concat workerMems)
    let mInsertProxy mVals px = maybe (return px) (\val -> insertProxyManyScalar agent (borl ^. settings) period val px) mVals
    -- let mInsertProxy mVal px = maybe (return px) (\val ->  insertProxy agent (borl ^. settings) period stateFeat aNr val px) mVal
    --     aNr = VB.map snd as
    let mTrainBatch !field !calculations !px =
          maybe (return (px, False)) (\xs -> (,True) <$> insertProxyMany agent (borl ^. settings) period xs px) (mapM (mapM (\c -> let (inp, mOut) = second field c in mOut >>= \out -> Just (inp, out))) calculations)
    !pRhoMin' <-
      if isNeuralNetwork pRhoMin
        then fst <$> mTrainBatch getRhoMinimumVal' calcs pRhoMin `using` rpar
        else mInsertProxy (traverse (traverse (getRhoMinimumVal' . snd)) calcs) pRhoMin `using` rpar
        -- else mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
    !pRho' <-
      if isNeuralNetwork pRho
        then fst <$> mTrainBatch getRhoVal' calcs pRho `using` rpar
        else mInsertProxy (traverse (traverse (getRhoVal' . snd)) calcs) pRho `using` rpar
        -- else mInsertProxy (getRhoVal' calc) pRho `using` rpar
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


avg :: [Value] -> Value
avg []        = error "empty values in avg in Proxy.Ops"
avg [x]       = x
avg xs'@(x:_) = applyToValue sum xs' / toValue (valueLength x) (fromIntegral (length xs'))

-- | Takes a list of calculations of consecutive periods, where the latest period is at the end.
executeAndCombineCalculations ::
     (MonadIO m)
  => (Experience -> ExpectedValuationNext -> m (Calculation, ExpectedValuationNext))
  -> [Experience]
  -> m [((StateFeatures, AgentActionIndices, RewardValue, IsRandomAction), Calculation)]
executeAndCombineCalculations _ [] = error "Empty experiences in executeAndCombineCalculations"
executeAndCombineCalculations calcFun experiences = fst <$> foldM eval ([], emptyExpectedValuationNext) (reverse experiences)
  where
    eval (res, lastExpVal) experience@((state, _), idx, rew, _, _) = do
      (calc, newExpVal) <- calcFun experience lastExpVal
      return (((state, VB.map snd idx, rew, VB.or $ VB.map fst idx), calc) : res, newExpVal)


-- | Insert a new (single) value to the proxy. For neural networks this will add the value to the startup table. See
-- `trainBatch` to train the neural networks.
insertProxy :: (MonadIO m) => AgentType -> Settings -> Period -> StateFeatures -> AgentActionIndices -> RewardValue -> IsRandomAction -> Value -> Proxy -> m Proxy
insertProxy !agent !setts !p !st !aNr !rew !aRand !val = insertProxyMany agent setts p [[((st, aNr, rew, aRand), val)]]

insertProxyManyScalar :: (MonadIO m) => AgentType -> Settings -> Period -> [[Value]] -> Proxy -> m Proxy
insertProxyManyScalar _ _ p [] px               = liftIO $ putStrLn ("\n\nEmpty input in insertProxyMany. Period: " ++ show p) >> return px
insertProxyManyScalar _ _ _ !xs (Scalar _ nrAs) = return $ Scalar (unpackValue $ avg $ map avg xs) nrAs
insertProxyManyScalar _ _ _ _ _                 = error "Called insertProxyManyScalar on nonScalar proxy! This is a programming error"


-- | Insert a new (single) value to the proxy. For neural networks this will add the value to the startup table. See
-- `trainBatch` to train the neural networks.
insertProxyMany :: (MonadIO m) => AgentType -> Settings -> Period -> [[((StateFeatures, AgentActionIndices, RewardValue, IsRandomAction), Value)]] -> Proxy -> m Proxy
insertProxyMany _ _ p [] px = liftIO $ putStrLn ("\n\nEmpty input in insertProxyMany. Period: " ++ show p) >> return px
insertProxyMany _ _ _ !xs (Scalar _ nrAs) = return $ Scalar (unpackValue $ avg $ map (avg . map snd) xs) nrAs
insertProxyMany _ _ _ !xs (Table !m !def acts) = return $ Table m' def acts
  where
    trunc x = fromInteger (round $ x * (10 ^ n)) / (10.0 ^^ n)
    n = 3 :: Int
    m' = foldl' (\m' ((st, as, _, _), AgentValue vs) -> update m' st as vs) m (concat xs)
    update :: M.Map (StateFeatures, ActionIndex) (V.Vector Double) -> StateFeatures -> AgentActionIndices -> V.Vector Double -> M.Map (StateFeatures, ActionIndex) (V.Vector Double)
    update m st as vs = foldl' (\m' (idx, aNr, v) -> M.alter (\mOld -> Just $ fromMaybe def mOld V.// [(idx, v)]) (V.map trunc st, aNr) m') m (zip3 [0 ..V.length vs - 1] (VB.toList as) (V.toList vs))
insertProxyMany _ _ !period !xs px@(RegressionProxy nodes nrAs) = do
  let regLayer = addGroundTruthValueLayer nodes (concatMap makeObservations (concat xs))
  return $ set proxyRegressionLayer (trainRegressionLayer regLayer) px
  where
    makeObservations :: ((StateFeatures, AgentActionIndices, RewardValue, IsRandomAction), Value) -> [(Observation, ActionIndex)]
    makeObservations ((inps, aId, rew, isRand), AgentValue y)
      | VB.length aId > 1 = error "insertProxyMany: not yet implemented aId"
      | otherwise = [(obs, VB.head aId)]
      where
        obs
          | V.length y > 1 = error "insertProxyMany: not yet implemented for v>1"
          | otherwise = Observation period inps (y V.! 0)

insertProxyMany _ setts !period !xs px@(CombinedProxy !subPx !col !vs) -- only accumulate data if an update will follow
  | (1 + period) `mod` (setts ^. nStep) == 0 = return $ CombinedProxy subPx col (vs <> xs)
  | otherwise = return px
insertProxyMany _ setts period xs px
  | period < memSize = return $ addWelford period xs px
  where config = px ^?! proxyNNConfig
        memSize = config ^. replayMemoryMaxSize
insertProxyMany agent setts !period !xs !px
  | (1 + period) `mod` (setts ^. nStep) /= 0 || period < px ^?! proxyNNConfig . replayMemoryMaxSize = emptyCache >> updateNNTargetNet agent setts period (addWelford period xs px) -- skip ANN learning if not nStep or terminal
insertProxyMany agent setts !period !xs !px = emptyCache >> trainBatch period xs (addWelford period xs px) >>= updateNNTargetNet agent setts period

addWelford :: Period -> [[((StateFeatures, AgentActionIndices, RewardValue, IsRandomAction), Value)]] -> Proxy -> Proxy
addWelford period xs px
  --  | otherwise = proxyWelford .~ wel' $ px
  | period < max 100000 (3 * memSize) = proxyWelford .~ wel' $ px
  | otherwise = px
  where
    config = px ^?! proxyNNConfig
    memSize = config ^. replayMemoryMaxSize
    fst4 (x,_,_,_) = x
    wel = px ^?! proxyWelford
    wel'
      | config ^. autoNormaliseInput = foldl' addValue wel (concatMap (map (fst4 . fst)) xs)
      | otherwise = wel


insertCombinedProxies :: (MonadIO m) => AgentType -> Settings -> Period -> [Proxy] -> m Proxy
insertCombinedProxies !agent !setts !period !pxs = set proxyType (head pxs ^?! proxyType) <$!> insertProxyMany agent setts period combineProxyExpectedOuts pxLearn
  where
    pxLearn = set proxyType (NoScaling CombinedUnichain mMinMaxs) $ head pxs ^?! proxySub
    combineProxyExpectedOuts :: [[((StateFeatures, VB.Vector ActionIndex, RewardValue, IsRandomAction), Value)]]
    combineProxyExpectedOuts = concatMap getAndScaleExpectedOutput (sortBy (compare `on` (^?! proxyOutCol)) pxs)
    nrAs = head pxs ^?! proxyNrActions
    nrAgents = VB.generate (pxLearn ^?! proxyNrAgents) id
    mMinMaxs = mapM getMinMaxVal pxs
    scaleAlg = pxLearn ^?! proxyNNConfig . scaleOutputAlgorithm
    convertData px pxNr ((ft, curIdx, rew, isRand), out) =
      -- ((ft, VB.zipWith (\agNr idx -> columnMajorModeIndex nrPxs (agNr * nrAs + idx) pxNr) nrAgents curIdx), scaleValue scaleAlg (getMinMaxVal px) out)
      ((ft, VB.zipWith (\agNr idx -> pxNr * len + agNr * nrAs + idx) nrAgents curIdx, rew, isRand), scaleValue scaleAlg (getMinMaxVal px) out)
    getAndScaleExpectedOutput px@(CombinedProxy _ col outs) = map (map (convertData px col)) outs
    getAndScaleExpectedOutput px                            = error $ "unexpected proxy in insertCombinedProxies" ++ show px
    nrPxs = length pxs
    len = nrAs * (pxLearn ^?! proxyNrAgents)


-- | Copy the worker net to the target.
updateNNTargetNet :: (MonadIO m) => AgentType -> Settings -> Period -> Proxy -> m Proxy
updateNNTargetNet _ setts period px@(Grenade netT' netW' tp' config' nrActs agents wel)
  | period <= memSize = return px
  | (smoothUpd == 1 || smoothUpd == 0) && updatePeriod = return $ Grenade netW' netW' tp' config' nrActs agents wel
  | updatePeriod = return $ Grenade (((1 - toRational smoothUpd) |* netT') |+ (toRational smoothUpd |* netW') `using` rdeepseq) netW' tp' config' nrActs agents wel
  | otherwise = return px
  where
    memSize = px ^?! proxyNNConfig . replayMemoryMaxSize
    config = px ^?! proxyNNConfig
    smoothUpd = config ^. grenadeSmoothTargetUpdate
    smoothUpdPer = config ^. grenadeSmoothTargetUpdatePeriod
    updatePeriod = (period - memSize - 1) `mod` smoothUpdPer < setts ^. nStep
updateNNTargetNet _ setts period px@(Hasktorch netT netW tp config nrActs agents adam mdl wel)
  | period <= memSize = return px
  | (smoothUpd == 1 || smoothUpd == 0) && updatePeriod =
    let netT' = Torch.replaceParameters netT (Torch.flattenParameters netW)
     in return $ Hasktorch netT' netW tp config nrActs agents adam mdl wel
  | updatePeriod = do
    params' <-
      liftIO $
      zipWithM
        (\t w -> do
           let t' = Torch.toDependent t
           let w' = Torch.toDependent w
           Torch.makeIndependent $ Torch.mulScalar (1 - smoothUpd) t' `Torch.add` Torch.mulScalar smoothUpd w')
        (Torch.flattenParameters netT)
        (Torch.flattenParameters netW)
    let netT' = Torch.replaceParameters netT params'
    return $
      -- trace ("netT: " ++ show (Torch.flattenParameters netT))
      -- trace ("netW: " ++ show (Torch.flattenParameters netW))
      -- trace ("update: " ++ show params')
      Hasktorch netT' netW tp config nrActs agents adam mdl wel
  | otherwise = return px
  where
    memSize = px ^?! proxyNNConfig . replayMemoryMaxSize
    smoothUpd = config ^. grenadeSmoothTargetUpdate
    smoothUpdPer = config ^. grenadeSmoothTargetUpdatePeriod
    updatePeriod = (period - memSize - 1) `mod` smoothUpdPer < setts ^. nStep
updateNNTargetNet _ _ _ px = error $ show px ++ " proxy in updateNNTargetNet. Should not happen!"


-- | Train the neural network from a given batch. The training instances are Unscaled, that is in the range [-1, 1] or similar.
trainBatch :: forall m . (MonadIO m) => Period -> [[((StateFeatures, AgentActionIndices, RewardValue, IsRandomAction), Value)]] -> Proxy -> m Proxy
trainBatch !period !trainingInstances px@(Grenade !netT !netW !tp !config !nrActs !agents !wel) = do
  netW' <- liftIO $ trainGrenade period opt config netW trainingInstances'
  return $! Grenade netT netW' tp config nrActs agents wel
  where
    minMaxVal =
      case px ^?! proxyType of
        NoScaling _ (Just minMaxVals) -> Just (minV, maxV)
          where minV = minimum $ map fst minMaxVals
                maxV = maximum $ map snd minMaxVals
        _ -> getMinMaxVal px
    convertTrainingInstances :: (Int -> Int -> Int) -> ((StateFeatures, VB.Vector ActionIndex, RewardValue, IsRandomAction), Value) -> [((StateFeatures, ActionIndex), Double)]
    convertTrainingInstances idxFun ((ft, as, _, _), AgentValue vs)
      | config ^. autoNormaliseInput = zipWith3 (\agNr aIdx val -> ((normaliseStateFeature wel ft, idxFun agNr aIdx), val)) [0 .. VB.length as - 1] (VB.toList as) (V.toList vs)
      | otherwise = zipWith3 (\agNr aIdx val -> ((ft, idxFun agNr aIdx), val)) [0 .. VB.length as - 1] (VB.toList as) (V.toList vs)
    trainingInstances' =
      case px ^?! proxyType of
        NoScaling CombinedUnichain _ -> concatMap (map (convertTrainingInstances (\_ idx -> idx))) trainingInstances -- combined proxies (idx already calculated)
        NoScaling {}                 -> concatMap (map (convertTrainingInstances mkIdx)) trainingInstances
        _                            -> concatMap (map (convertTrainingInstances mkIdx . second (scaleValue scaleAlg minMaxVal))) trainingInstances -- single proxy
    mkIdx agNr aIdx = agNr * nrActs + aIdx
    lRate = getLearningRate (config ^. grenadeLearningParams)
    scaleAlg = config ^. scaleOutputAlgorithm
    dec = decaySetup (config ^. learningParamsDecay) period
    opt = setLearningRate (realToFrac $ dec $ realToFrac lRate) (config ^. grenadeLearningParams)
trainBatch !period !trainingInstances px@(Hasktorch !netT !netW !tp !config !nrActs !agents !adam !mdl !wel) = do
  (netW', adam') <- liftIO $ trainHasktorch period lRate adam config netW trainingInstances'
  return $! Hasktorch netT netW' tp config nrActs agents adam' mdl wel
  where
    minMaxVal =
      case px ^?! proxyType of
        NoScaling _ (Just minMaxVals) -> Just (minV, maxV)
          where minV = minimum $ map fst minMaxVals
                maxV = maximum $ map snd minMaxVals
        _ -> getMinMaxVal px
    convertTrainingInstances :: (Int -> Int -> Int) -> ((StateFeatures, VB.Vector ActionIndex, RewardValue, IsRandomAction), Value) -> [((StateFeatures, ActionIndex), Double)]
    convertTrainingInstances idxFun ((ft, as, _, _), AgentValue vs)
      | config ^. autoNormaliseInput = zipWith3 (\agNr aIdx val -> ((normaliseStateFeature wel ft, idxFun agNr aIdx), val)) [0 .. VB.length as - 1] (VB.toList as) (V.toList vs)
      | otherwise = zipWith3 (\agNr aIdx val -> ((ft, idxFun agNr aIdx), val)) [0 .. VB.length as - 1] (VB.toList as) (V.toList vs)
    trainingInstances' =
      case px ^?! proxyType of
        NoScaling CombinedUnichain _ -> concatMap (map (convertTrainingInstances (\_ idx -> idx))) trainingInstances -- combined proxies (idx already calculated)
        NoScaling {}                 -> concatMap (map (convertTrainingInstances mkIdx)) trainingInstances
        _                            -> concatMap (map (convertTrainingInstances mkIdx . second (scaleValue scaleAlg minMaxVal))) trainingInstances -- single proxy
    mkIdx agNr aIdx = agNr * nrActs + aIdx
    lRate0 = getLearningRate (config ^. grenadeLearningParams)
    lRate = realToFrac $ dec $ realToFrac lRate0
    scaleAlg = config ^. scaleOutputAlgorithm
    dec = decaySetup (config ^. learningParamsDecay) period
trainBatch _ _ _ = error "called trainBatch on non-neural network proxy (programming error)"


------------------------------ lookup ------------------------------

lookupProxyAgent :: (MonadIO m) => AgentType -> Period -> LookupType -> AgentNumber -> (StateFeatures, ActionIndex) -> Proxy -> m Double
lookupProxyAgent _ _ _ agNr _ (Scalar x _)    = return $ x V.! agNr
lookupProxyAgent _ _ _ agNr (k, a) (Table m def _) = return $ M.findWithDefault def (k, a) m V.! agNr
lookupProxyAgent agTp _ _ 0 (k, a) (RegressionProxy ms _) = return $
  -- trace ("lookupProxyAgent as: " ++ show a)
  applyRegressionLayer ms a (VB.convert k)
lookupProxyAgent agTp _ _ agNr (k, a) (RegressionProxy ms _) = error "RegressionProx ydoes not work with multiple agents"
lookupProxyAgent _ _ lkType agNr (k, a) px = selectIndex agNr <$> lookupNeuralNetwork lkType (k, VB.replicate agents a) px
  where
    agents = px ^?! proxyNrAgents


-- | Retrieve a value.
lookupProxy :: (MonadIO m) => AgentType -> Period -> LookupType -> (StateFeatures, AgentActionIndices) -> Proxy -> m Value
lookupProxy _ _ _ _ (Scalar x _)           = return $ AgentValue x
lookupProxy _ _ _ (k, ass) (Table m def _) = return $ AgentValue $ V.convert $ VB.zipWith (\a agNr -> M.findWithDefault def (k, a) m V.! agNr) ass (VB.generate (VB.length ass) id)
lookupProxy agTp _ _ (k, ass) (RegressionProxy ms _)
  | length ass > 1 = error "RegressionProxy does not work with multiple agents"
  | otherwise = return $ AgentValue $ V.convert $ VB.map (\a ->
                                                            -- trace ("lookupProxy (a, ass): " ++ show (a, ass))
                                                            applyRegressionLayer ms a (VB.convert k)) ass
lookupProxy _ _ lkType k px                = lookupNeuralNetwork lkType k px


-- | Retrieve a value, but do not unscale! For DEBUGGING only!
lookupProxyNoUnscale :: (MonadIO m) => AgentType -> Period -> LookupType -> (StateFeatures, AgentActionIndices) -> Proxy -> m Value
lookupProxyNoUnscale _ _ _ _ (Scalar x _)                = return $ AgentValue x
lookupProxyNoUnscale _ _ _ (k,ass) (Table m def _)       = return $ AgentValue $ V.convert $ VB.zipWith (\a agNr -> M.findWithDefault def (k,a) m V.! agNr) ass (VB.generate (VB.length ass) id)
lookupProxyNoUnscale agTp p lkType k l@RegressionProxy{} = lookupProxy agTp p lkType k l
lookupProxyNoUnscale _ _ lkType k px                     = lookupNeuralNetworkUnscaled lkType k px


-- | Retrieves all action values for the state but filters to the provided actions.
lookupState :: (MonadIO m) => AgentType -> LookupType -> (StateFeatures, DisallowedActionIndicies) -> Proxy -> m Values
lookupState _ _ (_, nass) (Scalar x nrAs) = return $ AgentValues $ VB.zipWith (\agentValue as -> V.replicate (V.length as) agentValue) (V.convert x) ass
  where
    ass = toPositiveActionList nrAs nass
lookupState _ _ (k, nass) (Table m def nrAs) =
  return $ AgentValues $ VB.zipWith (\as agNr -> V.map (\a -> M.findWithDefault def (k, a) m V.! agNr) as) ass (VB.fromList [0 .. VB.length ass - 1])
  where
    ass = toPositiveActionList nrAs nass
lookupState agTp _ (k, nass) (RegressionProxy ms nrAs) =
  return $ AgentValues $ VB.zipWith (\as agNr ->
                                       -- trace ("lookupState as: " ++ show as)
                                       V.map (\a -> applyRegressionLayer ms a k) as) ass (VB.fromList [0 .. VB.length ass - 1])
  where
    ass = toPositiveActionList nrAs nass
lookupState _ tp (k, DisallowedActionIndicies ass) px = do
  AgentValues vals <- lookupActionsNeuralNetwork tp k px
  return $ AgentValues $ VB.zipWith filterActions ass vals
  where
    filterActions :: V.Vector ActionIndex -> V.Vector Double -> V.Vector Double
    filterActions as vs =
      dropAs 0 as vs
      -- if V.null nas
      --   then vs -- no need to filter
      --   else V.map (vs V.!) as
    dropAs idx ns xs
      | V.null ns = xs
      | V.null xs = V.empty
      | V.head ns == idx = dropAs (idx+1) (V.tail ns) (V.tail xs)
      | otherwise = V.head xs `V.cons` dropAs (idx+1) ns (V.tail xs)


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
lookupNeuralNetworkUnscaled !tp (!st, !actIdx) px@Grenade{}           = selectIndices actIdx <$> lookupActionsNeuralNetworkUnscaled tp st px
lookupNeuralNetworkUnscaled !tp (!st, !actIdx) px@Hasktorch{}         = selectIndices actIdx <$> lookupActionsNeuralNetworkUnscaled tp st px
lookupNeuralNetworkUnscaled !tp (!st, !actIdx) (CombinedProxy px _ _) = lookupNeuralNetworkUnscaled tp (st, actIdx) px
lookupNeuralNetworkUnscaled _ _ _                                     = error "lookupNeuralNetworkUnscaled called on non-neural network proxy"


-- | Retrieve all action values of a state from a neural network proxy. For other proxies an error is thrown.
lookupActionsNeuralNetworkUnscaled :: (MonadIO m) => LookupType -> StateFeatures -> Proxy -> m Values
lookupActionsNeuralNetworkUnscaled tp st px@(Grenade _ _ pxTp config _ _ _) = head <$> cached (tp', pxTp, st) (lookupActionsNeuralNetworkUnscaledFull tp' st px)
  where
    tp' = mkLookupType config tp
lookupActionsNeuralNetworkUnscaled tp st px@(Hasktorch _ _ pxTp config _ _ _ _ _) = head <$> cached (tp', pxTp, st) (lookupActionsNeuralNetworkUnscaledFull tp' st px)
  where
    tp' = mkLookupType config tp
lookupActionsNeuralNetworkUnscaled tp st (CombinedProxy px nr _) = (!! nr) <$> cached (tp', CombinedUnichain, st) (lookupActionsNeuralNetworkUnscaledFull tp' st px)
  where
    tp' = mkLookupType (px ^?! proxyNNConfig) tp
lookupActionsNeuralNetworkUnscaled _ _ _ = error "lookupNeuralNetworkUnscaled called on non-neural network proxy"

maybeWelford :: NNConfig -> WelfordExistingAggregate StateFeatures -> Maybe (WelfordExistingAggregate StateFeatures)
maybeWelford config wel
  | config ^. autoNormaliseInput = Just wel
  | otherwise = Nothing


-- | Retrieve all action values of a state from a neural network proxy. For other proxies an error is thrown.
lookupActionsNeuralNetworkUnscaledFull :: (MonadIO m) => LookupType -> StateFeatures -> Proxy -> m [Values]
lookupActionsNeuralNetworkUnscaledFull Worker st (Grenade _ netW _ cfg nrAs agents wel)         = return $ runGrenade netW nrAs agents   (maybeWelford cfg wel) st
lookupActionsNeuralNetworkUnscaledFull Target st (Grenade netT _ _ cfg nrAs agents wel)         = return $ runGrenade netT nrAs agents   (maybeWelford cfg wel) st
lookupActionsNeuralNetworkUnscaledFull Worker st (Hasktorch _ netW _ cfg nrAs agents _ mdl wel) = return $ runHasktorch netW nrAs agents (maybeWelford cfg wel) st
lookupActionsNeuralNetworkUnscaledFull Target st (Hasktorch netT _ _ cfg nrAs agents _ mdl wel) = return $ runHasktorch netT nrAs agents (maybeWelford cfg wel) st
lookupActionsNeuralNetworkUnscaledFull _ _ CombinedProxy{}                                      = error "lookupActionsNeuralNetworkUnscaledFull called on CombinedProxy"
lookupActionsNeuralNetworkUnscaledFull _ _ _                                                    = error "lookupActionsNeuralNetworkUnscaledFull called on a non-neural network proxy"


mkLookupType :: NNConfig -> LookupType -> LookupType
mkLookupType config lp
  | onlyUseWorker config = Worker
  | otherwise = lp

onlyUseWorker :: NNConfig -> Bool
onlyUseWorker config = config ^. grenadeSmoothTargetUpdate == 1 && config ^. grenadeSmoothTargetUpdatePeriod <= 1


-- -- | Retrieve all action values of a state from a neural network proxy. For other proxies an error is thrown.
-- lookupActionsNeuralNetworkUnscaled :: (MonadIO m) => LookupType -> StateFeatures -> Proxy -> m Values
-- lookupActionsNeuralNetworkUnscaled tp st px@(Grenade _ _ pxTp _ _ _) = head <$> cached (fromLookupType tp, pxTp, st) (lookupActionsNeuralNetworkUnscaledFull tp st px)
-- lookupActionsNeuralNetworkUnscaled tp st (CombinedProxy px nr _) = (!! nr) <$> cached (CacheCombined, CombinedUnichain, st) (lookupActionsNeuralNetworkUnscaledFull tp st px)
-- lookupActionsNeuralNetworkUnscaled _ _ _ = error "lookupNeuralNetworkUnscaled called on non-neural network proxy"


-- -- | Retrieve all action values of a state from a neural network proxy. For other proxies an error is thrown.
-- lookupActionsNeuralNetworkUnscaledFull :: (MonadIO m) => LookupType -> StateFeatures -> Proxy -> m [Values]
-- lookupActionsNeuralNetworkUnscaledFull Worker st (Grenade _ netW _ _ nrAs agents) = return $ runGrenade netW nrAs agents st
-- lookupActionsNeuralNetworkUnscaledFull Target st px@(Grenade netT _ _ config nrAs agents)
--   | config ^. grenadeSmoothTargetUpdate == 1 && config ^. grenadeSmoothTargetUpdatePeriod <= 1 = lookupActionsNeuralNetworkUnscaledFull Worker st px
--   | otherwise = return $ runGrenade netT nrAs agents st
-- lookupActionsNeuralNetworkUnscaledFull _ _ CombinedProxy{} = error "lookupActionsNeuralNetworkUnscaledFull called on CombinedProxy"
-- lookupActionsNeuralNetworkUnscaledFull _ _ _ = error "lookupActionsNeuralNetworkUnscaledFull called on a non-neural network proxy"


------------------------------ Helpers ------------------------------

hasLocked :: String -> IO a -> IO a
hasLocked msg action =
  action `catches`
  [ Handler $ \exc@BlockedIndefinitelyOnMVar -> sayString ("[MVar]: " ++ msg) >> throwIO exc
  , Handler $ \exc@BlockedIndefinitelyOnSTM -> sayString ("[STM]: " ++ msg) >> throwIO exc
  ]


-- | Caching of results
type CacheKey = (LookupType, ProxyType, StateFeatures)

cacheMVar :: MVar (M.Map CacheKey [Values])
cacheMVar = unsafePerformIO $ newMVar mempty
{-# NOINLINE cacheMVar #-}

emptyCache :: MonadIO m => m ()
emptyCache = liftIO $ hasLocked "emptyCache" $ modifyMVar_ cacheMVar (const mempty)

addCache :: (MonadIO m) => CacheKey -> [Values] -> m ()
addCache k val = liftIO $ hasLocked "addCache" $ modifyMVar_ cacheMVar (return . M.insert k val)

lookupCache :: (MonadIO m) => CacheKey -> m (Maybe [Values])
lookupCache k = liftIO $ hasLocked "lookupCache" $ (M.lookup k =<<) <$> tryReadMVar cacheMVar

-- | Get output of function f, if possible from cache according to key (st).
cached :: (MonadIO m) => CacheKey -> m [Values] -> m [Values]
cached st ~f = do
  c <- lookupCache st
  case c of
    Nothing -> do
      res <- f
      res `seq` addCache st res
      return res
    Just res -> do
      return res


-- | Finds the correct value for scaling.
getMinMaxVal :: Proxy -> Maybe (MinValue Double, MaxValue Double)
getMinMaxVal Table{} = error "getMinMaxVal called for Table"
getMinMaxVal !p =
  case unCombine (p ^?! proxyType) of
    VTable           -> Just (p ^?! proxyNNConfig . scaleParameters . scaleMinVValue, p ^?! proxyNNConfig . scaleParameters . scaleMaxVValue)
    WTable           -> Just (p ^?! proxyNNConfig . scaleParameters . scaleMinWValue, p ^?! proxyNNConfig . scaleParameters . scaleMaxWValue)
    R0Table          -> Just (p ^?! proxyNNConfig . scaleParameters . scaleMinR0Value, p ^?! proxyNNConfig . scaleParameters . scaleMaxR0Value)
    R1Table          -> Just (p ^?! proxyNNConfig . scaleParameters . scaleMinR1Value, p ^?! proxyNNConfig . scaleParameters . scaleMaxR1Value)
    PsiVTable        -> Just (1.0 * p ^?! proxyNNConfig . scaleParameters . scaleMinVValue, 1.0 * p ^?! proxyNNConfig . scaleParameters . scaleMaxVValue)
    PsiWTable        -> Just (1.0 * p ^?! proxyNNConfig . scaleParameters . scaleMinVValue, 1.0 * p ^?! proxyNNConfig . scaleParameters . scaleMaxVValue)
    NoScaling {}     -> Nothing
    CombinedUnichain -> error "should not happend"
    -- CombinedUnichainScaleAs {} -> error "should not happend"
  where
    unCombine CombinedUnichain
      | isCombinedProxy p = fromCombinedIndex (p ^?! proxyOutCol)
    -- unCombine (CombinedUnichainScaleAs x)
    --   | isCombinedProxy p = x
    unCombine x = x


-- | This function retrieves the data and builds a table like return value.
mkNNList :: (MonadIO m) => ARAL s as -> Bool -> Proxy -> m [(NetInputWoAction, ([(ActionIndex, Value)], [(ActionIndex, Value)]))]
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
       return (normaliseStateFeature wel st, (zip actIdxs (toActionValue target), zip actIdxs (toActionValue worker))))
       -- return (st, (replicate agents actIdxs))
    (conf ^. prettyPrintElems)
  where
    conf = pr ^?! proxyNNConfig
    wel
      | conf ^. autoNormaliseInput =
        case pr of
          Hasktorch {} -> _proxyHTWelford pr
          Grenade {}   -> _proxyNNWelford pr
          _            -> WelfordExistingAggregateEmpty
      | otherwise = WelfordExistingAggregateEmpty
    agents = pr ^?! proxyNrAgents
    actIdxs = [0 .. (nrActs - 1)]
    nrActs = pr ^?! proxyNrActions
