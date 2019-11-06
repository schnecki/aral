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
    , getMinMaxVal
    , mkNNList
    , StateFeatures
    , StateNextFeatures
    ,
    ) where


import           ML.BORL.Calculation.Type
import           ML.BORL.Fork
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import           ML.BORL.Proxy.Proxies
import           ML.BORL.Proxy.Type
import           ML.BORL.Reward
import           ML.BORL.Type
import           ML.BORL.Types                as T
import           ML.BORL.Types

import           Control.Arrow
import           Control.DeepSeq
import           Control.Lens
import           Control.Monad
import           Control.Parallel.Strategies  hiding (r0)
import           Data.List                    (foldl')
import qualified Data.Map.Strict              as M
import           Data.Maybe                   (fromJust, isNothing)
import qualified Data.Set                     as S
import           Data.Singletons.Prelude.List
import           GHC.Generics
import           GHC.TypeLits
import           Grenade
import           System.Random.Shuffle


import           Control.Lens
import qualified Data.Map.Strict              as M

mkStateActs borl state stateNext = (stateFeat, stateActs, stateNextActs)
    where
    sActIdxes = map fst $ actionsIndexed borl state
    sNextActIdxes = map fst $ actionsIndexed borl stateNext
    stateFeat = (borl ^. featureExtractor) state
    stateNextFeat = (borl ^. featureExtractor) stateNext
    stateActs = (stateFeat, sActIdxes)
    stateNextActs = (stateNextFeat, sNextActIdxes)


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
insert borl period state aNr randAct rew stateNext episodeEnd getCalc pxs@(Proxies pRhoMin pRho pPsiV pV pPsiW pW pPsiW2 pW2 pR0 pR1 Nothing) = do
  calc <- getCalc stateActs aNr randAct rew stateNextActs episodeEnd
  -- forkMv' <- liftSimple $ doFork $ P.insert period label vValStateNew mv
  -- mv' <- liftSimple $ collectForkResult forkMv'
  if borl ^. parameters.disableAllLearning
    then return (pxs, calc)
    else do
    let mInsertProxy mVal px = maybe (return px) (\val -> insertProxy period stateFeat aNr val px) mVal
    pRhoMin' <- mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
    pRho' <- mInsertProxy (getRhoVal' calc) pRho `using` rpar
    pV' <- mInsertProxy (getVValState' calc) pV `using` rpar
    pW' <- mInsertProxy (getWValState' calc) pW `using` rpar
    pW2' <- mInsertProxy (getW2ValState' calc) pW2 `using` rpar
    pPsiV' <- mInsertProxy (getPsiVValState' calc) pPsiV `using` rpar
    pPsiW' <- mInsertProxy (getPsiWValState' calc) pPsiW `using` rpar
    pPsiW2' <- mInsertProxy (getPsiW2ValState' calc) pPsiW2 `using` rpar
    pR0' <- mInsertProxy (getR0ValState' calc) pR0 `using` rpar
    pR1' <- mInsertProxy (getR1ValState' calc) pR1 `using` rpar
    return (Proxies pRhoMin' pRho' pPsiV' pV' pPsiW' pW' pPsiW2' pW2' pR0' pR1' Nothing, calc)
  where
    (stateFeat, stateActs, stateNextActs) = mkStateActs borl state stateNext
insert borl period state aNr randAct rew stateNext episodeEnd getCalc pxs@(Proxies pRhoMin pRho pPsiV pV pPsiW pW pPsiW2 pW2 pR0 pR1 (Just replMem))
  | pV ^?! proxyNNConfig . replayMemoryMaxSize == 1 = insert borl period state aNr randAct rew stateNext episodeEnd getCalc (Proxies pRhoMin pRho pPsiV pV pPsiW pW pPsiW2 pW2 pR0 pR1 Nothing)
  | period <= fromIntegral (replMem ^. replayMemorySize) - 1 = do
    replMem' <- liftSimple $ addToReplayMemory period (stateActs, aNr, randAct, rew, stateNextActs, episodeEnd) replMem
    (pxs', calc) <- insert borl period state aNr randAct rew stateNext episodeEnd getCalc (replayMemory .~ Nothing $ pxs)
    return (replayMemory ?~ replMem' $ pxs', calc)
  | otherwise = do
    replMem' <- liftSimple $ addToReplayMemory period (stateActs, aNr, randAct, rew, stateNextActs, episodeEnd) replMem
    calc <- getCalc stateActs aNr randAct rew stateNextActs episodeEnd
    if borl ^. parameters . disableAllLearning
      then return (set replayMemory (Just replMem') pxs, calc)
      else do
        let config = pV ^?! proxyNNConfig
        mems <- liftSimple $ getRandomReplayMemoryElements (config ^. trainBatchSize) replMem'
        let mkCalc (s, idx, rand, rew, s', epiEnd) = getCalc s idx rand rew s' epiEnd
        calcs <- parMap rdeepseq force <$> mapM (\m@((s, _), idx, _, _, _, _) -> mkCalc m >>= \v -> return ((s, idx), v)) mems
        let mInsertProxy mVal px = maybe (return px) (\val -> insertProxy period stateFeat aNr val px) mVal
        let mTrainBatch accessor calcs px =
              maybe
                (return px)
                (\xs -> insertProxyMany period xs px)
                (mapM
                   (\c ->
                      let (inp, mOut) = second accessor c
                       in mOut >>= \out -> Just (inp, out))
                   calcs)
        pRhoMin' <-
          if isNeuralNetwork pRhoMin
            then mTrainBatch getRhoMinimumVal' calcs pRhoMin `using` rpar
            else mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
        pRho' <-
          if isNeuralNetwork pRho
            then mTrainBatch getRhoVal' calcs pRho `using` rpar
            else mInsertProxy (getRhoVal' calc) pRho `using` rpar
        pV' <- mTrainBatch getVValState' calcs pV `using` rpar
        pW' <- mTrainBatch getWValState' calcs pW `using` rpar
        pW2' <- mTrainBatch getW2ValState' calcs pW2 `using` rpar
        pPsiV' <- mTrainBatch getPsiVValState' calcs pPsiV `using` rpar
        pPsiW' <- mTrainBatch getPsiWValState' calcs pPsiW `using` rpar
        pPsiW2' <- mTrainBatch getPsiW2ValState' calcs pPsiW2 `using` rpar
        pR0' <- mTrainBatch getR0ValState' calcs pR0 `using` rpar
        pR1' <- mTrainBatch getR1ValState' calcs pR1 `using` rpar
        return (Proxies pRhoMin' pRho' pPsiV' pV' pPsiW' pW' pPsiW2' pW2' pR0' pR1' (Just replMem'), calc)
  where
    (stateFeat, stateActs, stateNextActs) = mkStateActs borl state stateNext
insert borl period state aNr randAct rew stateNext episodeEnd getCalc pxs@(ProxiesCombinedUnichain pRhoMin pRho proxy Nothing) = do
  calc <- getCalc stateActs aNr randAct rew stateNextActs episodeEnd
  if borl ^. parameters . disableAllLearning
    then return (pxs, calc)
    else do
      let mInsertProxy mVal px = maybe (return px) (\val -> insertProxy period stateFeat aNr val px) mVal
      pRhoMin' <- mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
      pRho' <- mInsertProxy (getRhoVal' calc) pRho `using` rpar
      pV' <- mInsertProxy (getVValState' calc) (pxs ^. v) `using` rpar
      pW' <- mInsertProxy (getWValState' calc) (pxs ^. w) `using` rpar
      pW2' <- mInsertProxy (getW2ValState' calc) (pxs ^. w2) `using` rpar
      pPsiV' <- mInsertProxy (getPsiVValState' calc) (pxs ^. psiV) `using` rpar
      pPsiW' <- mInsertProxy (getPsiWValState' calc) (pxs ^. psiW) `using` rpar
      pPsiW2' <- mInsertProxy (getPsiW2ValState' calc) (pxs ^. psiW2) `using` rpar
      pR0' <- mInsertProxy (getR0ValState' calc) (pxs ^. r0) `using` rpar
      pR1' <- mInsertProxy (getR1ValState' calc) (pxs ^. r1) `using` rpar
      proxy' <- insertCombinedProxies period [pPsiV', pV', pPsiW', pW', pPsiW2', pW2', pR0', pR1']
      return (ProxiesCombinedUnichain pRhoMin' pRho' proxy' Nothing, calc)
  where
    (stateFeat, stateActs, stateNextActs) = mkStateActs borl state stateNext
insert borl period state aNr randAct rew stateNext episodeEnd getCalc pxs@(ProxiesCombinedUnichain pRhoMin pRho proxy (Just replMem))
  | proxy ^?! proxyNNConfig . replayMemoryMaxSize == 1 = insert borl period state aNr randAct rew stateNext episodeEnd getCalc (ProxiesCombinedUnichain pRhoMin pRho proxy Nothing)
  | period <= fromIntegral (replMem ^. replayMemorySize) - 1 = do
    replMem' <- liftSimple $ addToReplayMemory period (stateActs, aNr, randAct, rew, stateNextActs, episodeEnd) replMem
    (pxs', calc) <- insert borl period state aNr randAct rew stateNext episodeEnd getCalc (replayMemory .~ Nothing $ pxs)
    return (replayMemory ?~ replMem' $ pxs', calc)
  | otherwise = do
      undefined
  where
    (stateFeat, stateActs, stateNextActs) = mkStateActs borl state stateNext


-- | Insert a new (single) value to the proxy. For neural networks this will add the value to the startup table. See
-- `trainBatch` to train the neural networks.
insertProxy :: (MonadBorl' m) => Period -> StateFeatures -> ActionIndex -> Double -> Proxy -> m Proxy
insertProxy p st aNr v px = insertProxyMany p [((st,aNr), v)] px

-- | Insert a new (single) value to the proxy. For neural networks this will add the value to the startup table. See
-- `trainBatch` to train the neural networks.
insertProxyMany :: (MonadBorl' m) => Period -> [((StateFeatures, ActionIndex), Double)] -> Proxy -> m Proxy
insertProxyMany _ xs (Scalar _) = return $ Scalar (snd $ last xs)
insertProxyMany _ xs (Table m def) = return $ Table (foldl' (\m' ((st,aNr),v') -> M.insert (map trunc st, aNr) v' m') m xs) def
  where trunc x = (fromInteger $ round $ x * (10^n)) / (10.0^^n)
        n = 3
insertProxyMany _ xs (CombinedProxy subPx col vs) = return $ CombinedProxy subPx col (vs <> xs)
insertProxyMany period xs px
  | period < memSize - 1 && isNothing (px ^?! proxyNNConfig . trainMSEMax) = return px
  | period < memSize - 1 = return $ proxyNNStartup .~ foldl' (\m ((st, aNr), v) -> M.insert (st, aNr) v m) tab xs $ px
  | period == memSize - 1 && (isNothing (px ^?! proxyNNConfig . trainMSEMax) || px ^?! proxyNNConfig . replayMemoryMaxSize == 1) = updateNNTargetNet False period px
  | period == memSize - 1 = liftSimple (putStrLn $ "Initializing artificial neural networks: " ++ show (px ^? proxyType)) >> netInit px >>= updateNNTargetNet True period
  | otherwise = trainBatch xs px >>= updateNNTargetNet False period
  where
    netInit = trainMSE (Just 0) (M.toList tab) (config ^. grenadeLearningParams)
    config = px ^?! proxyNNConfig
    tab = px ^?! proxyNNStartup
    memSize = fromIntegral (px ^?! proxyNNConfig . replayMemoryMaxSize)


insertCombinedProxies :: (MonadBorl' m) => Period -> [Proxy] -> m Proxy
insertCombinedProxies period pxs = insertProxyMany period combineProxyExpectedOuts (head pxs ^?! proxySub)
  where combineProxyExpectedOuts = concatMap (\(CombinedProxy _ idx outs) -> map (\((ft, curIdx), out) -> ((ft, idx*len + curIdx), out)) outs) pxs
        len = length pxs


-- | Copy the worker net to the target.
updateNNTargetNet :: (MonadBorl' m) => Bool -> Period -> Proxy -> m Proxy
updateNNTargetNet _ _ px | not (isNeuralNetwork px) = error "updateNNTargetNet called on non-neural network proxy"
updateNNTargetNet forceReset period px
  | forceReset || config ^. trainBatchSize <= 1 = copyValues
  | period <= memSize = return px
  | ((period - memSize - 1) `mod` config ^. updateTargetInterval) == 0 = copyValues
  | otherwise = return px
  where
    memSize = px ^?! proxyNNConfig . replayMemoryMaxSize
    config = px ^?! proxyNNConfig
    copyValues =
      case px of
        (Grenade _ netW' tab' tp' config' nrActs) -> return $ Grenade netW' netW' tab' tp' config' nrActs
        (TensorflowProxy netT' netW' tab' tp' config' nrActs) -> do
          copyValuesFromTo netW' netT'
          return $ TensorflowProxy netT' netW' tab' tp' config' nrActs
        Table {} -> error "not possible"
        Scalar {} -> error "not possible"


-- | Train the neural network from a given batch. The training instances are Unscaled.
trainBatch :: forall m . (MonadBorl' m) => [((StateFeatures, ActionIndex), Double)] -> Proxy -> m Proxy
trainBatch trainingInstances px@(Grenade netT netW tab tp config nrActs) = do
  let netW' = foldl' (trainGrenade (config ^. grenadeLearningParams)) netW (map return trainingInstances')
  return $ Grenade netT netW' tab tp config nrActs
  where trainingInstances' = map (second $ scaleValue (getMinMaxVal px)) trainingInstances
trainBatch trainingInstances px@(TensorflowProxy netT netW tab tp config nrActs) = do
  backwardRunRepMemData netW trainingInstances'
  return $ TensorflowProxy netT netW tab tp config nrActs
  where trainingInstances' = map (second $ scaleValue (getMinMaxVal px)) trainingInstances

trainBatch _ _ = error "called trainBatch on non-neural network proxy (programming error)"


-- | Train until MSE hits the value given in NNConfig.
trainMSE :: (MonadBorl' m) => Maybe Int -> [(([Double], ActionIndex), Double)] -> LearningParameters -> Proxy -> m Proxy
trainMSE _ _ _ px@Table{} = return px
trainMSE mPeriod dataset lp px@(Grenade _ netW tab tp config nrActs)
  | isNothing (config ^. trainMSEMax) = return px
  | mse < mseMax = do
    liftSimple $ putStrLn $ "Final MSE for " ++ show tp ++ ": " ++ show mse
    return px
  | otherwise = do
    datasetShuffled <- liftSimple $ shuffleM dataset
    let net' = foldl' (trainGrenade lp) netW (zipWith (curry return) (map fst datasetShuffled) (map snd datasetShuffled))
    when (maybe False ((== 0) . (`mod` 5)) mPeriod) $ liftSimple $ putStrLn $ "Current MSE for " ++ show tp ++ ": " ++ show mse
    fmap force <$> trainMSE ((+ 1) <$> mPeriod) dataset lp $ Grenade net' net' tab tp config nrActs
  where
    mseMax = fromJust (config ^. trainMSEMax)
    -- net' = trainGrenade lp netSGD (zip kScaled vScaled)
    vScaled = map (scaleValue (getMinMaxVal px) . snd) dataset
    vUnscaled = map snd dataset
    kScaled = map fst dataset
    (minV,maxV) = getMinMaxVal px
    getValue k =
       -- unscaleValue (getMinMaxVal px) $
       (!! snd k) $ snd $ fromLastShapes netW $ runNetwork netW ((toHeadShapes netW . fst) k)
    mse = 1 / fromIntegral (length dataset) * sum (zipWith (\k v -> (v - getValue k) ** 2) (map fst dataset) (map (min 1 . max (-1)) vScaled)) -- scaled or unscaled ones?
trainMSE mPeriod dataset lp px@(TensorflowProxy netT netW tab tp config nrActs)
  | isNothing (config ^. trainMSEMax) = return px
  | otherwise = do
    datasetShuffled <- liftSimple $ shuffleM dataset
    let mseMax = fromJust (config ^. trainMSEMax)
        kFullScaled = map (first (map realToFrac) . fst) dataset :: [([Float], ActionIndex)]
        kScaled = map fst kFullScaled
        actIdxs = map snd kFullScaled
        (minV,maxV) = getMinMaxVal px
        vScaledDbl = map (scaleValue (getMinMaxVal px) . snd) datasetShuffled
        vScaled = map realToFrac vScaledDbl
        vUnscaled = map (realToFrac . snd) dataset
        datasetRepMem = map (first (first (map realToFrac))) dataset
    current <- forwardRun netW kScaled
    zipWithM_ (backwardRun netW) (map return kScaled) (map return $ zipWith3 replace actIdxs vScaled current)
    -- backwardRunRepMemData netW datasetRepMem
    let forward k = realToFrac <$> lookupNeuralNetwork Worker k px -- lookupNeuralNetworkUnscaled Worker k px -- scaled or unscaled ones?
    mse <- (1 / fromIntegral (length dataset) *) . sum <$> zipWithM (\k vU -> (** 2) . (vU -) <$> forward k) (map fst dataset) (map (min 1 . max (-1)) vScaledDbl) -- scaled or unscaled ones?
    if (realToFrac mse < mseMax)
      then do
      liftSimple $ putStrLn ("Final MSE for " ++ show tp ++ ": " ++ show mse)
      void $ saveModelWithLastIO netW -- Save model to ensure correct values when reading from another session
      return px
      else do
      when (maybe False ((== 0) . (`mod` 5)) mPeriod) $ do
        liftSimple $ putStrLn $ "Current MSE for " ++ show tp ++ ": " ++ show mse
      trainMSE ((+ 1) <$> mPeriod) dataset lp (TensorflowProxy netT netW tab tp config nrActs)
trainMSE _ _ _ _ = error "trainMSE should not have been callable with this type of proxy. programming error!"

-- | Retrieve a value.
lookupProxy :: (MonadBorl' m) => Period -> LookupType -> (StateFeatures, ActionIndex) -> Proxy -> m Double
lookupProxy _ _ _ (Scalar x) = return x
lookupProxy _ _ k (Table m def) = return $ M.findWithDefault def k m
lookupProxy period lkType k@(_, aNr) px
  | period <= fromIntegral (config ^. replayMemoryMaxSize) && (config ^. trainBatchSize) /= 1 = return $ M.findWithDefault 0 k tab
  | otherwise = lookupNeuralNetwork lkType k px
  where
    config = px ^?! proxyNNConfig
    tab = px ^?! proxyNNStartup


-- | Retrieve a value from a neural network proxy. The output is sclaed to the original range. For other proxies an
-- error is thrown. The returned value is up-scaled to the original interval before returned.
lookupNeuralNetwork :: (MonadBorl' m) => LookupType -> (StateFeatures, ActionIndex) -> Proxy -> m Double
lookupNeuralNetwork tp k px@Grenade {} = unscaleValue (getMinMaxVal px) <$> lookupNeuralNetworkUnscaled tp k px
lookupNeuralNetwork tp k px@TensorflowProxy {} = unscaleValue (getMinMaxVal px) <$> lookupNeuralNetworkUnscaled tp k px
lookupNeuralNetwork _ _ _ = error "lookupNeuralNetwork called on non-neural network proxy"

-- | Retrieve all values of one feature from a neural network proxy. The output is sclaed to the original range. For
-- other proxies an error is thrown. The returned value is up-scaled to the original interval before returned.
lookupActionsNeuralNetwork :: (MonadBorl' m) => LookupType -> StateFeatures -> Proxy -> m [Double]
lookupActionsNeuralNetwork tp k px@Grenade {} = map (unscaleValue (getMinMaxVal px)) <$> lookupActionsNeuralNetworkUnscaled tp k px
lookupActionsNeuralNetwork tp k px@TensorflowProxy {} = map (unscaleValue (getMinMaxVal px)) <$> lookupActionsNeuralNetworkUnscaled tp k px
lookupActionsNeuralNetwork _ _ _ = error "lookupNeuralNetwork called on non-neural network proxy"


-- | Retrieve a value from a neural network proxy. The output is *not* scaled to the original range. For other proxies
-- an error is thrown.
lookupNeuralNetworkUnscaled :: (MonadBorl' m) => LookupType -> (StateFeatures, ActionIndex) -> Proxy -> m Double
lookupNeuralNetworkUnscaled Worker (st, actIdx) (Grenade _ netW _ _ conf _) = return $ (!! actIdx) $ snd $ fromLastShapes netW $ runNetwork netW (toHeadShapes netW st)
lookupNeuralNetworkUnscaled Target (st, actIdx) (Grenade netT _ _ _ conf _) = return $ (!!actIdx) $ snd $ fromLastShapes netT $ runNetwork netT (toHeadShapes netT st)
lookupNeuralNetworkUnscaled Worker (st, actIdx) (TensorflowProxy _ netW _ _ conf _) = realToFrac . (!!actIdx) . headLookup <$> forwardRun netW [map realToFrac st]
lookupNeuralNetworkUnscaled Target (st, actIdx) (TensorflowProxy netT _ _ _ conf _) = realToFrac . (!!actIdx) . headLookup <$> forwardRun netT [map realToFrac st]
lookupNeuralNetworkUnscaled _ _ _ = error "lookupNeuralNetworkUnscaled called on non-neural network proxy"

headLookup []    = error "head: empty input data in lookupNeuralNetworkUnscaled"
headLookup (x:_) = x
headLookupActions []    = error "head: empty input data in lookupActionsNeuralNetworkUnscaled"
headLookupActions (x:_) = x


-- | Retrieve all action values of a state from a neural network proxy. For other proxies an error is thrown.
lookupActionsNeuralNetworkUnscaled :: (MonadBorl' m) => forall k . LookupType -> [Double] -> Proxy -> m [Double]
lookupActionsNeuralNetworkUnscaled Worker st (Grenade _ netW _ _ conf _) = return $ snd $ fromLastShapes netW $ runNetwork netW (toHeadShapes netW st)
lookupActionsNeuralNetworkUnscaled Target st (Grenade netT _ _ _ conf _) = return $ snd $ fromLastShapes netT $ runNetwork netT (toHeadShapes netT st)
lookupActionsNeuralNetworkUnscaled Worker st (TensorflowProxy _ netW _ _ conf _) = map realToFrac . headLookupActions <$> forwardRun netW [map realToFrac st]
lookupActionsNeuralNetworkUnscaled Target st (TensorflowProxy netT _ _ _ conf _) = map realToFrac . headLookupActions <$> forwardRun netT [map realToFrac st]
lookupActionsNeuralNetworkUnscaled _ _ _ = error "lookupNeuralNetworkUnscaled called on non-neural network proxy"


-- | Finds the correct value for scaling.
getMinMaxVal :: Proxy -> (MinValue,MaxValue)
getMinMaxVal Table{} = error "getMinMaxVal called for Table"
getMinMaxVal p  = case p ^?! proxyType of
  VTable  -> (p ^?! proxyNNConfig.scaleParameters.scaleMinVValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxVValue)
  WTable  -> (p ^?! proxyNNConfig.scaleParameters.scaleMinWValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxWValue)
  W2Table  -> (2*p ^?! proxyNNConfig.scaleParameters.scaleMinWValue, 2*p ^?! proxyNNConfig.scaleParameters.scaleMaxWValue)
  R0Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinR0Value, p ^?! proxyNNConfig.scaleParameters.scaleMaxR0Value)
  R1Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinR1Value, p ^?! proxyNNConfig.scaleParameters.scaleMaxR1Value)
  PsiVTable -> (p ^?! proxyNNConfig.scaleParameters.scaleMinVValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxVValue)
  PsiWTable -> (p ^?! proxyNNConfig.scaleParameters.scaleMinVValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxVValue)
  PsiW2Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinVValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxVValue)


-- | This function loads the model from the checkpoint file and finds then retrieves the data.
mkNNList :: (MonadBorl' m) => BORL k -> Bool -> Proxy -> m [(NetInput, ([(ActionIndex, Double)], [(ActionIndex, Double)]))]
mkNNList borl scaled pr =
  mapM
    (\st -> do
       t <-
         if useTable
           then return $ lookupTable scaled st
           else if scaled
                  then lookupActionsNeuralNetwork Target st pr
                  else lookupActionsNeuralNetworkUnscaled Target st pr
       w <-
         if scaled
           then lookupActionsNeuralNetwork Worker st pr
           else lookupActionsNeuralNetworkUnscaled Worker st pr
       return (st, (zip [0..] t,zip [0..] w)))
    (conf ^. prettyPrintElems)
  where
    conf =
      case pr of
        Grenade _ _ _ _ conf _         -> conf
        TensorflowProxy _ _ _ _ conf _ -> conf
        _                              -> error "mkNNList called on non-neural network"
    actIdxs = [0 .. _proxyNrActions pr]
    actFilt = borl ^. actionFilter
    useTable = borl ^. t == fromIntegral (_proxyNNConfig pr ^?! replayMemoryMaxSize) && (_proxyNNConfig pr ^?! trainBatchSize) /= 1
    lookupTable :: Bool -> [Double] -> [Double]
    lookupTable scale st
      | scale = val -- values are being unscaled, thus let table value be unscaled
      | otherwise = map (scaleValue (getMinMaxVal pr)) val
      where
        val = map (\actNr -> M.findWithDefault 0 (st, actNr) (_proxyNNStartup pr)) [0 .. _proxyNrActions pr]
