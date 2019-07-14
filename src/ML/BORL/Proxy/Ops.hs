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
import           ML.BORL.Proxy.Type
import           ML.BORL.Type
import           ML.BORL.Reward
import           ML.BORL.Parameters
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


import           Control.Lens
import qualified Data.Map.Strict              as M

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
insert borl period state aNr randAct rew stateNext episodeEnd getCalc pxs@(Proxies pRhoMin pRho pPsiV pV pW pR0 pR1 Nothing) = do
  let 
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
    pPsiV' <- mInsertProxy (getPsiVVal' calc) pPsiV `using` rpar
    pR0' <- mInsertProxy (getR0ValState' calc) pR0 `using` rpar
    pR1' <- insertProxy period stateFeat aNr (getR1ValState' calc) pR1 `using` rpar
    return (Proxies pRhoMin' pRho' pPsiV' pV' pW' pR0' pR1' Nothing, calc)
  where
    sActIdxes = map fst $ actionsIndexed borl state
    sNextActIdxes = map fst $ actionsIndexed borl stateNext
    period = borl ^. t
    stateFeat = (borl ^. featureExtractor) state
    stateNextFeat = (borl ^. featureExtractor) stateNext
    stateActs = (stateFeat, sActIdxes)
    stateNextActs = (stateNextFeat, sNextActIdxes)
insert borl period state aNr randAct rew stateNext episodeEnd getCalc pxs@(Proxies pRhoMin pRho pPsiV pV pW pR0 pR1 (Just replMem))
  | period <= fromIntegral (replMem ^. replayMemorySize) - 1 = do
    replMem' <- liftSimple $ addToReplayMemory period (stateActs, aNr, randAct, rew, stateNextActs, episodeEnd) replMem
    (pxs', calc) <- insert borl period state aNr randAct rew stateNext episodeEnd getCalc (replayMemory .~ Nothing $ pxs)
    return (replayMemory ?~ replMem' $ pxs', calc)
  | pV ^?! proxyNNConfig . trainBatchSize == 1 = do
    replMem' <- liftSimple $ addToReplayMemory period (stateActs, aNr, randAct, rew, stateNextActs, episodeEnd) replMem
    calc <- getCalc stateActs aNr randAct rew stateNextActs episodeEnd
    if borl ^. parameters . disableAllLearning
      then return (set replayMemory (Just replMem') pxs, calc)
      else do
        let mInsertProxy mVal px = maybe (return px) (\val -> insertProxy period stateFeat aNr val px) mVal
        pRho' <- mInsertProxy (getRhoVal' calc) pRho `using` rpar
        pRhoMin' <- mInsertProxy (getRhoMinimumVal' calc) pRhoMin `using` rpar
        pV' <- mInsertProxy (getVValState' calc) pV `using` rpar
        pW' <- mInsertProxy (getWValState' calc) pW `using` rpar
        pPsiV' <- mInsertProxy (getPsiVVal' calc) pPsiV `using` rpar
        pR0' <- mInsertProxy (getR0ValState' calc) pR0 `using` rpar
        pR1' <- insertProxy period stateFeat aNr (getR1ValState' calc) pR1 `using` rpar
        return (Proxies pRhoMin' pRho' pPsiV' pV' pW' pR0' pR1' (Just replMem'), calc)
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
                (flip trainBatch px)
                (mapM
                   (\c ->
                      let (inp, mOut) = second accessor c
                       in mOut >>= \out -> return (inp, out))
                   calcs)
      -- let avgCalc = avgCalculation (map snd calcs)
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
        pPsiV' <- mTrainBatch getPsiVVal' calcs pPsiV `using` rpar
        pR0' <- mTrainBatch getR0ValState' calcs pR0 `using` rpar
        pR1' <- trainBatch (map (second getR1ValState') calcs) pR1 `using` rpar
        return (Proxies pRhoMin' pRho' pPsiV' pV' pW' pR0' pR1' (Just replMem'), calc)
            -- avgCalculation (map snd calcs))
  where
    sActIdxes = map fst $ actionsIndexed borl state
    sNextActIdxes = map fst $ actionsIndexed borl stateNext
    period = borl ^. t
    state = borl ^. s
    stateFeat = (borl ^. featureExtractor) state
    stateNextFeat = (borl ^. featureExtractor) stateNext
    stateActs = (stateFeat, sActIdxes)
    stateNextActs = (stateNextFeat, sNextActIdxes)

-- | Insert a new (single) value to the proxy. For neural networks this will add the value to the startup table. See
-- `trainBatch` to train the neural networks.
insertProxy :: (MonadBorl' m) => Period -> StateFeatures -> ActionIndex -> Double -> Proxy -> m Proxy
insertProxy _ _ _ v (Scalar _) = return $ Scalar v
insertProxy _ st aNr v (Table m def) = return $ Table (M.insert (map trunc st, aNr) v m) def
  where trunc x = (fromInteger $ round $ x * (10^n)) / (10.0^^n)
        n = 3
insertProxy period st idx v px
  | period < fromIntegral (px ^?! proxyNNConfig . replayMemoryMaxSize) - 1 && (px ^?! proxyNNConfig . trainBatchSize) == 1 =
    trainBatch [((st, idx), v)] px >>= updateNNTargetNet False period
  | period < fromIntegral (px ^?! proxyNNConfig . replayMemoryMaxSize) - 1 && isNothing (px ^?! proxyNNConfig . trainMSEMax) = return px
  | period < fromIntegral (px ^?! proxyNNConfig . replayMemoryMaxSize) - 1 = return $ proxyNNStartup .~ M.insert (st, idx) v tab $ px
  | period == fromIntegral (px ^?! proxyNNConfig . replayMemoryMaxSize) - 1 && isNothing (px ^?! proxyNNConfig . trainMSEMax) = updateNNTargetNet False period px
  | period == fromIntegral (px ^?! proxyNNConfig . replayMemoryMaxSize) - 1 =
    liftSimple (putStrLn $ "Initializing artificial neural networks: " ++ show (px ^? proxyType)) >> netInit px >>= updateNNTargetNet True period
  | otherwise = trainBatch [((st, idx), v)] px >>= updateNNTargetNet False period
  where
    netInit = trainMSE (Just 0) (M.toList tab) (config ^. grenadeLearningParams)
    config = px ^?! proxyNNConfig
    tab = px ^?! proxyNNStartup

-- | Copy the worker net to the target.
updateNNTargetNet :: (MonadBorl' m) => Bool -> Period -> Proxy -> m Proxy
updateNNTargetNet forceReset period px@(Grenade _ netW' tab' tp' config' nrActs)
  | not forceReset && period <= fromIntegral memSize && config' ^. trainBatchSize /= 1 = return px
  | forceReset || ((period - fromIntegral memSize - 1) `mod` config' ^. updateTargetInterval) == 0 = return $ Grenade netW' netW' tab' tp' config' nrActs
  | otherwise = return px
  where
    memSize = px ^?! proxyNNConfig . replayMemoryMaxSize
updateNNTargetNet forceReset period px@(TensorflowProxy netT' netW' tab' tp' config' nrActs)
  | not forceReset && period <= fromIntegral memSize && config' ^. trainBatchSize /= 1 = return px
  | forceReset || ((period - fromIntegral memSize - 1) `mod` config' ^. updateTargetInterval) == 0 = do
      copyValuesFromTo netW' netT'
      return $ TensorflowProxy netT' netW' tab' tp' config' nrActs
  | otherwise = return px
  where
    memSize = px ^?! proxyNNConfig . replayMemoryMaxSize
updateNNTargetNet _ _ _ = error "updateNNTargetNet called on non-neural network proxy"


-- | Train the neural network from a given batch. The training instances are Unscaled.
trainBatch :: forall m s . (MonadBorl' m) => [((StateFeatures, ActionIndex), Double)] -> Proxy -> m Proxy
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
    when (maybe False ((== 0) . (`mod` 100)) mPeriod) $ liftSimple $ putStrLn $ "Current MSE for " ++ show tp ++ ": " ++ show mse
    fmap force <$> trainMSE ((+ 1) <$> mPeriod) dataset lp $ Grenade net' net' tab tp config nrActs
  where
    mseMax = fromJust (config ^. trainMSEMax)
    net' = foldl' (trainGrenade lp) netW (zipWith (curry return) kScaled vScaled)
    -- net' = trainGrenade lp netW (zip kScaled vScaled)
    vScaled = map (scaleValue (getMinMaxVal px) . snd) dataset
    vUnscaled = map snd dataset
    kScaled = map fst dataset
    getValue k
      -- unscaleValue (getMinMaxVal px) $                                                                                 -- scaled or unscaled ones?
     = (!! snd k) $ snd $ fromLastShapes netW $ runNetwork netW ((toHeadShapes netW . fst) k)
    mse = 1 / fromIntegral (length dataset) * sum (zipWith (\k v -> (v - getValue k) ** 2) (map fst dataset) (map (min 1 . max (-1)) vScaled)) -- scaled or unscaled ones?
trainMSE mPeriod dataset lp px@(TensorflowProxy netT netW tab tp config nrActs)
  | isNothing (config ^. trainMSEMax) = return px
  | otherwise =
    let mseMax = fromJust (config ^. trainMSEMax)
        kFullScaled = map (first (map realToFrac) . fst) dataset :: [([Float], ActionIndex)]
        kScaled = map fst kFullScaled
        actIdxs = map snd kFullScaled
        vScaled = map (realToFrac . scaleValue (getMinMaxVal px) . snd) dataset
        vUnscaled = map (realToFrac . snd) dataset
        datasetRepMem = map (first (first (map realToFrac))) dataset
    in do current <- forwardRun netW kScaled
          zipWithM_ (backwardRun netW) (map return kScaled) (map return $ zipWith3 replace actIdxs vScaled current)
        -- backwardRunRepMemData netW datasetRepMem
          let forward k = realToFrac <$> lookupNeuralNetworkUnscaled Worker k px -- lookupNeuralNetwork Worker k px -- scaled or unscaled ones?
          mse <- (1 / fromIntegral (length dataset) *) . sum <$> zipWithM (\k vS -> (** 2) . (vS -) <$> forward k) (map fst dataset) (map (min 1 . max (-1)) vScaled) -- vUnscaled -- scaled or unscaled ones?
          if realToFrac mse < mseMax
            then liftSimple $ putStrLn ("Final MSE for " ++ show tp ++ ": " ++ show mse) >> return px
            else do
              when (maybe False ((== 0) . (`mod` 100)) mPeriod) $ do
                void $ saveModelWithLastIO netW -- Save model to ensure correct values when reading from another session
                liftSimple $ putStrLn $ "Current MSE for " ++ show tp ++ ": " ++ show mse
              trainMSE ((+ 1) <$> mPeriod) dataset lp (TensorflowProxy netT netW tab tp config nrActs)
trainMSE _ _ _ _ = error "trainMSE should not have been callable with this type of proxy. programming error!"

-- | Retrieve a value.
lookupProxy :: (MonadBorl' m) => Period -> LookupType -> (StateFeatures, ActionIndex) -> Proxy -> m Double
lookupProxy _ _ _ (Scalar x) = return x
lookupProxy _ _ k (Table m def) = return $ M.findWithDefault def k m
lookupProxy period lkType k px
  | period <= fromIntegral (config ^. replayMemoryMaxSize) && (config ^. trainBatchSize) /= 1 = return $ M.findWithDefault 0 k tab
  | otherwise = lookupNeuralNetwork lkType k px
  where config = px ^?! proxyNNConfig
        tab = px ^?! proxyNNStartup


-- | Retrieve a value from a neural network proxy. For other proxies an error is thrown. The returned value is up-scaled
-- to the original interval before returned.
lookupNeuralNetwork :: (MonadBorl' m) => LookupType -> (StateFeatures, ActionIndex) -> Proxy -> m Double
lookupNeuralNetwork tp k px@Grenade {} = unscaleValue (getMinMaxVal px) <$> lookupNeuralNetworkUnscaled tp k px
lookupNeuralNetwork tp k px@TensorflowProxy {} = unscaleValue (getMinMaxVal px) <$> lookupNeuralNetworkUnscaled tp k px
lookupNeuralNetwork _ _ _ = error "lookupNeuralNetwork called on non-neural network proxy"

-- | Retrieve a value from a neural network proxy. For other proxies an error is thrown. The returned value is up-scaled
-- to the original interval before returned.
lookupActionsNeuralNetwork :: (MonadBorl' m) => LookupType -> StateFeatures -> Proxy -> m [Double]
lookupActionsNeuralNetwork tp k px@Grenade {} = map (unscaleValue (getMinMaxVal px)) <$> lookupActionsNeuralNetworkUnscaled tp k px
lookupActionsNeuralNetwork tp k px@TensorflowProxy {} = map (unscaleValue (getMinMaxVal px)) <$> lookupActionsNeuralNetworkUnscaled tp k px
lookupActionsNeuralNetwork _ _ _ = error "lookupNeuralNetwork called on non-neural network proxy"


-- | Retrieve a value from a neural network proxy. For other proxies an error is thrown.
lookupNeuralNetworkUnscaled :: (MonadBorl' m) => LookupType -> (StateFeatures, ActionIndex) -> Proxy -> m Double
lookupNeuralNetworkUnscaled Worker (st, actIdx) (Grenade _ netW _ _ conf _) = return $ (!!actIdx) $ snd $ fromLastShapes netW $ runNetwork netW (toHeadShapes netW st)
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
  R0Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinR0Value, p ^?! proxyNNConfig.scaleParameters.scaleMaxR0Value)
  R1Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinR1Value, p ^?! proxyNNConfig.scaleParameters.scaleMaxR1Value)
  PsiVTable -> (p ^?! proxyNNConfig.scaleParameters.scaleMinVValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxVValue)


-- | This function loads the model from the checkpoint file and finds then retrieves the data.
mkNNList :: (MonadBorl' m) => BORL k -> Bool -> Proxy -> m [(NetInput, ([(ActionIndex, Double)], [(ActionIndex, Double)]))]
mkNNList borl scaled pr =
  mapM
    (\st -> do
       -- let fil = actFilt st
       --     filterActions xs = map (\(_, a, b) -> (a, b)) $ filter (\(f, _, _) -> f) $ zip3 fil [(0 :: Int) ..] xs
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
       -- return (st, (filterActions t, filterActions w)))
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
