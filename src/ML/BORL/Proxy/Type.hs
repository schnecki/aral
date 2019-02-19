{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE ExplicitForAll            #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TemplateHaskell           #-}
{-# LANGUAGE UndecidableInstances      #-}

module ML.BORL.Proxy.Type
    ( Proxy (..)
    , ProxyType (..)
    , NNConfig (..)
    , LookupType (..)
    , insert
    , lookupProxy
    , lookupNeuralNetwork
    , lookupNeuralNetworkUnscaled
    , lookupActionsNeuralNetwork
    , lookupActionsNeuralNetworkUnscaled
    , getMinMaxVal
    ) where


import           ML.BORL.NeuralNetwork
import           ML.BORL.Types                as T

import           Control.Arrow
import           Control.DeepSeq
import           Control.Lens
import           Control.Monad
import           Data.List                    (foldl')
import qualified Data.Map.Strict              as M
import           Data.Singletons.Prelude.List
import           GHC.Generics
import           GHC.TypeLits
import           Grenade


-- | Type of approximation (needed for scaling of values).
data ProxyType
  = VTable
  | WTable
  | R0Table
  | R1Table
  deriving (Show, NFData, Generic)

data LookupType = Target | Worker

-- Todo: 2 Networks (target, worker)
data Proxy k = Table
               { _proxyTable :: !(M.Map (k,ActionIndex) Double)
               }
             | forall nrL nrH shapes layers. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, NFData (Tapes layers shapes), NFData (Network layers shapes)) =>
                Grenade
                { _proxyNNTarget  :: !(Network layers shapes)
                , _proxyNNWorker  :: !(Network layers shapes)
                , _proxyNNStartup :: !(M.Map (k,ActionIndex) Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !(NNConfig k)
                , _proxyNrActions :: !Int
                }
             | TensorflowProxy
                { _proxyTFTarget  :: TensorflowModel'
                , _proxyTFWorker  :: TensorflowModel'
                , _proxyNNStartup :: !(M.Map (k,ActionIndex) Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !(NNConfig k)
                , _proxyNrActions :: !Int
                }
makeLenses ''Proxy

instance (NFData k) => NFData (Proxy k) where
  rnf (Table x)           = rnf x
  rnf (Grenade t w tab tp cfg nrActs) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs
  rnf (TensorflowProxy t w tab tp cfg nrActs) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs

-- | Insert (or update) a value. The provided value will may be down-scaled to the interval [-1,1].
insert :: forall k . (NFData k, Ord k) => Period -> (k, ActionIndex) -> Double -> Proxy k -> T.MonadBorl (Proxy k)
insert _ k v (Table m)          = return $ Table (M.insert k v m)
insert period k@(st, idx) v px = do
  replMem' <- Simple $ addToReplayMemory period (k, scaleValue (getMinMaxVal px) v) (config ^. replayMemory)
  let config' = replayMemory .~ replMem' $ config
  if period < fromIntegral (config' ^. replayMemory . replayMemorySize) - 1
    then return $ proxyNNStartup .~ M.insert (st,idx) v tab $ proxyNNConfig .~ config' $ px
    else if period == fromIntegral (config' ^. replayMemory . replayMemorySize) - 1
           then do
             Simple $ putStrLn $ "Initializing artifical neural networks: " ++ show (px ^? proxyType)
             netInit (proxyNNConfig .~ config' $ px) >>= updateNNTargetNet True
           else trainNNConf period (proxyNNConfig .~ config' $ px) >>= updateNNTargetNet False
  where
    updateNNTargetNet :: Bool -> Proxy s -> T.MonadBorl (Proxy s)
    updateNNTargetNet forceReset px'@(Grenade _ netW' tab' tp' config' nrActs)
      | forceReset || period `mod` config' ^. updateTargetInterval == 0 = return $ Grenade netW' netW' tab' tp' config' nrActs
      | otherwise = return px'
    updateNNTargetNet forceReset px'@(TensorflowProxy netT' netW' tab' tp' config' nrActs)
      | forceReset || period `mod` config' ^. updateTargetInterval == 0 = do
        copyValuesFromTo netW' netT'
        return $ TensorflowProxy netT' netW' tab' tp' config' nrActs
      | otherwise = return px'
    updateNNTargetNet _ _ = error "updateNNTargetNet called on non-neural network proxy"
    netInit = trainMSE (Just 0) (M.toList tab) (config ^. learningParams)

    config = px ^?! proxyNNConfig
    tab = px ^?! proxyNNStartup


-- | Train until MSE hits the value given in NNConfig.
trainMSE :: (NFData k) => Maybe Int -> [((k, ActionIndex), Double)] -> LearningParameters -> Proxy k -> T.MonadBorl (Proxy k)
trainMSE _ _ _ px@Table{} = return px
trainMSE mPeriod dataset lp px@(Grenade _ netW tab tp config nrActs)
  | mse < mseMax = do
      Simple $ putStrLn $ "Final MSE for " ++ show tp ++ ": " ++ show mse
      return px
  | otherwise = do
      when (maybe False ((==0) . (`mod` 100)) mPeriod) $
        Simple $ putStrLn $ "Current MSE for " ++ show tp ++ ": " ++ show mse
      fmap force <$> trainMSE ((+ 1) <$> mPeriod) dataset lp $ Grenade net' net' tab tp config nrActs
  where
    mseMax = config ^. trainMSEMax
    net' = foldl' (trainGrenade lp) netW (zipWith (curry return) kScaled vScaled)
    -- net' = trainGrenade lp netW (zip kScaled vScaled)
    vScaled = map (scaleValue (getMinMaxVal px) . snd) dataset
    vUnscaled = map snd dataset
    kScaled = map (first (config ^. toNetInp) . fst) dataset
    getValue k =
      unscaleValue (getMinMaxVal px) $                                                                                 -- scaled or unscaled ones?
      (!!snd k) $ snd $ fromLastShapes netW $ runNetwork netW ((toHeadShapes netW . (config ^. toNetInp) . fst) k)
    mse = 1 / fromIntegral (length dataset) * sum (zipWith (\k v -> abs (v - getValue k)) (map fst dataset) vUnscaled) -- scaled or unscaled ones?
trainMSE mPeriod dataset lp px@(TensorflowProxy netT netW tab tp config nrActs) =
  let mseMax = config ^. trainMSEMax
      kFullScaled = map (first (map realToFrac . (config ^. toNetInp)) . fst) dataset :: [([Float], ActionIndex)]
      kScaled = map fst kFullScaled
      actIdxs = map snd kFullScaled
      vScaled = map (realToFrac . scaleValue (getMinMaxVal px) . snd) dataset
  in do current <- forwardRun netW kScaled
        zipWithM_ (backwardRun netW) (map return kScaled) (map return $ zipWith3 replace actIdxs vScaled current)
         -- backwardRun netW kScaled vScaled

        let forward k = realToFrac <$> lookupNeuralNetwork Worker k px
        mse <- (1 / fromIntegral (length dataset) *) . sum <$> zipWithM (\k vS -> abs . (vS -) <$> forward k) (map fst dataset) vScaled
        if realToFrac mse < mseMax
          then Simple $ putStrLn ("Final MSE for " ++ show tp ++ ": " ++ show mse) >> return px
          else do
            when (maybe False ((== 0) . (`mod` 100)) mPeriod) $ do
              void $ saveModelWithLastIO netW -- Save model to ensure correct values when reading from another session
              Simple $ putStrLn $ "Current MSE for " ++ show tp ++ ": " ++ show mse
            trainMSE ((+ 1) <$> mPeriod) dataset lp (TensorflowProxy netT netW tab tp config nrActs)


trainNNConf :: forall k . Period -> Proxy k -> T.MonadBorl (Proxy k)
trainNNConf period (Grenade netT netW tab tp config nrActs) = do
  rands <- Simple $ getRandomReplayMemoryElements period (config ^. trainBatchSize) (config ^. replayMemory)
  let trainingInstances = map (first (first $ config ^. toNetInp)) rands
      -- netW' = trainGrenade (config ^. learningParams) netW trainingInstances
      netW' = foldl' (trainGrenade (config ^. learningParams)) netW (map return trainingInstances)
  return $ Grenade netT netW' tab tp config nrActs
trainNNConf period (TensorflowProxy netT netW tab tp config nrActs) = do
  rands <- Simple $ getRandomReplayMemoryElements period (config ^. trainBatchSize) (config ^. replayMemory)
  let trainingInstances = map (first (first $ config ^. toNetInp)) rands
  backwardRunRepMemData netW trainingInstances
  return $ TensorflowProxy netT netW tab tp config nrActs

trainNNConf _ _ = error "called trainNNConf on non-neural network proxy (programming error)"


-- | Retrieve a value.
lookupProxy :: (Ord k) => Period -> LookupType -> (k, ActionIndex) -> Proxy k -> T.MonadBorl Double
lookupProxy _ _ k (Table m) = return $ M.findWithDefault 0 k m
lookupProxy period lkType k px
  | period <= fromIntegral (config ^. replayMemory.replayMemorySize) = return $ M.findWithDefault 0 k tab
  | otherwise = lookupNeuralNetwork lkType k px
  where config = px ^?! proxyNNConfig
        tab = px ^?! proxyNNStartup


-- | Retrieve a value from a neural network proxy. For other proxies an error is thrown. The returned value is up-scaled
-- to the original interval before returned.
lookupNeuralNetwork :: LookupType -> (k, ActionIndex) -> Proxy k -> T.MonadBorl Double
lookupNeuralNetwork tp k px@Grenade {} = unscaleValue (getMinMaxVal px) <$> lookupNeuralNetworkUnscaled tp k px
lookupNeuralNetwork tp k px@TensorflowProxy {} = unscaleValue (getMinMaxVal px) <$> lookupNeuralNetworkUnscaled tp k px
lookupNeuralNetwork _ _ _ = error "lookupNeuralNetwork called on non-neural network proxy"

-- | Retrieve a value from a neural network proxy. For other proxies an error is thrown. The returned value is up-scaled
-- to the original interval before returned.
lookupActionsNeuralNetwork :: LookupType -> k -> Proxy k -> T.MonadBorl [Double]
lookupActionsNeuralNetwork tp k px@Grenade {} = map (unscaleValue (getMinMaxVal px)) <$> lookupActionsNeuralNetworkUnscaled tp k px
lookupActionsNeuralNetwork tp k px@TensorflowProxy {} = map (unscaleValue (getMinMaxVal px)) <$> lookupActionsNeuralNetworkUnscaled tp k px
lookupActionsNeuralNetwork _ _ _ = error "lookupNeuralNetwork called on non-neural network proxy"


-- | Retrieve a value from a neural network proxy. For other proxies an error is thrown.
lookupNeuralNetworkUnscaled :: LookupType -> (k, ActionIndex) -> Proxy k -> T.MonadBorl Double
lookupNeuralNetworkUnscaled Worker (st, actIdx) (Grenade _ netW _ _ conf _) = return $ (!!actIdx) $ snd $ fromLastShapes netW $ runNetwork netW (toHeadShapes netW $ (conf ^. toNetInp) st)
lookupNeuralNetworkUnscaled Target (st, actIdx) (Grenade netT _ _ _ conf _) = return $ (!!actIdx) $ snd $ fromLastShapes netT $ runNetwork netT (toHeadShapes netT $ (conf ^. toNetInp) st)
lookupNeuralNetworkUnscaled Worker (st, actIdx) (TensorflowProxy _ netW _ _ conf _) = realToFrac . (!!actIdx) . head <$> forwardRun netW [map realToFrac $ (conf^. toNetInp) st]
lookupNeuralNetworkUnscaled Target (st, actIdx) (TensorflowProxy netT _ _ _ conf _) = realToFrac . (!!actIdx) . head <$> forwardRun netT [map realToFrac $ (conf^. toNetInp) st]
lookupNeuralNetworkUnscaled _ _ _ = error "lookupNeuralNetworkUnscaled called on non-neural network proxy"

-- | Retrieve all action values of a state from a neural network proxy. For other proxies an error is thrown.
lookupActionsNeuralNetworkUnscaled :: forall k . LookupType -> k -> Proxy k -> T.MonadBorl [Double]
lookupActionsNeuralNetworkUnscaled Worker st (Grenade _ netW _ _ conf _) = return $ snd $ fromLastShapes netW $ runNetwork netW (toHeadShapes netW $ (conf ^. toNetInp) st)
lookupActionsNeuralNetworkUnscaled Target st (Grenade netT _ _ _ conf _) = return $ snd $ fromLastShapes netT $ runNetwork netT (toHeadShapes netT $ (conf ^. toNetInp) st)
lookupActionsNeuralNetworkUnscaled Worker st (TensorflowProxy _ netW _ _ conf _) = map realToFrac . head <$> forwardRun netW [map realToFrac $ (conf^. toNetInp) st]
lookupActionsNeuralNetworkUnscaled Target st (TensorflowProxy netT _ _ _ conf _) = map realToFrac . head <$> forwardRun netT [map realToFrac $ (conf^. toNetInp) st]
lookupActionsNeuralNetworkUnscaled _ _ _ = error "lookupNeuralNetworkUnscaled called on non-neural network proxy"


-- | Finds the correct value for scaling.
getMinMaxVal :: Proxy k -> (MinValue,MaxValue)
getMinMaxVal Table{} = error "getMinMaxVal called for Table"
getMinMaxVal p  = case p ^?! proxyType of
  VTable  -> (p ^?! proxyNNConfig.scaleParameters.scaleMinVValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxVValue)
  WTable  -> (p ^?! proxyNNConfig.scaleParameters.scaleMinWValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxWValue)
  R0Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinR0Value, p ^?! proxyNNConfig.scaleParameters.scaleMaxR0Value)
  R1Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinR1Value, p ^?! proxyNNConfig.scaleParameters.scaleMaxR1Value)

