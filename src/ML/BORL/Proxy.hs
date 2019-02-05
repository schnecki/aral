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

module ML.BORL.Proxy
    ( Proxy (..)
    , ProxyType (..)
    , NNConfig (..)
    , LookupType (..)
    , insert
    , lookupProxy
    , lookupNeuralNetwork
    , lookupNeuralNetworkUnscaled
    , mkNNList
    ) where


import           ML.BORL.NeuralNetwork
import           ML.BORL.Types                as T

import           Control.Arrow
import           Control.DeepSeq
import           Control.Lens
import           Control.Monad
import           Control.Monad.IO.Class       (MonadIO, liftIO)
import           Control.Parallel.Strategies
import           Data.List                    (foldl')
import qualified Data.Map.Strict              as M
import           Data.Singletons.Prelude.List
import qualified Data.Vector                  as V
import           GHC.Generics
import           GHC.TypeLits
import           Grenade
import qualified TensorFlow.Core              as TF
import           TensorFlow.Session


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
               { _proxyTable :: !(M.Map k Double)
               }
             | forall nrL nrH shapes layers. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, NFData (Tapes layers shapes), NFData (Network layers shapes)) =>
                Grenade
                { _proxyNNTarget  :: !(Network layers shapes)
                , _proxyNNWorker  :: !(Network layers shapes)
                , _proxyNNStartup :: !(M.Map k Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !(NNConfig k)
                }
             | TensorflowProxy
                { _proxyTFTarget  :: TensorflowModel'
                , _proxyTFWorker  :: TensorflowModel'
                , _proxyNNStartup :: !(M.Map k Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !(NNConfig k)
                }
makeLenses ''Proxy

instance (NFData k) => NFData (Proxy k) where
  rnf (Table x)           = rnf x
  rnf (Grenade t w tab tp cfg) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg
  rnf (TensorflowProxy t w tab tp cfg) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg

-- | Insert (or update) a value. The provided value will may be down-scaled to the interval [-1,1].
insert :: forall k . (Ord k) => Period -> k -> Double -> Proxy k -> T.MonadBorl (Proxy k)
insert _ k v (Table m)          = return $ Table (M.insert k v m)
insert period k v px-- @(Grenade netT netW tab tp config)
  = do
  replMem' <- Pure $ addToReplayMemory period (k, scaleValue (getMinMaxVal px) v) (config ^. replayMemory)
  let config' = replayMemory .~ replMem' $ config
  --
  --   then return $ Grenade netT netW  tp config'
  --   else
  if period < fromIntegral (config' ^. replayMemory.replayMemorySize)-1
    then return $ proxyNNStartup .~ M.insert k v tab $ proxyNNConfig .~  config' $ px
    else if period == fromIntegral (config' ^. replayMemory.replayMemorySize) - 1
    then do Pure $ putStrLn "Initializing artifical neural network"
            netInit (proxyNNConfig .~ config' $ px) >>= updateNNTargetNet True
    else trainNNConf period (proxyNNConfig .~  config' $ px) >>= updateNNTargetNet False
  where
    updateNNTargetNet :: Bool -> Proxy s -> T.MonadBorl (Proxy s)
    updateNNTargetNet forceReset px'@(Grenade _ netW' tab' tp' config')
      | forceReset || period `mod` config' ^. updateTargetInterval == 0 = return $ Grenade netW' netW' tab' tp' config'
      | otherwise = return px'
    updateNNTargetNet forceReset px'@(TensorflowProxy netT' netW' tab' tp' config')
      | forceReset || period `mod` config' ^. updateTargetInterval == 0 = do
          copyValuesFromTo netW' netT'
          return $ TensorflowProxy netT' netW' tab' tp' config'
      | otherwise = return px'
    updateNNTargetNet _ _ = error "updateNNTargetNet called on non-neural network proxy"
    netInit = trainMSE (Just 0) (M.toList tab) (config ^. learningParams)
    config = px ^?! proxyNNConfig
    tab = px ^?! proxyNNStartup


trainMSE :: Maybe Int -> [(k, Double)] -> LearningParameters -> Proxy k -> T.MonadBorl (Proxy k)
trainMSE _ _ _ px@Table{} = return px
trainMSE mPeriod dataset lp px@(Grenade _ netW tab tp config)
  | mse < mseMax = do
      Pure $ putStrLn $ "Final MSE for " ++ show tp ++ ": " ++ show mse
      return px
  | otherwise = do
      when (maybe False ((==0) . (`mod` 100)) mPeriod) $
        Pure $ putStrLn $ "Current MSE for " ++ show tp ++ ": " ++ show mse
      trainMSE ((+ 1) <$> mPeriod) dataset lp $ Grenade net' net' tab tp config
  where
    mseMax = config ^. trainMSEMax
    net' = foldl' (trainGrenade lp) netW (zipWith (curry return) kScaled vScaled)
    vScaled = map (scaleValue (getMinMaxVal px) . snd) dataset
    kScaled = map ((config ^. toNetInp) . fst) dataset
    getValue k = head $ snd $ fromLastShapes netW $ runNetwork netW (toHeadShapes netW $ (config ^. toNetInp) k)
    mse = 1 / fromIntegral (length dataset) * sum (zipWith (\k vS -> abs (vS - getValue k)) (map fst dataset) vScaled)
trainMSE mPeriod dataset lp px@(TensorflowProxy netT netW _ _ _) = do
    -- restoreModelWithLastIO netW
    px' <- trainMSETensorflow mPeriod dataset lp px
    -- netW' <- saveModelWithLastIO (px' ^?! proxyTFWorker)
    return $ proxyTFWorker .~ netW $ px'

-- | Train a Tensorflow object in a single session.
trainMSETensorflow :: Maybe Int -> [(k, Double)] -> t -> Proxy k -> T.MonadBorl (Proxy k)
trainMSETensorflow mPeriod dataset lp px@(TensorflowProxy netT netW tab tp config) =
  let mseMax = config ^. trainMSEMax
      kScaled = map (map realToFrac . (config ^. toNetInp) . fst) dataset :: [[Float]]
      vScaled = map (realToFrac . scaleValue (getMinMaxVal px) . snd) dataset :: [Float]
   in do zipWithM_ (backwardRunSession netW) (map return kScaled) (map return vScaled)
         let forward k = head <$> forwardRunSession netW [map realToFrac $ (config ^. toNetInp) k]
         mse <- (1 / fromIntegral (length dataset) *) . sum <$> T.Tensorflow (zipWithM (\k vS -> abs . (vS -) <$> forward k) (map fst dataset) vScaled)
         if realToFrac mse < mseMax
           then Pure $ putStrLn ("Final MSE for " ++ show tp ++ ": " ++ show mse) >> return px
           else do
             when (maybe False ((== 0) . (`mod` 100)) mPeriod) $ do
               -- varVals :: [V.Vector Float] <- TF.run (neuralNetworkVariables $ tensorflowModel netW)
               -- Pure $ putStrLn $ "Weights: " ++ show (V.toList <$> varVals)
               void $ saveModelWithLastIO netW -- Save model to ensure correct values when reading from another session
               Pure $ do
                 -- list <- mkNNList False px
                 -- let list' = map ((config ^. toNetInp) *** snd) list
                 -- mapM_ (\((ks, w), (ks', v')) -> putStrLn $ show ks ++ ":\t" ++ show w ++ "\t" ++ show ks' ++ ":\t" ++ show v') (zip list' (zip kScaled vScaled))
                 putStrLn $ "Current MSE for " ++ show tp ++ ": " ++ show mse
             trainMSETensorflow ((+ 1) <$> mPeriod) dataset lp $ TensorflowProxy netT netW tab tp config -- TODO check if
                                                                                                    -- using same net'
                                                                                                    -- is OK. I don't
                                                                                                    -- think so
trainMSETensorflow  _ _ _ _ = error "called trainMSETensorflow on non-Tensorflow data structure"

trainNNConf :: forall k . Period -> Proxy k -> T.MonadBorl (Proxy k)
-- trainNNConf period (Grenade netT netW tab tp config) | period < fromIntegral (config ^. replayMemory.replayMemorySize) = return $ Grenade netT netW tab tp config
trainNNConf period (Grenade netT netW tab tp config) = do
  rands <- Pure $ getRandomReplayMemoryElements period (config ^. trainBatchSize) (config ^. replayMemory)
  let trainingInstances = map (first $ config ^. toNetInp) rands
      netW' = trainGrenade (config ^. learningParams) netW trainingInstances
  return $ Grenade netT netW' tab tp config
trainNNConf period (TensorflowProxy netT netW tab tp config) = do
  rands <- Pure $ getRandomReplayMemoryElements period (config ^. trainBatchSize) (config ^. replayMemory)
  let trainingInstances = map (first $ config ^. toNetInp) rands
      inputs = map (map realToFrac . fst) trainingInstances
      labels = map (realToFrac . snd) trainingInstances
  backwardRun netW inputs labels
  return $ TensorflowProxy netT netW tab tp config

trainNNConf _ _ = error "called trainNNConf on non-neural network proxy (programming error)"


-- | Retrieve a value.
lookupProxy :: (Ord k) => Period -> LookupType -> k -> Proxy k -> T.MonadBorl Double
lookupProxy _ _ k (Table m) = return $ M.findWithDefault 0 k m
lookupProxy period lkType k px
  | period <= fromIntegral (config ^. replayMemory.replayMemorySize) = return $ M.findWithDefault 0 k tab
  | otherwise = lookupNeuralNetwork lkType k px
  where config = px ^?! proxyNNConfig
        tab = px ^?! proxyNNStartup


-- | Retrieve a value from a neural network proxy. For other proxies an error is thrown. The returned value is up-scaled
-- to the original interval before returned.
lookupNeuralNetwork :: LookupType -> k -> Proxy k -> T.MonadBorl Double
lookupNeuralNetwork tp k px@Grenade {} = unscaleValue (getMinMaxVal px) <$> lookupNeuralNetworkUnscaled tp k px
lookupNeuralNetwork tp k px@TensorflowProxy {} = unscaleValue (getMinMaxVal px) <$> lookupNeuralNetworkUnscaled tp k px
lookupNeuralNetwork _ _ _ = error "lookupNeuralNetwork called on non-neural network proxy"

-- | Retrieve a value from a neural network proxy. For other proxies an error is thrown.
lookupNeuralNetworkUnscaled :: LookupType -> k -> Proxy k -> T.MonadBorl Double
lookupNeuralNetworkUnscaled Worker k (Grenade _ netW _ _ conf) = return $ head $ snd $ fromLastShapes netW $ runNetwork netW (toHeadShapes netW $ (conf ^. toNetInp) k)
lookupNeuralNetworkUnscaled Target k (Grenade netT _ _ _ conf) = return $ head $ snd $ fromLastShapes netT $ runNetwork netT (toHeadShapes netT $ (conf ^. toNetInp) k)
lookupNeuralNetworkUnscaled Worker k (TensorflowProxy _ netW _ _ conf) = realToFrac . head <$> forwardRun netW [map realToFrac $ (conf^. toNetInp) k]
lookupNeuralNetworkUnscaled Target k (TensorflowProxy netT _ _ _ conf) = realToFrac . head <$> forwardRun netT [map realToFrac $ (conf^. toNetInp) k]
lookupNeuralNetworkUnscaled _ _ _ = error "lookupNeuralNetworkUnscaled called on non-neural network proxy"


-- | Finds the correct value for scaling.
getMinMaxVal :: Proxy k -> (MinValue,MaxValue)
getMinMaxVal Table{} = error "getMinMaxVal called for Table"
getMinMaxVal p  = case p ^?! proxyType of
  VTable  -> (p ^?! proxyNNConfig.scaleParameters.scaleMinVValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxVValue)
  WTable  -> (p ^?! proxyNNConfig.scaleParameters.scaleMinWValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxWValue)
  R0Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinR0Value, p ^?! proxyNNConfig.scaleParameters.scaleMaxR0Value)
  R1Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinR1Value, p ^?! proxyNNConfig.scaleParameters.scaleMaxR1Value)


-- | This function loads the model from the checkpoint file and finds then retrieves the data.
mkNNList :: Bool -> Proxy k -> T.MonadBorl [(k, (Double, Double))]
mkNNList scaled pr =
  mapM
    (\inp -> do
       t <-
         if scaled
           then lookupNeuralNetwork Target inp pr
           else lookupNeuralNetworkUnscaled Target inp pr
       w <-
         if scaled
           then lookupNeuralNetwork Worker inp pr
           else lookupNeuralNetworkUnscaled Worker inp pr
       return (inp, (t, w)))
    (conf ^. prettyPrintElems)
  where
    conf =
      case pr of
        Grenade _ _ _ _ conf         -> conf
        TensorflowProxy _ _ _ _ conf -> conf
        _                            -> error "mkNNList called on non-neural network"
