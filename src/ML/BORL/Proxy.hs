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
    ) where


import           ML.BORL.NeuralNetwork
import           ML.BORL.Types

import           Control.Arrow
import           Control.DeepSeq
import           Control.Lens
import           Control.Monad
import           Control.Parallel.Strategies
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
               { _proxyTable :: !(M.Map k Double)
               }
             | forall nrL nrH shapes layers. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, NFData (Tapes layers shapes), NFData (Network layers shapes)) =>
                NN
                { _proxyNNTarget  :: !(Network layers shapes)
                , _proxyNNWorker  :: !(Network layers shapes)
                , _proxyNNStartup :: !(M.Map k Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !(NNConfig k)
                }
            | Tensorflow
makeLenses ''Proxy

instance (NFData k) => NFData (Proxy k) where
  rnf (Table x)           = rnf x
  rnf (NN t w tab tp cfg) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg


-- | Insert (or update) a value. The provided value will may be down-scaled to the interval [-1,1].
insert :: forall k . (Ord k) => Period -> k -> Double -> Proxy k -> IO (Proxy k)
insert _ k v (Table m)          = return $ Table (M.insert k v m)
insert period k v px@(NN netT netW tab tp config) = do
  replMem' <- addToReplayMemory period (k, scaleValue (getMinMaxVal px) v) (config ^. replayMemory)
  let config' = replayMemory .~ replMem' $ config
  --
  --   then return $ NN netT netW  tp config'
  --   else
  if period < fromIntegral (config' ^. replayMemory.replayMemorySize)-1
    then return $ proxyNNStartup .~ M.insert k v tab $ proxyNNConfig .~  config' $ px
    else if period == fromIntegral (config' ^. replayMemory.replayMemorySize) - 1
    then do putStrLn "Initializing artifical neural network"
            updateNNTargetNet True <$> netInit (NN netT netW tab tp config')
    else updateNNTargetNet False <$> trainNNConf period (proxyNNConfig .~  config' $ px)
  where
    updateNNTargetNet forceReset px'@(NN _ netW' tab' tp' config')
      | forceReset || period `mod` config' ^. updateTargetInterval == 0 = NN netW' netW' tab' tp' config'
      | otherwise = px'
    updateNNTargetNet _ _ = error "updateNNTargetNet called on non-neural network proxy"
    netInit = trainMSE (Just 0) (M.toList tab) (config ^. learningParams)


trainMSE :: Maybe Int -> [(k, Double)] -> LearningParameters -> Proxy k -> IO (Proxy k)
trainMSE _ _ _ px@Table{} = return px
trainMSE mPeriod dataset lp px@(NN _ netW tab tp config)
  | mse < mseMax = do
      putStrLn $ "Current MSE for " ++ show tp ++ ": " ++ show mse
      return px
  | otherwise = do
      when (maybe False ((==0) . (`mod` 10)) mPeriod) $
        putStrLn $ "Current MSE for " ++ show tp ++ ": " ++ show mse
      trainMSE ((+ 1) <$> mPeriod) dataset lp $ NN net' net' tab tp config
  where
    mseMax = config ^. trainMSEMax
    net' = trainNetwork lp netW (zip kScaled vScaled)
    vScaled = map (scaleValue (getMinMaxVal px) . snd) dataset
    kScaled = map ((config ^. toNetInp) . fst) dataset
    forwardRun k = head $ snd $ fromLastShapes netW $ runNetwork netW (toHeadShapes netW $ (config ^. toNetInp) k)
    mse = 1 / fromIntegral (length dataset) * sum (zipWith (\k vS -> abs (vS - forwardRun k)) (map fst dataset) vScaled)

trainNNConf :: forall k . (Ord k) => Period -> Proxy k -> IO (Proxy k)
-- trainNNConf period (NN netT netW tab tp config) | period < fromIntegral (config ^. replayMemory.replayMemorySize) = return $ NN netT netW tab tp config
trainNNConf period (NN netT netW tab tp config) = do
  rands <- getRandomReplayMemoryElements period (config ^. trainBatchSize) (config ^. replayMemory)
  let trainingInstances = map (first $ config ^. toNetInp) rands
      netW' = trainNetwork (config ^. learningParams) netW trainingInstances
  return $ NN netT netW' tab tp config
trainNNConf _ _ = error "called trainNNConf on non-neural network proxy (programming error)"


-- | Retrieve a value.
lookupProxy :: (Ord k) => Period -> LookupType -> k -> Proxy k -> Double
lookupProxy _ _ k (Table m) = M.findWithDefault 0 k m
lookupProxy period lkTp k px@(NN _ _ tab _ config)
  | period <= fromIntegral (config ^. replayMemory.replayMemorySize) = M.findWithDefault 0 k tab
  | otherwise = lookupNeuralNetwork lkTp k px


-- | Retrieve a value from a neural network proxy. For other proxies an error is thrown. The returned value is up-scaled
-- to the original interval before returned.
lookupNeuralNetwork :: LookupType -> k -> Proxy k -> Double
lookupNeuralNetwork Worker k px@(NN _ netW _ _ conf) = unscaleValue (getMinMaxVal px) $ head $ snd $ fromLastShapes netW $ runNetwork netW (toHeadShapes netW $ (conf ^. toNetInp) k)
lookupNeuralNetwork Target k px@(NN netT _ _ _ conf) = unscaleValue (getMinMaxVal px) $ head $ snd $ fromLastShapes netT $ runNetwork netT (toHeadShapes netT $ (conf ^. toNetInp) k)
lookupNeuralNetwork _ _ _ = error "lookupNeuralNetwork called on non-neural network proxy"


-- | Finds the correct value for scaling.
getMinMaxVal :: Proxy k -> (MinValue,MaxValue)
getMinMaxVal Table{} = error "getMinMaxVal called for Table"
getMinMaxVal p@NN {}  = case p ^?! proxyType of
  VTable  -> (p ^?! proxyNNConfig.scaleParameters.scaleMinVValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxVValue)
  WTable  -> (p ^?! proxyNNConfig.scaleParameters.scaleMinWValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxWValue)
  R0Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinR0Value, p ^?! proxyNNConfig.scaleParameters.scaleMaxR0Value)
  R1Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinR1Value, p ^?! proxyNNConfig.scaleParameters.scaleMaxR1Value)
