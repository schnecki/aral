{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE ExplicitForAll            #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TemplateHaskell           #-}

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
import           Control.Lens
import           Control.Monad
import           Control.Parallel.Strategies
import qualified Data.Map.Strict              as M
import           Data.Singletons.Prelude.List
import           GHC.TypeLits
import           Grenade


-- | Type of approximation (needed for scaling of values).
data ProxyType
  = VTable
  | WTable
  | R0Table
  | R1Table

data LookupType = Target | Worker

-- Todo: 2 Networks (target, worker)
data Proxy k = Table
               { _proxyTable :: !(M.Map k Double)
               }
             | forall nrL nrH shapes layers. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, NFData (Tapes layers shapes)) =>
                NN
                { _proxyNNTarget :: !(Network layers shapes)
                , _proxyNNWorker :: !(Network layers shapes)
                , _proxyType     :: !ProxyType
                , _proxyNNConfig :: !(NNConfig k)
                }
makeLenses ''Proxy

-- | Insert (or update) a value. The provided value will may be down-scaled to the interval [-1,1].
insert :: forall k . (Ord k) => Period -> k -> Double -> Proxy k -> IO (Proxy k)
insert _ k v (Table m)          = return $ Table (M.insert k v m)
insert period k v px@(NN netT netW tp config) = do
  replMem' <- addToReplayMemory (k, scaleValue (getMinMaxVal px) v) (config ^. replayMemory)
  updateTargetNet <$> trainNNConf (replayMemory .~ replMem' $ config)
  where
    trainNNConf config' = do
      rands <- getRandomReplayMemoryElements period (config' ^. trainBatchSize) (config' ^. replayMemory)
      let trainingInstances = map (first $ config' ^. toNetInp) rands
          netW' = trainNetwork (config' ^. learningParams) netW trainingInstances
      return $ NN netT netW' tp config'
    updateTargetNet px'@(NN _ nW _ _)
      | period `mod` config ^. updateTargetInterval == 0 = NN nW nW tp config
      | otherwise = px'
    updateTargetNet _ = error "updateTargetNet called on non-neural network proxy"

-- | Retrieve a value.
lookupProxy :: (Ord k) => Period -> LookupType -> k -> Proxy k -> Double
lookupProxy _ _ k (Table m) = M.findWithDefault 0 k m
lookupProxy period lkTp k px
  -- | period < 1000 = 0 -- fromIntegral period / 1000 * (unscaleValue (getMinMaxVal px) $ lookupNeuralNetwork lkTp k px)
  | otherwise = lookupNeuralNetwork lkTp k px


-- | Retrieve a value from a neural network proxy. For other proxies an error is thrown. The returned value is up-scaled
-- to the original interval before returned.
lookupNeuralNetwork :: LookupType -> k -> Proxy k -> Double
lookupNeuralNetwork Worker k px@(NN _ netW _ conf) = unscaleValue (getMinMaxVal px) $ head $ snd $ fromLastShapes netW $ runNetwork netW (toHeadShapes netW $ (conf ^. toNetInp) k)
lookupNeuralNetwork Target k px@(NN netT _ _ conf) = unscaleValue (getMinMaxVal px) $ head $ snd $ fromLastShapes netT $ runNetwork netT (toHeadShapes netT $ (conf ^. toNetInp) k)
lookupNeuralNetwork _ _ _ = error "lookupNeuralNetwork called on non-neural network proxy"


-- | Finds the correct value for scaling.
getMinMaxVal :: Proxy k -> (MinValue,MaxValue)
getMinMaxVal Table {} = (1,1)
getMinMaxVal p@NN {}  = case p ^?! proxyType of
  VTable  -> (p ^?! proxyNNConfig.scaleParameters.scaleMinVValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxVValue)
  WTable  -> (p ^?! proxyNNConfig.scaleParameters.scaleMinWValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxWValue)
  R0Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinR0Value, p ^?! proxyNNConfig.scaleParameters.scaleMaxR0Value)
  R1Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinR1Value, p ^?! proxyNNConfig.scaleParameters.scaleMaxR1Value)
