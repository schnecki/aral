{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE ExplicitForAll            #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE TemplateHaskell           #-}

module ML.BORL.Proxy
    ( Proxy (..)
    , ProxyType (..)
    , NNConfig (..)
    , insert
    , findWithDefault
    , findNeuralNetwork
    ) where


import           ML.BORL.NeuralNetwork

import           Control.Arrow
import           Control.Lens
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

-- Todo: 2 Networks (target, worker)
data Proxy k = Table
               { _proxyTable :: !(M.Map k Double)
               }
             | forall nrL nrH shapes layers. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL) =>
                NN
                { _proxyNN       :: !(Network layers shapes)
                , _proxyType     :: !ProxyType
                , _proxyNNConfig :: !(NNConfig k)
                }
makeLenses ''Proxy

-- | Insert (or update) a value. The provided value will may be down-scaled to the interval [-1,1].
insert :: (Ord k) => k -> Double -> Proxy k -> Proxy k
insert k v (Table m)          = Table (M.insert k v m)
insert k v px@(NN net tp config) = checkTrainBatchsize ((k, scaleValue (getMaxVal px) v) : config ^. cache)
  where
    checkTrainBatchsize cache'
      | length cache' >= config ^. trainBatchSize = NN (trainNetwork (config ^. learningParams) net (map (first $ config ^. toNetInp) cache')) tp (cache .~ [] $ config)
      | otherwise = NN net tp (cache .~  cache' $ config)


-- | Retrieve a value.
findWithDefault :: (Ord k) => Integer -> k -> Proxy k -> Double
findWithDefault _ k (Table m) = M.findWithDefault 0 k m
findWithDefault period k px  | period < 1000 = fromIntegral period/1000 * (unscaleValue (getMaxVal px) $ findNeuralNetwork k px)
                             | otherwise = unscaleValue (getMaxVal px) $ findNeuralNetwork k px


-- | Retrieve a value from a neural network proxy. For other proxies an error is thrown. The returned value is up-scaled
-- to the original interval before returned.
findNeuralNetwork :: k -> Proxy k -> Double
findNeuralNetwork k px@(NN net _ conf)= head $ snd $ fromLastShapes net $ runNetwork net (toHeadShapes net $ (conf ^. toNetInp) k)
findNeuralNetwork _ _ = error "findNeuralNetwork called on non-neural network proxy"


-- | Finds the correct value for scaling.
getMaxVal :: Proxy k -> MaxValue
getMaxVal Table {} = 1
getMaxVal p@NN {}  = case p ^?! proxyType of
  VTable  -> p ^?! proxyNNConfig.scaleParameters.scaleMaxVValue
  WTable  -> p ^?! proxyNNConfig.scaleParameters.scaleMaxWValue
  R0Table -> p ^?! proxyNNConfig.scaleParameters.scaleMaxR0Value
  R1Table -> p ^?! proxyNNConfig.scaleParameters.scaleMaxR1Value
