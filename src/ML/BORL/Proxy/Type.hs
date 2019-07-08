{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE TemplateHaskell           #-}
{-# LANGUAGE TypeFamilies              #-}

module ML.BORL.Proxy.Type where

import           ML.BORL.NeuralNetwork
import           ML.BORL.Types                as T

import           Control.Arrow                (first, second)
import           Control.DeepSeq
import           Control.Lens
import qualified Data.Map.Strict              as M
import           Data.Serialize
import qualified Data.Set                     as S
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
  | PsiVTable
  deriving (Show, NFData, Generic, Serialize)

data LookupType = Target | Worker

type TableStateGeneraliser s = s -> [Double]


data Proxy s = Scalar           -- ^ Combines multiple proxies in one for performance benefits.
               { _proxyScalar :: !Double
               }
             | Table            -- ^ Representation using a table.
               { _proxyTable            :: !(M.Map ([Double],ActionIndex) Double)
               , _proxyDefault          :: !Double
               , _proxyStateGeneraliser :: !(TableStateGeneraliser s)
               }
             | forall nrL nrH shapes layers. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, NFData (Tapes layers shapes), NFData (Network layers shapes), Serialize (Network layers shapes)) =>
                Grenade         -- ^ Use Grenade neural networks.
                { _proxyNNTarget  :: !(Network layers shapes)
                , _proxyNNWorker  :: !(Network layers shapes)
                , _proxyNNStartup :: !(M.Map ([Double],ActionIndex) Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !(NNConfig s)
                , _proxyNrActions :: !Int
                }
             | TensorflowProxy  -- ^ Use Tensorflow neural networks.
                { _proxyTFTarget  :: TensorflowModel'
                , _proxyTFWorker  :: TensorflowModel'
                , _proxyNNStartup :: !(M.Map ([Double],ActionIndex) Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !(NNConfig s)
                , _proxyNrActions :: !Int
                }
makeLenses ''Proxy


instance (NFData s) => NFData (Proxy s) where
  rnf (Table x def gen)           = rnf x `seq` rnf def `seq` rnf gen
  rnf (Grenade t w tab tp cfg nrActs) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs
  rnf (TensorflowProxy t w tab tp cfg nrActs) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs
  rnf (Scalar x) = rnf x

mapProxyForSerialise :: (Ord s') => Proxy s -> Proxy s'
mapProxyForSerialise (Scalar x)          = Scalar x
mapProxyForSerialise (Table tbl def gen) = Table tbl def (const [])
mapProxyForSerialise (Grenade t w st tp config nr) = Grenade t w st tp (mapNNConfigForSerialise config) nr
mapProxyForSerialise (TensorflowProxy t w st tp config nr) = TensorflowProxy t w st tp (mapNNConfigForSerialise config) nr


multiplyProxy :: Double -> Proxy s -> Proxy s
multiplyProxy v (Scalar x) = Scalar (v*x)
multiplyProxy v (Table m d g) = Table (fmap (v*) m) d g
multiplyProxy v (Grenade t w s tp config nr) = Grenade t w s tp (over scaleParameters (multiplyScale (1/v)) config) nr
multiplyProxy v (TensorflowProxy t w s tp config nr) = TensorflowProxy t w s tp (over scaleParameters (multiplyScale (1/v)) config) nr


isNeuralNetwork :: Proxy s -> Bool
isNeuralNetwork Grenade{}         = True
isNeuralNetwork TensorflowProxy{} = True
isNeuralNetwork _                 = False

isTensorflow :: Proxy s -> Bool
isTensorflow TensorflowProxy{} = True
isTensorflow _                 = False


isTable :: Proxy s -> Bool
isTable Table{} = True
isTable _       = False


data Proxies s = Proxies        -- ^ This data type holds all data for BORL.
  { _rhoMinimum   :: !(Proxy s)
  , _rho          :: !(Proxy s)
  , _psiV         :: !(Proxy s)
  , _v            :: !(Proxy s)
  , _w            :: !(Proxy s)
  , _r0           :: !(Proxy s)
  , _r1           :: !(Proxy s)
  , _replayMemory :: !(Maybe (ReplayMemory s))
  } deriving (Generic)
makeLenses ''Proxies

instance NFData s => NFData (Proxies s) where
  rnf (Proxies rhoMin rho psiV v w r0 r1 repMem) = rnf rhoMin `seq` rnf rho `seq` rnf psiV
    `seq` rnf v `seq` rnf w `seq` rnf r0 `seq` rnf r1 `seq` rnf repMem


allProxies :: Proxies s -> [Proxy s]
allProxies pxs = [pxs ^. rhoMinimum, pxs ^. rho, pxs ^. psiV, pxs ^. v, pxs ^. w, pxs ^. r0, pxs ^. r1]


-- | Note: Only to be used for serialisation, as not all values are converted!
mapProxiesForSerialise :: (Ord s') => Period -> (s -> s') -> Proxies s -> Proxies s'
mapProxiesForSerialise t f (Proxies rm rho psiV v w r0 r1 replMem) =
  Proxies
    (mapProxyForSerialise rm)
    (mapProxyForSerialise rho)
    (mapProxyForSerialise psiV)
    (mapProxyForSerialise v)
    (mapProxyForSerialise w)
    (mapProxyForSerialise r0)
    (mapProxyForSerialise r1)
    (fmap (mapReplayMemoryForSeialisable f) replMem)
