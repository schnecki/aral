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
import           Data.List                    (foldl')
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
  | W2Table
  | R0Table
  | R1Table
  | PsiVTable
  | PsiWTable
  | PsiW2Table
  | CombinedUnichain
  deriving (Eq, Ord, Show, NFData, Generic, Serialize)

data LookupType = Target | Worker

data Proxy = Scalar             -- ^ Combines multiple proxies in one for performance benefits.
               { _proxyScalar :: !Double
               }
             | Table            -- ^ Representation using a table.
               { _proxyTable   :: !(M.Map ([Double],ActionIndex) Double)
               , _proxyDefault :: !Double
               }
             | forall nrL nrH shapes layers. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, NFData (Tapes layers shapes), NFData (Network layers shapes), Serialize (Network layers shapes)) =>
                Grenade         -- ^ Use Grenade neural networks.
                { _proxyNNTarget  :: !(Network layers shapes)
                , _proxyNNWorker  :: !(Network layers shapes)
                , _proxyNNStartup :: !(M.Map ([Double],ActionIndex) Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !NNConfig
                , _proxyNrActions :: !Int
                }
             | TensorflowProxy  -- ^ Use Tensorflow neural networks.
                { _proxyTFTarget  :: TensorflowModel'
                , _proxyTFWorker  :: TensorflowModel'
                , _proxyNNStartup :: !(M.Map ([Double],ActionIndex) Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !NNConfig
                , _proxyNrActions :: !Int
                }
             | CombinedProxy
                { _subproxy  :: Proxy
                , _outputCol :: Int
                }
makeLenses ''Proxy


instance NFData Proxy where
  rnf (Table x def)           = rnf x `seq` rnf def
  rnf (Grenade t w tab tp cfg nrActs) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs
  rnf (TensorflowProxy t w tab tp cfg nrActs) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs
  rnf (Scalar x) = rnf x
  rnf (CombinedProxy p nr) = rnf p `seq` rnf nr


isNeuralNetwork :: Proxy -> Bool
isNeuralNetwork Grenade{}         = True
isNeuralNetwork TensorflowProxy{} = True
isNeuralNetwork _                 = False

isTensorflow :: Proxy -> Bool
isTensorflow TensorflowProxy{} = True
isTensorflow _                 = False


isTable :: Proxy -> Bool
isTable Table{} = True
isTable _       = False


