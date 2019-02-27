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

import           Control.DeepSeq
import           Control.Lens
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
  | PsiVTable
  | PsiWTable
  deriving (Show, NFData, Generic)

data LookupType = Target | Worker

-- Todo: 2 Networks (target, worker)
data Proxy s = Table
               { _proxyTable :: !(M.Map (s,ActionIndex) Double)
               }
             | forall nrL nrH shapes layers. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, NFData (Tapes layers shapes), NFData (Network layers shapes)) =>
                Grenade
                { _proxyNNTarget  :: !(Network layers shapes)
                , _proxyNNWorker  :: !(Network layers shapes)
                , _proxyNNStartup :: !(M.Map (s,ActionIndex) Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !(NNConfig s)
                , _proxyNrActions :: !Int
                }
             | TensorflowProxy
                { _proxyTFTarget  :: TensorflowModel'
                , _proxyTFWorker  :: TensorflowModel'
                , _proxyNNStartup :: !(M.Map (s,ActionIndex) Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !(NNConfig s)
                , _proxyNrActions :: !Int
                }
makeLenses ''Proxy

instance (NFData s) => NFData (Proxy s) where
  rnf (Table x)           = rnf x
  rnf (Grenade t w tab tp cfg nrActs) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs
  rnf (TensorflowProxy t w tab tp cfg nrActs) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs


