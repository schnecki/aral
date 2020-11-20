{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TypeFamilies              #-}


module ML.BORL.Proxy.Proxies where

import           Control.DeepSeq
import           Control.Lens
import           Control.Parallel.Strategies (rdeepseq, rpar, rparWith, using)
import           Data.List                   (foldl', maximumBy)
import           Data.Ord
import           GHC.Generics
import           ML.BORL.Proxy.Type

import           ML.BORL.Algorithm
import           ML.BORL.NeuralNetwork


data Proxies =
  Proxies -- ^ This data type holds all data for BORL.
    { _rhoMinimum   :: !Proxy
    , _rho          :: !Proxy
    , _psiV         :: !Proxy
    , _v            :: !Proxy
    , _psiW         :: !Proxy
    , _w            :: !Proxy
    , _r0           :: !Proxy
    , _r1           :: !Proxy
    , _replayMemory :: !(Maybe ReplayMemories)
    } | ProxiesCombinedUnichain
    { _rhoMinimum   :: !Proxy
    , _rho          :: !Proxy
    , _proxy        :: !Proxy
    , _replayMemory :: !(Maybe ReplayMemories)
    }
  deriving (Generic, Show)


isCombinedProxies :: Proxies -> Bool
isCombinedProxies Proxies{}                 = False
isCombinedProxies ProxiesCombinedUnichain{} = True

allProxiesLenses :: Functor f => Proxies -> [(Proxy -> f Proxy) -> Proxies -> f Proxies]
allProxiesLenses Proxies {}                 = [rhoMinimum, rho, psiV, v, psiW, w, r0, r1]
allProxiesLenses ProxiesCombinedUnichain {} = [rhoMinimum, rho, proxy]

rhoMinimum :: Lens' Proxies Proxy
rhoMinimum f px@Proxies{}  = (\rhoMinimum' -> px { _rhoMinimum = rhoMinimum' }) <$> f (_rhoMinimum px)
rhoMinimum f px@ProxiesCombinedUnichain{}  = (\rhoMinimum' -> px { _rhoMinimum = rhoMinimum' }) <$> f (_rhoMinimum px)

rho :: Lens' Proxies Proxy
rho f px@Proxies{}                 = (\rho' -> px { _rho = rho' }) <$> f (_rho px)
rho f px@ProxiesCombinedUnichain{} = (\rho' -> px { _rho = rho' }) <$> f (_rho px)


fromCombinedIndex :: Int -> ProxyType
fromCombinedIndex 0 = R1Table
fromCombinedIndex 1 = R0Table
fromCombinedIndex 2 = VTable
fromCombinedIndex 3 = PsiVTable
fromCombinedIndex 4 = PsiWTable
fromCombinedIndex 5 = WTable
fromCombinedIndex _ = error "Proxies.hs fromCombinedIndex: Out of Range, Proxy does not exist!"


r1 :: Lens' Proxies Proxy
r1 f px@Proxies{}  = (\r1' -> px { _r1 = r1' }) <$> f (_r1 px)
r1 f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = x}) <$> f (CombinedProxy (_proxy px) 0 [])

r0 :: Lens' Proxies Proxy
r0 f px@Proxies{}  = (\r0' -> px { _r0 = r0' }) <$> f (_r0 px)
r0 f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = x}) <$> f (CombinedProxy (_proxy px) 1 [])

v :: Lens' Proxies Proxy
v f px@Proxies{}  = (\v' -> px { _v = v' }) <$> f (_v px)
v f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = x}) <$> f (CombinedProxy (_proxy px) 2 [])

psiV :: Lens' Proxies Proxy
psiV f px@Proxies{}                  = (\psiV' -> px { _psiV = psiV' }) <$> f (_psiV px)
psiV f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = x}) <$> f (CombinedProxy (_proxy px) 3 [])

psiW :: Lens' Proxies Proxy
psiW f px@Proxies{}  = (\psiW' -> px { _psiW = psiW' }) <$> f (_psiW px)
psiW f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = x}) <$> f (CombinedProxy (_proxy px) 4 [])

w :: Lens' Proxies Proxy
w f px@Proxies{}              = (\w' -> px { _w = w' }) <$> f (_w px)
w f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = x}) <$> f (CombinedProxy (_proxy px) 5 [])

proxy :: Lens' Proxies Proxy
proxy f px@ProxiesCombinedUnichain{} = (\x -> px {_proxy = x}) <$> f (_proxy px)
proxy f px@Proxies{} = error "calling proxy on Proxies in ML.BORL.Proxy.Proxies"

replayMemory :: Lens' Proxies (Maybe ReplayMemories)
replayMemory f px  = (\replayMemory' -> px { _replayMemory = replayMemory' }) <$> f (_replayMemory px)

instance NFData Proxies where
  rnf (Proxies rhoMin rho psiV v psiW w r0 r1 repMem) = rnf rhoMin `seq` rnf rho `seq` rnf psiV `seq` rnf v `seq` rnf psiW `seq` rnf w `seq` rnf r0 `seq` rnf r1 `seq` rnf1 repMem
  rnf (ProxiesCombinedUnichain rhoMin rho proxy replMem) = rnf rhoMin `seq` rnf rho `seq` rnf proxy `seq` rnf1 replMem
