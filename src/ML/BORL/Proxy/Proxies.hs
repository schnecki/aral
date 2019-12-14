{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TypeFamilies              #-}


module ML.BORL.Proxy.Proxies where

import           ML.BORL.NeuralNetwork

import           Control.DeepSeq
import           Control.Lens
import           GHC.Generics
import           ML.BORL.Proxy.Type


-- class Proxies' pxs where
--   getRhoMinimum :: pxs -> Proxy
--   getRho :: pxs -> Proxy
--   getPsiV :: pxs -> Proxy
--   getV :: pxs -> Proxy
--   getPsiW :: pxs -> Proxy
--   getW :: pxs -> Proxy
--   getPsiW2 :: pxs -> Proxy
--   getW2 :: pxs -> Proxy
--   getR0 :: pxs -> Proxy
--   getR1 :: pxs -> Proxy
--   getReplayMemory :: pxs -> Proxy


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
    , _replayMemory :: !(Maybe ReplayMemory)
    } | ProxiesCombinedUnichain
    { _rhoMinimum   :: !Proxy
    , _rho          :: !Proxy
    , _proxy        :: !Proxy
    , _replayMemory :: !(Maybe ReplayMemory)
    }
  deriving (Generic)
-- makeLenses ''Proxies


 -- data FooBar
 --   = Foo { _x, _y :: Int }
 --   | Bar { _x :: Int }
 -- makeLenses ''FooBar

-- type Lens' s a = forall (f :: * -> *). Functor f => (a -> f a) -> s -> f s
-- x :: Lens' FooBar Int
-- x f (Foo a b) = (\a' -> Foo a' b) <$> f a
-- x f (Bar a)   = Bar <$> f a

-- allProxiesLenses :: [(Proxies -> f Proxy) -> Proxies -> f Proxy]
allProxiesLenses :: Functor f => Proxies -> [(Proxy -> f Proxy) -> Proxies -> f Proxies]
allProxiesLenses Proxies {}                 = [rhoMinimum, rho, psiV, v, psiW, w, r0, r1]
allProxiesLenses ProxiesCombinedUnichain {} = [rhoMinimum, rho, proxy]


rhoMinimum :: Lens' Proxies Proxy
rhoMinimum f px@Proxies{}  = (\rhoMinimum' -> px { _rhoMinimum = rhoMinimum' }) <$> f (_rhoMinimum px)
rhoMinimum f px@ProxiesCombinedUnichain{}  = (\rhoMinimum' -> px { _rhoMinimum = rhoMinimum' }) <$> f (_rhoMinimum px)

rho :: Lens' Proxies Proxy
rho f px@Proxies{}                 = (\rho' -> px { _rho = rho' }) <$> f (_rho px)
rho f px@ProxiesCombinedUnichain{} = (\rho' -> px { _rho = rho' }) <$> f (_rho px)

r1 :: Lens' Proxies Proxy
r1 f px@Proxies{}  = (\r1' -> px { _r1 = r1' }) <$> f (_r1 px)
r1 f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = x}) <$> f (CombinedProxy (_proxy px) 0 [])

r0 :: Lens' Proxies Proxy
r0 f px@Proxies{}  = (\r0' -> px { _r0 = r0' }) <$> f (_r0 px)
r0 f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = x}) <$> f (CombinedProxy (_proxy px) 1 [])


psiV :: Lens' Proxies Proxy
psiV f px@Proxies{}                  = (\psiV' -> px { _psiV = psiV' }) <$> f (_psiV px)
psiV f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = x}) <$> f (CombinedProxy (_proxy px) 2 [])

v :: Lens' Proxies Proxy
v f px@Proxies{}  = (\v' -> px { _v = v' }) <$> f (_v px)
v f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = x}) <$> f (CombinedProxy (_proxy px) 3 [])

psiW :: Lens' Proxies Proxy
psiW f px@Proxies{}  = (\psiW' -> px { _psiW = psiW' }) <$> f (_psiW px)
psiW f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = x}) <$> f (CombinedProxy (_proxy px) 4 [])

w :: Lens' Proxies Proxy
w f px@Proxies{}              = (\w' -> px { _w = w' }) <$> f (_w px)
w f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = x}) <$> f (CombinedProxy (_proxy px) 5 [])

proxy :: Lens' Proxies Proxy
proxy f px@ProxiesCombinedUnichain{} = (\x -> px {_proxy = x}) <$> f (_proxy px)
proxy f px@Proxies{} = error "calling proxy on Proxies in ML.BORL.Proxy.Proxies"

replayMemory :: Lens' Proxies (Maybe ReplayMemory)
replayMemory f px  = (\replayMemory' -> px { _replayMemory = replayMemory' }) <$> f (_replayMemory px)

instance NFData Proxies where
  rnf (Proxies rhoMin rho psiV v psiW w r0 r1 repMem) = rnf rhoMin `seq` rnf rho `seq` rnf psiV `seq` rnf v `seq` rnf psiW `seq` rnf w `seq` rnf r0 `seq` rnf r1 `seq` rnf repMem
  rnf (ProxiesCombinedUnichain rhoMin rho proxy replMem) = rnf rhoMin `seq` rnf rho `seq` rnf proxy `seq` rnf replMem
