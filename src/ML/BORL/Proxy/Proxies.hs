{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE TemplateHaskell           #-}
{-# LANGUAGE TypeFamilies              #-}


module ML.BORL.Proxy.Proxies where

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
    , _psiW2        :: !Proxy
    , _w2           :: !Proxy
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

-- type Lens s a = forall (f :: * -> *). Functor f => (a -> f a) -> s -> f s
-- x :: Lens' FooBar Int
-- x f (Foo a b) = (\a' -> Foo a' b) <$> f a
-- x f (Bar a)   = Bar <$> f a

-- type Traversal' s a = forall (f :: * -> *). Applicative f => (a -> f a) -> s -> f s
-- y :: Traversal' FooBar Int
-- y f (Foo a b) = (\b' -> Foo a  b') <$> f b
-- y _ c@(Bar _) = pure c

rhoMinimum :: Lens' Proxies Proxy
rhoMinimum f px@Proxies{}  = (\rhoMinimum' -> px { _rhoMinimum = rhoMinimum' }) <$> f (_rhoMinimum px)
rhoMinimum f px@ProxiesCombinedUnichain{}  = (\rhoMinimum' -> px { _rhoMinimum = rhoMinimum' }) <$> f (_rhoMinimum px)

rho :: Lens' Proxies Proxy
rho f px@Proxies{}                 = (\rho' -> px { _rho = rho' }) <$> f (_rho px)
rho f px@ProxiesCombinedUnichain{} = (\rho' -> px { _rho = rho' }) <$> f (_rho px)

psiV :: Getter Proxies Proxy
psiV f px@Proxies{}             = (\psiV' -> px { _psiV = psiV' }) <$> f (_psiV px)
psiV f px@ProxiesCombinedUnichain{} = (\x -> px { _proxy = CombinedProxy x 0 []})  <$> f (_proxy px)

v :: Getter Proxies Proxy
v f px@Proxies{}  = (\v' -> px { _v = v' }) <$> f (_v px)
v f px@ProxiesCombinedUnichain{} = (\x -> px { _proxy = CombinedProxy x 1 []})  <$> f (_proxy px)

psiW :: Getter Proxies Proxy
psiW f px@Proxies{}  = (\psiW' -> px { _psiW = psiW' }) <$> f (_psiW px)
psiW f px@ProxiesCombinedUnichain{} = (\x -> px { _proxy = CombinedProxy x 2 []})  <$> f (_proxy px)

w :: Getter Proxies Proxy
w f px@Proxies{}              = (\w' -> px { _w = w' }) <$> f (_w px)
w f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = CombinedProxy x 3 []}) <$> f (_proxy px)

psiW2 :: Getter Proxies Proxy
psiW2 f px@Proxies{}  = (\psiW2' -> px { _psiW2 = psiW2' }) <$> f (_psiW2 px)
psiW2 f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = CombinedProxy x 4 []}) <$> f (_proxy px)


w2 :: Getter Proxies Proxy
w2 f px@Proxies{}  = (\w2' -> px { _w2 = w2' }) <$> f (_w2 px)
w2 f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = CombinedProxy x 5 []}) <$> f (_proxy px)

r0 :: Getter Proxies Proxy
r0 f px@Proxies{}  = (\r0' -> px { _r0 = r0' }) <$> f (_r0 px)
r0 f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = CombinedProxy x 6 []}) <$> f (_proxy px)

r1 :: Getter Proxies Proxy
r1 f px@Proxies{}  = (\r1' -> px { _r1 = r1' }) <$> f (_r1 px)
r1 f px@ProxiesCombinedUnichain {} = (\x -> px {_proxy = CombinedProxy x 7 []}) <$> f (_proxy px)

replayMemory :: Lens' Proxies (Maybe ReplayMemory)
replayMemory f px  = (\replayMemory' -> px { _replayMemory = replayMemory' }) <$> f (_replayMemory px)

instance NFData Proxies where
  rnf (Proxies rhoMin rho psiV v psiW w psiW2 w2 r0 r1 repMem) =
    rnf rhoMin `seq` rnf rho `seq` rnf psiV `seq` rnf v `seq` rnf psiW `seq` rnf w `seq` rnf psiW2 `seq` rnf w2 `seq` rnf r0 `seq` rnf r1 `seq` rnf repMem
  rnf (ProxiesCombinedUnichain rhoMin rho proxy replMem) = rnf rhoMin `seq` rnf rho `seq` rnf proxy `seq` rnf replMem


