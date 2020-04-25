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

-- | Merge the proxies of a list of proxies. The replay memory will not be merged, but the one of the first element is returned!
mergeProxies :: Algorithm s -> [Proxies] -> Proxies
mergeProxies _ [] = error "empty input in mergeProxies"
mergeProxies _ [px] = px
mergeProxies alg (px:pxs) = avgProxies $ foldl' addProxies px pxs
  where
    ifBorl2 f x1 x2
      | isAlgBorl alg = f x1 x2
      | otherwise = x2
    ifDqnAvgRewardAdjusted2 f x1 x2
      | isAlgDqnAvgRewardAdjusted alg = f x1 x2
      | otherwise = x2
    ifBorl1 f x1 x2
      | isAlgBorl alg = f x1 x2
      | otherwise = x1
    ifDqnAvgRewardAdjusted1 f x1 x2
      | isAlgDqnAvgRewardAdjusted alg = f x1 x2
      | otherwise = x1
    avgProxies = scaleProxies (1 / (1 + fromIntegral (length pxs)))
    scaleProxies n (Proxies pRMin pRho pPsiV pV pPsiW pW pR0 pR1 rep) =
      Proxies
        (multiplyWorkerProxy n pRMin                       `using` rparWith rdeepseq)
        (multiplyWorkerProxy n pRho                        `using` rparWith rdeepseq)
        (ifBorl2 multiplyWorkerProxy n pPsiV               `using` rparWith rdeepseq)
        (ifBorl2 multiplyWorkerProxy n pV                  `using` rparWith rdeepseq)
        (ifBorl2 multiplyWorkerProxy n pPsiW               `using` rparWith rdeepseq)
        (ifBorl2 multiplyWorkerProxy n pW                  `using` rparWith rdeepseq)
        (ifDqnAvgRewardAdjusted2 multiplyWorkerProxy n pR0 `using` rparWith rdeepseq)
        (multiplyWorkerProxy n pR1                         `using` rparWith rdeepseq)
        rep
    scaleProxies n (ProxiesCombinedUnichain pRMin pRho pProxy rep) =
      ProxiesCombinedUnichain
        (multiplyWorkerProxy n pRMin  `using` rparWith rdeepseq)
        (multiplyWorkerProxy n pRho   `using` rparWith rdeepseq)
        (multiplyWorkerProxy n pProxy `using` rparWith rdeepseq)
        rep
    addProxies px1@Proxies {} px2@Proxies {} =
      Proxies
        (addWorkerProxy (view rhoMinimum px1) (view rhoMinimum px2)         `using` rpar)
        (addWorkerProxy (view rho px1) (view rho px2)                       `using` rpar)
        (ifBorl1 addWorkerProxy (view psiV px1) (view psiV px2)             `using` rpar)
        (ifBorl1 addWorkerProxy (view v px1) (view v px2)                   `using` rpar)
        (ifBorl1 addWorkerProxy (view psiW px1) (view psiW px2)             `using` rpar)
        (ifBorl1 addWorkerProxy (view w px1) (view w px2)                   `using` rpar)
        (ifDqnAvgRewardAdjusted1 addWorkerProxy (view r0 px1) (view r0 px2) `using` rpar)
        (addWorkerProxy (view r1 px1) (view r1 px2)                         `using` rpar)
        (view replayMemory px1)
    addProxies px1@ProxiesCombinedUnichain {} px2@ProxiesCombinedUnichain {} =
      ProxiesCombinedUnichain
        (addWorkerProxy (view rhoMinimum px1) (view rhoMinimum px2) `using` rpar)
        (addWorkerProxy (view rho px1) (view rho px2)               `using` rpar)
        (addWorkerProxy (view proxy px1) (view proxy px2)           `using` rpar)
        (view replayMemory px1)
    addProxies px1 px2 = error $ "Cannot merge proxies of different types: " ++ show (px1, px2)

-- | Replaces only the target networks, scalars and tables from the first proxies to the second one. Everything else stays untouched.
replaceTargetProxiesFromTo :: Algorithm s -> Proxies -> Proxies -> Proxies
replaceTargetProxiesFromTo alg px1@Proxies {} px2@Proxies {} =
  Proxies
    (replaceTargetProxyFromTo (view rhoMinimum px1) (view rhoMinimum px2))
    (replaceTargetProxyFromTo (view rho px1) (view rho px2))
    (ifBorl1 replaceTargetProxyFromTo (view psiV px1) (view psiV px2))
    (ifBorl1 replaceTargetProxyFromTo (view v px1) (view v px2))
    (ifBorl1 replaceTargetProxyFromTo (view psiW px1) (view psiW px2))
    (ifBorl1 replaceTargetProxyFromTo (view w px1) (view w px2))
    (ifDqnAvgRewardAdjusted1 replaceTargetProxyFromTo (view r0 px1) (view r0 px2))
    (replaceTargetProxyFromTo (view r1 px1) (view r1 px2))
    (view replayMemory px2)
  where
    ifBorl1 f x1 x2
      | isAlgBorl alg = f x1 x2
      | otherwise = x1
    ifDqnAvgRewardAdjusted1 f x1 x2
      | isAlgDqnAvgRewardAdjusted alg = f x1 x2
      | otherwise = x1
replaceTargetProxiesFromTo alg px1@ProxiesCombinedUnichain {} px2@ProxiesCombinedUnichain {} =
      ProxiesCombinedUnichain
        (replaceTargetProxyFromTo (view rhoMinimum px1) (view rhoMinimum px2))
        (replaceTargetProxyFromTo (view rho px1) (view rho px2))
        (replaceTargetProxyFromTo (view proxy px1) (view proxy px2))
        (view replayMemory px1)
replaceTargetProxiesFromTo _ px1 px2 = error $ "Cannot replace proxies of different types: " ++ show (px1, px2)


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
