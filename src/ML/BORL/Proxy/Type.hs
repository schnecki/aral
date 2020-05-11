{-# LANGUAGE BangPatterns              #-}
{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE OverloadedStrings         #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TypeFamilies              #-}

module ML.BORL.Proxy.Type
  ( ProxyType (..)
  , Proxy (Scalar, Table, Grenade, TensorflowProxy, CombinedProxy)
  , prettyProxyType
  , proxyScalar
  , proxyTable
  , proxyDefault
  , proxyTFTarget
  , proxyTFWorker
  , proxyType
  , proxyNNConfig
  , proxyNrActions
  , proxySub
  , proxyOutCol
  , proxyExpectedOutput
  , _proxyNNTarget
  , _proxyNNWorker
  , addWorkerProxy
  , multiplyWorkerProxy
  , replaceTargetProxyFromTo
  , isNeuralNetwork
  , isTensorflow
  , isGrenade
  , isCombinedProxy
  , isTable
  , proxyTypeName
  )
where

import           ML.BORL.NeuralNetwork
import           ML.BORL.Types                as T

import           Control.DeepSeq
import           Control.Lens
import           Data.Constraint              (Dict (..))
import qualified Data.Map.Strict              as M
import           Data.Serialize
import           Data.Singletons              (SingI)
import           Data.Singletons.Prelude.List
import qualified Data.Text                    as Text
import           Data.Typeable                (Typeable, cast)
import qualified Data.Vector.Storable         as V
import           GHC.Generics
import           GHC.TypeLits
import           Grenade
import           Unsafe.Coerce                (unsafeCoerce)

import           Debug.Trace

-- | Type of approximation (needed for scaling of values).
data ProxyType
  = VTable
  | WTable
  | R0Table
  | R1Table
  | PsiVTable
  | PsiWTable
  | CombinedUnichain
--  | CombinedUnichainScaleAs ProxyType
  | NoScaling !ProxyType (Maybe [(MinValue Float, MaxValue Float)] )
  deriving (Eq, Ord, Show, NFData, Generic, Serialize)

proxyTypeName :: ProxyType -> Text.Text
proxyTypeName VTable           = "v"
proxyTypeName WTable           = "w"
proxyTypeName R0Table          = "r0"
proxyTypeName R1Table          = "r1"
proxyTypeName PsiVTable        = "psiV"
proxyTypeName PsiWTable        = "psiW"
proxyTypeName CombinedUnichain = "combinedUnichain"
-- proxyTypeName (CombinedUnichainScaleAs p) = "combinedUnichainScaleAs" <> proxyTypeName p
proxyTypeName (NoScaling p _)  = "noscaling-" <> proxyTypeName p


data Proxy
  = Scalar -- ^ Combines multiple proxies in one for performance benefits.
      { _proxyScalar :: !Float
      }
  | Table -- ^ Representation using a table.
      { _proxyTable   :: !(M.Map (V.Vector Float, ActionIndex) Float)
      , _proxyDefault :: !Float
      }
  | forall nrH shapes layers. ( KnownNat nrH
                              , Head shapes ~ 'D1 nrH
                              , Typeable layers
                              , Typeable shapes
                              , GNum (Gradients layers)
                              , FoldableGradient (Gradients layers)
                              , GNum (Network layers shapes)
                              , SingI (Last shapes)
                              , FromDynamicLayer (Network layers shapes)
                              , NFData (Tapes layers shapes)
                              , NFData (Network layers shapes)
                              , NFData (Gradients layers)
                              , Serialize (Network layers shapes)
                              ) =>
                              Grenade -- ^ Use Grenade neural networks.
                                { _proxyNNTarget  :: !(Network layers shapes)
                                , _proxyNNWorker  :: !(Network layers shapes)
                                , _proxyType      :: !ProxyType
                                , _proxyNNConfig  :: !NNConfig
                                , _proxyNrActions :: !Int
                                }
  | TensorflowProxy -- ^ Use Tensorflow neural networks.
      { _proxyTFTarget  :: !TensorflowModel'
      , _proxyTFWorker  :: !TensorflowModel'
      , _proxyType      :: !ProxyType
      , _proxyNNConfig  :: !NNConfig
      , _proxyNrActions :: !Int
      }
  | CombinedProxy
      { _proxySub            :: Proxy -- ^ The actual proxy holding all combined values.
      , _proxyOutCol         :: Int -- ^ Output column/row of the data.
      , _proxyExpectedOutput :: [[((StateFeatures, ActionIndex), Float)]] -- ^ List of batches of list of n-step results. Used to save the data for learning.
      }

-- | This function adds two proxies. The proxies must be of the same type and for Grenade of the same shape. It does not work with Tenforflow proxies!
addWorkerProxy :: Proxy -> Proxy -> Proxy
addWorkerProxy (Scalar x) (Scalar y)   = Scalar (x+y)
addWorkerProxy (Table x d) (Table y _) = Table (mergeTables x y) d
addWorkerProxy (Grenade (target1 :: Network layers1 shapes1) worker1 tp1 nnCfg1 nrActs1) (Grenade (target2 :: Network layers2 shapes2) worker2 _ _ _) =
  case (cast worker2, cast target2) of
    (Just worker2', Just target2') ->
      Grenade
        (target1 |+ target2')  -- (disabled in multiplyWorkerProxy)
        (worker1 |+ worker2')
        tp1
        nnCfg1
        nrActs1
    _ -> error "cannot replace worker1 of different type"
addWorkerProxy TensorflowProxy{} _ = error "addWorkerProxy does not work on Tensorflow Proxies"
addWorkerProxy (CombinedProxy px1 outCol1 expOut1) (CombinedProxy px2 _ _) = CombinedProxy (addWorkerProxy px1 px2) outCol1 expOut1
addWorkerProxy x1 x2 = error $ "Cannot add proxies of differnt types: " ++ show (x1, x2)

multiplyWorkerProxy :: Float -> Proxy -> Proxy
multiplyWorkerProxy n (Scalar x)  = Scalar (n*x)
multiplyWorkerProxy n (Table x d) = Table (M.map (*n) x) d
multiplyWorkerProxy n (Grenade (target1 :: Network layers1 shapes1) worker1 tp1 nnCfg1 nrActs1) =
  Grenade
    (toRational n |* target1) -- (disabled in addWorkerProxy)
    (toRational n |* worker1)
    tp1 nnCfg1 nrActs1
multiplyWorkerProxy n TensorflowProxy{} = error "multiplyWorkerProxy does not work on Tensorflow Proxies"
multiplyWorkerProxy n (CombinedProxy px1 outCol1 expOut1) = CombinedProxy (multiplyWorkerProxy n px1) outCol1 expOut1


replaceTargetProxyFromTo :: Proxy -> Proxy -> Proxy
replaceTargetProxyFromTo (Scalar x) (Scalar _)   = Scalar x
replaceTargetProxyFromTo (Table x d) (Table _ _) = Table x d
replaceTargetProxyFromTo (Grenade (target1 :: Network layers1 shapes1) worker1 _ _ _) (Grenade (target2 :: Network layers2 shapes2) worker2 tp2 nnCfg2 nrActs2) =
  case cast worker1 of
    Nothing -> error "cannot replace target1 of different type"
    Just worker1' ->
      Grenade
        target2 -- (0.5 |* (target1' |+ target2))
        worker1' -- target1' -- (0.5 |* (worker1' |+ worker2))
        tp2
        nnCfg2
        nrActs2
replaceTargetProxyFromTo TensorflowProxy{} _ = error "replaceTargetProxy does not work on Tensorflow Proxies"
replaceTargetProxyFromTo (CombinedProxy px1 _ _) (CombinedProxy px2 outCol2 expOut2) = CombinedProxy (replaceTargetProxyFromTo px1 px2) outCol2 expOut2
replaceTargetProxyFromTo x1 x2 = error $ "Cannot replace proxies of differnt types: " ++ show (x1, x2)


mergeTables :: (Ord k, Num n) => M.Map k n -> M.Map k n -> M.Map k n
mergeTables = M.mergeWithKey (\_ v1 v2 -> Just (v1 + v2)) (M.map (*2)) (M.map (*2))

proxyScalar :: Traversal' Proxy Float
proxyScalar f (Scalar x) = Scalar <$> f x
proxyScalar _ p          = pure p

proxyTable :: Traversal' Proxy (M.Map (StateFeatures, ActionIndex) Float)
proxyTable f (Table m d) = flip Table d <$> f m
proxyTable  _ p          = pure p

proxyDefault :: Traversal' Proxy Float
proxyDefault f (Table m d) = Table m <$> f d
proxyDefault _ p           = pure p

proxyTFTarget :: Traversal' Proxy TensorflowModel'
proxyTFTarget f (TensorflowProxy t w tp conf acts) = (\t' -> TensorflowProxy t' w tp conf acts) <$> f t
proxyTFTarget _ p = pure p

proxyTFWorker :: Traversal' Proxy TensorflowModel'
proxyTFWorker f (TensorflowProxy t w tp conf acts) = (\t' -> TensorflowProxy t' w tp conf acts) <$> f t
proxyTFWorker _ p = pure p

proxyType :: Traversal' Proxy ProxyType
proxyType f (Grenade t w tp conf acts) = (\tp' -> Grenade t w tp' conf acts) <$> f tp
proxyType f (TensorflowProxy t w tp conf acts) = (\tp' -> TensorflowProxy t w tp' conf acts) <$> f tp
proxyType f (CombinedProxy p c out) = (\tp' -> CombinedProxy (p { _proxyType = tp'}) c out) <$> f (_proxyType p)
proxyType  _ p = pure p

proxyNNConfig :: Traversal' Proxy NNConfig
proxyNNConfig f (Grenade t w tp conf acts) = (\conf' -> Grenade t w tp conf' acts) <$> f conf
proxyNNConfig f (TensorflowProxy t w tp conf acts) = (\conf' -> TensorflowProxy t w tp conf' acts) <$> f conf
proxyNNConfig f (CombinedProxy p c out) = (\conf' -> CombinedProxy (p { _proxyNNConfig = conf'}) c out) <$> f (_proxyNNConfig p)
proxyNNConfig  _ p = pure p

proxyNrActions :: Traversal' Proxy Int
proxyNrActions f (Grenade t w tp conf acts) = (\acts' -> Grenade t w tp conf acts') <$> f acts
proxyNrActions f (TensorflowProxy t w tp conf acts) = (\acts' -> TensorflowProxy t w tp conf acts') <$> f acts
proxyNrActions f (CombinedProxy p c out) = (\acts' -> CombinedProxy (p { _proxyNrActions = acts'}) c out) <$> f (_proxyNrActions p)
proxyNrActions  _ p = pure p

proxySub :: Traversal' Proxy Proxy
proxySub f (CombinedProxy p c out) = (\p' -> CombinedProxy p' c out) <$> f p
proxySub _ p                       = pure p

proxyOutCol :: Traversal' Proxy Int
proxyOutCol f (CombinedProxy p c out) = (\c' -> CombinedProxy p c' out) <$> f c
proxyOutCol _ p                       = pure p

proxyExpectedOutput :: Traversal' Proxy [[((StateFeatures, ActionIndex), Float)]]
proxyExpectedOutput f (CombinedProxy p c out) = (\out' -> CombinedProxy p c out') <$> f out
proxyExpectedOutput _ p = pure p


instance Show Proxy where
  show (Scalar x)                  = "Scalar: " ++ show x
  show Table{}                     = "Table"
  show (Grenade _ _ t _ _)         = "Grenade " ++ show t
  show (TensorflowProxy _ _ t _ _) = "TensorflowProxy " ++ show t
  show (CombinedProxy p col _)     = "CombinedProxy of " ++ show p ++ " at column " ++ show col

prettyProxyType :: Proxy -> String
prettyProxyType Scalar{} = "Scalar"
prettyProxyType Table{} = "Tabular"
prettyProxyType Grenade{} = "Grenade"
prettyProxyType (TensorflowProxy _ w _ _ _) = "Tensorflow with " ++ show (map prettyOptimizerNames (optimizerVariables $ tensorflowModel w)) ++ " optimizer"
prettyProxyType (CombinedProxy p _ _) = "Combined Proxy built on " <> prettyProxyType p


instance NFData Proxy where
  rnf (Table x def) = rnf x `seq` rnf def
  rnf (Grenade t w tp cfg nrActs) = rnf t `seq` rnf w `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs
  rnf (TensorflowProxy t w tp cfg nrActs) = rnf t `seq` rnf w `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs
  rnf (Scalar x) = rnf x
  rnf (CombinedProxy p nr xs) = rnf p `seq` rnf nr `seq` rnf xs


isNeuralNetwork :: Proxy -> Bool
isNeuralNetwork Grenade{}             = True
isNeuralNetwork TensorflowProxy{}     = True
isNeuralNetwork (CombinedProxy p _ _) = isNeuralNetwork p
isNeuralNetwork _                     = False

isTensorflow :: Proxy -> Bool
isTensorflow TensorflowProxy{}     = True
isTensorflow (CombinedProxy p _ _) = isTensorflow p
isTensorflow _                     = False

isGrenade :: Proxy -> Bool
isGrenade Grenade{}             = True
isGrenade (CombinedProxy p _ _) = isTensorflow p
isGrenade _                     = False


isCombinedProxy :: Proxy -> Bool
isCombinedProxy CombinedProxy{} = True
isCombinedProxy _               = False

isTable :: Proxy -> Bool
isTable Table{}               = True
isTable (CombinedProxy p _ _) = isTable p
isTable _                     = False


