{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE OverloadedStrings         #-}
{-# LANGUAGE TemplateHaskell           #-}
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
  , proxyNNStartup
  , proxyType
  , proxyNNConfig
  , proxyNrActions
  , proxySub
  , proxyOutCol
  , proxyExpectedOutput
  , _proxyNNTarget
  , _proxyNNWorker
  , isNeuralNetwork
  , isTensorflow
  , isCombinedProxy
  , isTable
  , proxyTypeName
  )
where

import           ML.BORL.NeuralNetwork
import           ML.BORL.Types                as T

import           Control.DeepSeq
import           Control.Lens
import qualified Data.Map.Strict              as M
import           Data.Serialize
import           Data.Singletons.Prelude.List
import qualified Data.Text                    as Text
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
  | CombinedUnichain
--  | CombinedUnichainScaleAs ProxyType
  | NoScaling ProxyType
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
proxyTypeName (NoScaling p)    = "noscaling-" <> proxyTypeName p


data Proxy = Scalar             -- ^ Combines multiple proxies in one for performance benefits.
               { _proxyScalar :: !Double
               }
             | Table            -- ^ Representation using a table.
               { _proxyTable   :: !(M.Map ([Double], ActionIndex) Double)
               , _proxyDefault :: !Double
               }
             | forall nrL nrH shapes layers. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, GNum (Gradients layers),
                                              NFData (Tapes layers shapes), NFData (Network layers shapes), Serialize (Network layers shapes)) =>
                Grenade         -- ^ Use Grenade neural networks.
                { _proxyNNTarget  :: !(Network layers shapes)
                , _proxyNNWorker  :: !(Network layers shapes)
                , _proxyNNStartup :: !(M.Map ([Double], ActionIndex) Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !NNConfig
                , _proxyNrActions :: !Int
                }
             | TensorflowProxy  -- ^ Use Tensorflow neural networks.
                { _proxyTFTarget  :: !TensorflowModel'
                , _proxyTFWorker  :: !TensorflowModel'
                , _proxyNNStartup :: !(M.Map ([Double], ActionIndex) Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !NNConfig
                , _proxyNrActions :: !Int
                }
             | CombinedProxy
                { _proxySub            :: Proxy                                    -- ^ The actual proxy holding all combined values.
                , _proxyOutCol         :: Int                                      -- ^ Output column/row of the data.
                , _proxyExpectedOutput :: [((StateFeatures, ActionIndex), Double)] -- ^ Used to save the data for learning.
                }
-- makeLenses ''Proxy

proxyScalar :: Traversal' Proxy Double
proxyScalar f (Scalar x) = Scalar <$> f x
proxyScalar _ p          = pure p

proxyTable :: Traversal' Proxy (M.Map ([Double], ActionIndex) Double)
proxyTable f (Table m d) = flip Table d <$> f m
proxyTable  _ p          = pure p

proxyDefault :: Traversal' Proxy Double
proxyDefault f (Table m d) = Table m <$> f d
proxyDefault _ p           = pure p

proxyTFTarget :: Traversal' Proxy TensorflowModel'
proxyTFTarget f (TensorflowProxy t w s tp conf acts) = (\t' -> TensorflowProxy t' w s tp conf acts) <$> f t
proxyTFTarget _ p = pure p

proxyTFWorker :: Traversal' Proxy TensorflowModel'
proxyTFWorker f (TensorflowProxy t w s tp conf acts) = (\t' -> TensorflowProxy t' w s tp conf acts) <$> f t
proxyTFWorker _ p = pure p

proxyNNStartup :: Traversal' Proxy (M.Map ([Double], ActionIndex) Double)
proxyNNStartup f (Grenade t w s tp conf acts) = (\s' -> Grenade t w s' tp conf acts) <$> f s
proxyNNStartup f (TensorflowProxy t w s tp conf acts) = (\s' -> TensorflowProxy t w s' tp conf acts) <$> f s
proxyNNStartup f (CombinedProxy p c out) = (\s' -> CombinedProxy (p { _proxyNNStartup = s'}) c out) <$> f (_proxyNNStartup p)
proxyNNStartup  _ p = pure p

proxyType :: Traversal' Proxy ProxyType
proxyType f (Grenade t w s tp conf acts) = (\tp' -> Grenade t w s tp' conf acts) <$> f tp
proxyType f (TensorflowProxy t w s tp conf acts) = (\tp' -> TensorflowProxy t w s tp' conf acts) <$> f tp
proxyType f (CombinedProxy p c out) = (\tp' -> CombinedProxy (p { _proxyType = tp'}) c out) <$> f (_proxyType p)
proxyType  _ p = pure p

proxyNNConfig :: Traversal' Proxy NNConfig
proxyNNConfig f (Grenade t w s tp conf acts) = (\conf' -> Grenade t w s tp conf' acts) <$> f conf
proxyNNConfig f (TensorflowProxy t w s tp conf acts) = (\conf' -> TensorflowProxy t w s tp conf' acts) <$> f conf
proxyNNConfig f (CombinedProxy p c out) = (\conf' -> CombinedProxy (p { _proxyNNConfig = conf'}) c out) <$> f (_proxyNNConfig p)
proxyNNConfig  _ p = pure p

proxyNrActions :: Traversal' Proxy Int
proxyNrActions f (Grenade t w s tp conf acts) = (\acts' -> Grenade t w s tp conf acts') <$> f acts
proxyNrActions f (TensorflowProxy t w s tp conf acts) = (\acts' -> TensorflowProxy t w s tp conf acts') <$> f acts
proxyNrActions f (CombinedProxy p c out) = (\acts' -> CombinedProxy (p { _proxyNrActions = acts'}) c out) <$> f (_proxyNrActions p)
proxyNrActions  _ p = pure p

proxySub :: Traversal' Proxy Proxy
proxySub f (CombinedProxy p c out) = (\p' -> CombinedProxy p' c out) <$> f p
proxySub _ p                       = pure p

proxyOutCol :: Traversal' Proxy Int
proxyOutCol f (CombinedProxy p c out) = (\c' -> CombinedProxy p c' out) <$> f c
proxyOutCol _ p                       = pure p

proxyExpectedOutput :: Traversal' Proxy [((StateFeatures, ActionIndex), Double)]
proxyExpectedOutput f (CombinedProxy p c out) = (\out' -> CombinedProxy p c out') <$> f out
proxyExpectedOutput _ p = pure p


instance Show Proxy where
  show (Scalar x)              = "Scalar: " ++ show x
  show Table{}                 = "Table"
  show Grenade{}               = "Grenade"
  show TensorflowProxy{}       = "TensorflowProxy"
  show (CombinedProxy p col _) = "CombinedProxy of " ++ show p ++ " at row " ++ show col

prettyProxyType :: Proxy -> String
prettyProxyType Scalar{} = "Scalar"
prettyProxyType Table{} = "Tabular"
prettyProxyType Grenade{} = "Grenade with SGD (+ momentum + l2) optimizer"
prettyProxyType (TensorflowProxy _ w _ _ _ _) = "Tensorflow with " ++ show (map prettyOptimizerNames (optimizerVariables $ tensorflowModel w)) ++ " optimizer"
prettyProxyType (CombinedProxy p _ _) = "Combined Proxy built on " <> prettyProxyType p


instance NFData Proxy where
  rnf (Table x def)           = rnf x `seq` rnf def
  rnf (Grenade t w tab tp cfg nrActs) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs
  rnf (TensorflowProxy t w tab tp cfg nrActs) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs
  rnf (Scalar x) = rnf x
  rnf (CombinedProxy p nr xs) = rnf p `seq` rnf nr `seq` rnf xs


isNeuralNetwork :: Proxy -> Bool
isNeuralNetwork Grenade{}             = True
isNeuralNetwork TensorflowProxy{}     = True
isNeuralNetwork (CombinedProxy p _ _) = isNeuralNetwork p
isNeuralNetwork _                     = False

isTensorflow :: Proxy -> Bool
isTensorflow TensorflowProxy{}     = True
isTensorflow (CombinedProxy p _ _) = isNeuralNetwork p
isTensorflow _                     = False

isCombinedProxy :: Proxy -> Bool
isCombinedProxy CombinedProxy{} = True
isCombinedProxy _               = False

isTable :: Proxy -> Bool
isTable Table{}               = True
isTable (CombinedProxy p _ _) = isTable p
isTable _                     = False


