{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE OverloadedStrings         #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TypeFamilies              #-}

module ML.ARAL.Proxy.Type
  ( ProxyType (..)
  , Proxy (Scalar, Table, Grenade, Hasktorch, CombinedProxy, RegressionProxy)
  , prettyProxyType
  , proxyScalar
  , proxyTable
  , proxyRegressionLayer
  , proxyDefault
  , proxyType
  , proxyNNConfig
  , _proxyNNWelford
  , proxyNrActions
  , proxyNrAgents
  , proxySub
  , proxyOutCol
  , proxyExpectedOutput
  , proxyWelford
  , _proxyNNTarget
  , _proxyNNWorker
  , _proxyHTTarget
  , _proxyHTWelford
  , _proxyHTWorker
  , _proxyHTAdam
  , _proxyHTModelSpec
  , isNeuralNetwork
  , isGrenade
  , isHasktorch
  , isCombinedProxy
  , isTable
  , proxyTypeName
  )
where

import           Control.DeepSeq
import           Control.Lens
import           Data.Constraint                             (Dict (..))
import qualified Data.Map.Strict                             as M
import           Data.Serialize
import           Data.Singletons                             (SingI)
import           Data.Singletons.Prelude.List
import qualified Data.Text                                   as Text
import           Data.Typeable                               (Typeable, cast)
import qualified Data.Vector                                 as VB
import qualified Data.Vector.Storable                        as V
import           GHC.Generics
import           GHC.TypeLits
import           Grenade
import           Statistics.Sample.WelfordOnlineMeanVariance
import qualified Torch                                       as Torch
import qualified Torch.Serialize                             as Torch
import qualified Torch.Typed.Vision                          as Torch (initMnist)
import qualified Torch.Vision                                as Torch.V
import           Unsafe.Coerce                               (unsafeCoerce)

-- import           ML.ARAL.Proxy.Regression.RegressionLayer
import           RegNet

import           ML.ARAL.NeuralNetwork
import           ML.ARAL.NeuralNetwork.AdamW
import           ML.ARAL.NeuralNetwork.Hasktorch
import           ML.ARAL.Types                               as T


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
  | NoScaling !ProxyType (Maybe [(MinValue Double, MaxValue Double)] )
  deriving (Eq, Ord, Show, NFData, Generic, Serialize)

proxyTypeName :: ProxyType -> Text.Text
proxyTypeName VTable           = "v"
proxyTypeName WTable           = "w"
proxyTypeName R0Table          = "r0"
proxyTypeName R1Table          = "r1"
proxyTypeName PsiVTable        = "psiV"
proxyTypeName PsiWTable        = "psiW"
proxyTypeName CombinedUnichain = "combinedUnichain"
proxyTypeName (NoScaling p _)  = "noscaling-" <> proxyTypeName p


data Proxy
  = Scalar -- ^ Combines multiple proxies in one for performance benefits.
      { _proxyScalar    :: !(V.Vector Double) -- ^ One value for each agent
      , _proxyNrActions :: !Int
      }
  | Table -- ^ Representation using a table.
      { _proxyTable     :: !(M.Map (StateFeatures, ActionIndex) (V.Vector Double)) -- ^ Shared state and one action for each agent, returns one value for each agent.
      , _proxyDefault   :: !(V.Vector Double)
      , _proxyNrActions :: !Int
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
                                , _proxyNrAgents  :: !Int
                                , _proxyNNWelford :: !(WelfordExistingAggregate StateFeatures)
                                }
  | CombinedProxy
      { _proxySub            :: !Proxy                                            -- ^ The actual proxy holding all combined values.
      , _proxyOutCol         :: !Int                                              -- ^ Index of data
      , _proxyExpectedOutput :: ![[((StateFeatures, AgentActionIndices, RewardValue, IsRandomAction), Value)]] -- ^ List of batches of list of n-step results.g Used to save the data for learning.
      }
  | Hasktorch
      { _proxyHTTarget    :: !MLP
      , _proxyHTWorker    :: !MLP
      , _proxyType        :: !ProxyType
      , _proxyNNConfig    :: !NNConfig
      , _proxyNrActions   :: !Int
      , _proxyNrAgents    :: !Int
      , _proxyHTAdam      :: !AdamW
      , _proxyHTModelSpec :: !MLPSpec
      , _proxyHTWelford   :: !(WelfordExistingAggregate StateFeatures)
      }
   | RegressionProxy
     { _proxyRegressionLayer :: !RegressionLayer -- One for each worker?
     , _proxyNrActions       :: !Int
     }

proxyScalar :: Traversal' Proxy (V.Vector Double)
proxyScalar f (Scalar x nrAs) = flip Scalar nrAs <$> f x
proxyScalar _ p               = pure p

proxyTable :: Traversal' Proxy (M.Map (StateFeatures, ActionIndex) (V.Vector Double))
proxyTable f (Table m d acts) = (\m' -> Table m' d acts) <$> f m
proxyTable  _ p               = pure p

proxyRegressionLayer :: Traversal' Proxy RegressionLayer
proxyRegressionLayer f (RegressionProxy ms acts) = (\ms' -> RegressionProxy ms' acts) <$> f ms
proxyRegressionLayer  _ p                        = pure p


proxyDefault :: Traversal' Proxy (V.Vector Double)
proxyDefault f (Table m d acts) = (\d' -> Table m d' acts) <$> f d
proxyDefault _ p                = pure p

proxyType :: Traversal' Proxy ProxyType
proxyType f (Grenade t w tp conf acts agents wel)            = (\tp' -> Grenade t w tp' conf acts agents wel) <$> f tp
proxyType f (Hasktorch t w tp conf acts agents adam mdl wel) = (\tp' -> Hasktorch t w tp' conf acts agents adam mdl wel) <$> f tp
proxyType f (CombinedProxy p c out)                          = (\tp' -> CombinedProxy (p { _proxyType = tp'}) c out) <$> f (_proxyType p)
proxyType  _ p                                               = pure p

proxyNNConfig :: Traversal' Proxy NNConfig
proxyNNConfig f (Grenade t w tp conf acts agents wel)            = (\conf' -> Grenade t w tp conf' acts agents wel) <$> f conf
proxyNNConfig f (Hasktorch t w tp conf acts agents adam mdl wel) = (\conf' -> Hasktorch t w tp conf' acts agents adam mdl wel) <$> f conf
proxyNNConfig f (CombinedProxy p c out)                          = (\conf' -> CombinedProxy (p { _proxyNNConfig = conf'}) c out) <$> f (_proxyNNConfig p)
proxyNNConfig  _ p                                               = pure p

proxyNrActions :: Traversal' Proxy Int
proxyNrActions f (Table m d acts)                                 = (\acts' -> Table m d acts') <$> f acts
proxyNrActions f (RegressionProxy ms acts)                        = (\acts' -> RegressionProxy ms acts') <$> f acts
proxyNrActions f (Grenade t w tp conf acts agents wel)            = (\acts' -> Grenade t w tp conf acts' agents wel) <$> f acts
proxyNrActions f (Hasktorch t w tp conf acts agents adam mdl wel) = (\acts' -> Hasktorch t w tp conf acts' agents adam mdl wel) <$> f acts
proxyNrActions f (CombinedProxy p c out)                          = (\acts' -> CombinedProxy (p { _proxyNrActions = acts'}) c out) <$> f (_proxyNrActions p)
proxyNrActions  _ p                                               = pure p

proxyNrAgents :: Traversal' Proxy Int
proxyNrAgents f (Grenade t w tp conf acts agents wel)            = (\agents' -> Grenade t w tp conf acts agents' wel) <$> f agents
proxyNrAgents f (Hasktorch t w tp conf acts agents adam mdl wel) = (\agents' -> Hasktorch t w tp conf acts agents' adam mdl wel) <$> f agents
proxyNrAgents f (CombinedProxy p c out)                          = (\agents' -> CombinedProxy (p {_proxyNrAgents = agents'}) c out) <$> f (_proxyNrAgents p)
proxyNrAgents _ p                                                = pure p

proxyWelford :: Traversal' Proxy (WelfordExistingAggregate StateFeatures)
proxyWelford f (Grenade t w tp conf acts agents wel)            = (\wel' -> Grenade t w tp conf acts agents wel') <$> f wel
proxyWelford f (Hasktorch t w tp conf acts agents adam mdl wel) = (\wel' -> Hasktorch t w tp conf acts agents adam mdl wel') <$> f wel
proxyWelford _ p                                                = pure p


proxySub :: Traversal' Proxy Proxy
proxySub f (CombinedProxy p c out) = (\p' -> CombinedProxy p' c out) <$> f p
proxySub _ p                       = pure p

proxyOutCol :: Traversal' Proxy Int
proxyOutCol f (CombinedProxy p c out) = (\c' -> CombinedProxy p c' out) <$> f c
proxyOutCol _ p                       = pure p

proxyExpectedOutput :: Traversal' Proxy [[((StateFeatures, AgentActionIndices, RewardValue, IsRandomAction), Value)]]
proxyExpectedOutput f (CombinedProxy p c out) = CombinedProxy p c <$> f out
proxyExpectedOutput _ p                       = pure p


instance Show Proxy where
  show (Scalar x _)              = "Scalar: " ++ show x
  show (Table t _ _)             = "Table: " ++ take 300 txt ++ (if length txt > 300 then "..." else "")
    where txt = show t
  show (Grenade _ _ t _ _ _ _)     = "Grenade " ++ show t
  show (Hasktorch _ _ t _ _ _ _ _ _) = "Hasktorch " ++ show t
  show (CombinedProxy p col _)   = "CombinedProxy of " ++ show p ++ " at column " ++ show col

prettyProxyType :: Proxy -> String
prettyProxyType Scalar{}              = "Scalar"
prettyProxyType Table{}               = "Tabular"
prettyProxyType RegressionProxy{}     = "RegressionProxy"
prettyProxyType Grenade{}             = "Grenade"
prettyProxyType Hasktorch{}           = "Hasktorch"
prettyProxyType (CombinedProxy p _ _) = "Combined Proxy built on " <> prettyProxyType p


instance NFData Proxy where
  rnf (Table x def acts)                           = rnf x `seq` rnf def `seq` rnf acts
  rnf (RegressionProxy x nrActs)                   = rnf x `seq` rnf nrActs
  rnf (Grenade t w tp cfg nrActs agents wel)       = rnf t `seq` rnf w `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs `seq` rnf agents `seq` rnf wel
  rnf (Hasktorch t w tp cfg nrActs agents _ _ wel) = rnf tp `seq` rnf cfg `seq` rnf nrActs `seq` rnf agents `seq` rnf wel
  rnf (Scalar x nrAs)                              = rnf x `seq` rnf nrAs
  rnf (CombinedProxy p nr xs)                      = rnf p `seq` rnf nr `seq` rnf xs


isNeuralNetwork :: Proxy -> Bool
isNeuralNetwork Grenade{}             = True
isNeuralNetwork Hasktorch{}           = True
isNeuralNetwork (CombinedProxy p _ _) = isNeuralNetwork p
isNeuralNetwork _                     = False

isGrenade :: Proxy -> Bool
isGrenade Grenade{}             = True
isGrenade (CombinedProxy p _ _) = isGrenade p
isGrenade _                     = False

isHasktorch :: Proxy -> Bool
isHasktorch Hasktorch{}           = True
isHasktorch (CombinedProxy p _ _) = isHasktorch p
isHasktorch _                     = False

isCombinedProxy :: Proxy -> Bool
isCombinedProxy CombinedProxy{} = True
isCombinedProxy _               = False

isTable :: Proxy -> Bool
isTable Table{}               = True
isTable (CombinedProxy p _ _) = isTable p
isTable _                     = False
