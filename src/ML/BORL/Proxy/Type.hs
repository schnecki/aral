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
  , Proxy (Scalar, Table, Grenade, CombinedProxy)
  , prettyProxyType
  , proxyScalar
  , proxyTable
  , proxyDefault
  , proxyType
  , proxyNNConfig
  , proxyNrActions
  , proxyNrAgents
  , proxySub
  , proxyOutCol
  , proxyExpectedOutput
  , _proxyNNTarget
  , _proxyNNWorker
  , isNeuralNetwork
  , isGrenade
  , isCombinedProxy
  , isTable
  , proxyTypeName
  )
where


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

import           ML.BORL.NeuralNetwork
import           ML.BORL.Types                as T


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
-- proxyTypeName (CombinedUnichainScaleAs p) = "combinedUnichainScaleAs" <> proxyTypeName p
proxyTypeName (NoScaling p _)  = "noscaling-" <> proxyTypeName p


data Proxy
  = Scalar -- ^ Combines multiple proxies in one for performance benefits.
      { _proxyScalar    :: !(V.Vector Double) -- ^ One value for each agent
      , _proxyNrActions :: !Int
      }
  | Table  -- ^ Representation using a table.
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
                                }
  | CombinedProxy
      { _proxySub            :: Proxy                                                -- ^ The actual proxy holding all combined values.
      , _proxyOutCol         :: Int                                                  -- ^ Index of data
      , _proxyExpectedOutput :: [[((StateFeatures, AgentActionIndices), Value)]]     -- ^ List of batches of list of n-step results. Used to save the data for learning.
      }

proxyScalar :: Traversal' Proxy (V.Vector Double)
proxyScalar f (Scalar x nrAs) = (flip Scalar nrAs) <$> f x
proxyScalar _ p               = pure p

proxyTable :: Traversal' Proxy (M.Map (StateFeatures, ActionIndex) (V.Vector Double))
proxyTable f (Table m d acts) = (\m' -> Table m' d acts) <$> f m
proxyTable  _ p               = pure p

proxyDefault :: Traversal' Proxy (V.Vector Double)
proxyDefault f (Table m d acts) = (\d' -> Table m d' acts) <$> f d
proxyDefault _ p                = pure p

proxyType :: Traversal' Proxy ProxyType
proxyType f (Grenade t w tp conf acts agents) = (\tp' -> Grenade t w tp' conf acts agents) <$> f tp
proxyType f (CombinedProxy p c out) = (\tp' -> CombinedProxy ( p { _proxyType = tp'}) c out) <$> f (_proxyType p)
proxyType  _ p = pure p

proxyNNConfig :: Traversal' Proxy NNConfig
proxyNNConfig f (Grenade t w tp conf acts agents) = (\conf' -> Grenade t w tp conf' acts agents) <$> f conf
proxyNNConfig f (CombinedProxy p c out) = (\conf' -> CombinedProxy (p { _proxyNNConfig = conf'}) c out) <$> f (_proxyNNConfig p)
proxyNNConfig  _ p = pure p

proxyNrActions :: Traversal' Proxy Int
proxyNrActions f (Table m d acts) = (\acts' -> Table m d acts') <$> f acts
proxyNrActions f (Grenade t w tp conf acts agents) = (\acts' -> Grenade t w tp conf acts' agents) <$> f acts
proxyNrActions f (CombinedProxy p c out) = (\acts' -> CombinedProxy (p { _proxyNrActions = acts'}) c out) <$> f (_proxyNrActions p)
proxyNrActions  _ p = pure p

proxyNrAgents :: Traversal' Proxy Int
proxyNrAgents f (Grenade t w tp conf acts agents) = (\agents' -> Grenade t w tp conf acts agents') <$> f agents
proxyNrAgents f (CombinedProxy p c out) = (\agents' -> CombinedProxy (p {_proxyNrAgents = agents'}) c out) <$> f (_proxyNrAgents p)
proxyNrAgents _ p = pure p

proxySub :: Traversal' Proxy Proxy
proxySub f (CombinedProxy p c out) = (\p' -> CombinedProxy p' c out) <$> f p
proxySub _ p                       = pure p

proxyOutCol :: Traversal' Proxy Int
proxyOutCol f (CombinedProxy p c out) = (\c' -> CombinedProxy p c' out) <$> f c
proxyOutCol _ p                       = pure p

proxyExpectedOutput :: Traversal' Proxy [[((StateFeatures, AgentActionIndices), Value)]]
proxyExpectedOutput f (CombinedProxy p c out) = CombinedProxy p c <$> f out
proxyExpectedOutput _ p                       = pure p


instance Show Proxy where
  show (Scalar x _)            = "Scalar: " ++ show x
  show (Table t _ _)           = "Table: " ++ take 300 txt ++ (if length txt > 300 then "..." else "")
    where txt = show t
  show (Grenade _ _ t _ _ _)   = "Grenade " ++ show t
  show (CombinedProxy p col _) = "CombinedProxy of " ++ show p ++ " at column " ++ show col

prettyProxyType :: Proxy -> String
prettyProxyType Scalar{}              = "Scalar"
prettyProxyType Table{}               = "Tabular"
prettyProxyType Grenade{}             = "Grenade"
prettyProxyType (CombinedProxy p _ _) = "Combined Proxy built on " <> prettyProxyType p


instance NFData Proxy where
  rnf (Table x def acts) = rnf x `seq` rnf def `seq` rnf acts
  rnf (Grenade t w tp cfg nrActs agents) = rnf t `seq` rnf w `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs `seq` rnf agents
  rnf (Scalar x nrAs) = rnf x `seq` rnf nrAs
  rnf (CombinedProxy p nr xs) = rnf p `seq` rnf nr `seq` rnf xs


isNeuralNetwork :: Proxy -> Bool
isNeuralNetwork Grenade{}             = True
isNeuralNetwork (CombinedProxy p _ _) = isNeuralNetwork p
isNeuralNetwork _                     = False

isGrenade :: Proxy -> Bool
isGrenade Grenade{}             = True
isGrenade (CombinedProxy p _ _) = isGrenade p
isGrenade _                     = False


isCombinedProxy :: Proxy -> Bool
isCombinedProxy CombinedProxy{} = True
isCombinedProxy _               = False

isTable :: Proxy -> Bool
isTable Table{}               = True
isTable (CombinedProxy p _ _) = isTable p
isTable _                     = False
