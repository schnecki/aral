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
  , addWorkerProxy
  , multiplyWorkerProxy
  , replaceTargetProxyFromTo
  , isNeuralNetwork
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
      { _proxyScalar :: !(V.Vector Float) -- ^ One value for each agent
      }
  | Table  -- ^ Representation using a table.
      { _proxyTable     :: !(M.Map (StateFeatures, ActionIndex) (V.Vector Float)) -- ^ Shared state and one action for each agent, returns one value for each agent.
      , _proxyDefault   :: !(V.Vector Float)
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
      , _proxyOutCol         :: Int                                                  -- ^ Output column/row of the data.
      , _proxyExpectedOutput :: [[((StateFeatures, [ActionIndex]), Value)]]          -- ^ List of batches of list of n-step results. Used to save the data for learning.
      }

-- | This function adds two proxies. The proxies must be of the same type and for Grenade of the same shape. It does not work with Tenforflow proxies!
addWorkerProxy :: Proxy -> Proxy -> Proxy
addWorkerProxy (Scalar x) (Scalar y)   = Scalar (x+y)
addWorkerProxy (Table x d acts) (Table y _ _) = Table (mergeTables x y) d acts
addWorkerProxy (Grenade (target1 :: Network layers1 shapes1) worker1 tp1 nnCfg1 nrActs1 nrAgents1) (Grenade (target2 :: Network layers2 shapes2) worker2 _ _ _ _) =
  case (cast worker2, cast target2) of
    (Just worker2', Just target2') ->
      Grenade
        (target1 |+ target2')  -- (disabled in multiplyWorkerProxy)
        (worker1 |+ worker2')
        tp1
        nnCfg1
        nrActs1
        nrAgents1
    _ -> error "cannot replace worker1 of different type"
addWorkerProxy (CombinedProxy px1 outCol1 expOut1) (CombinedProxy px2 _ _) = CombinedProxy (addWorkerProxy px1 px2) outCol1 expOut1
addWorkerProxy x1 x2 = error $ "Cannot add proxies of differnt types: " ++ show (x1, x2)

multiplyWorkerProxy :: Float -> Proxy -> Proxy
multiplyWorkerProxy n (Scalar x) = Scalar (V.map (n *) x)
multiplyWorkerProxy n (Table x d acts) = Table (M.map (V.map (* n)) x) d acts
multiplyWorkerProxy n (Grenade (target1 :: Network layers1 shapes1) worker1 tp1 nnCfg1 nrActs1 nrAgents1) =
  Grenade
    (toRational n |* target1) -- (disabled in addWorkerProxy)
    (toRational n |* worker1)
    tp1
    nnCfg1
    nrActs1
    nrAgents1
multiplyWorkerProxy n (CombinedProxy px1 outCol1 expOut1) = CombinedProxy (multiplyWorkerProxy n px1) outCol1 expOut1


replaceTargetProxyFromTo :: Proxy -> Proxy -> Proxy
replaceTargetProxyFromTo (Scalar x) (Scalar _)   = Scalar x
replaceTargetProxyFromTo (Table x d acts) (Table _ _ _) = Table x d acts
replaceTargetProxyFromTo (Grenade (target1 :: Network layers1 shapes1) worker1 _ _ _ _) (Grenade (target2 :: Network layers2 shapes2) worker2 tp2 nnCfg2 nrActs2 nrAgents2) =
  case cast worker1 of
    Nothing -> error "cannot replace target1 of different type"
    Just worker1' ->
      Grenade
        target2 -- (0.5 |* (target1' |+ target2))
        worker1' -- target1' -- (0.5 |* (worker1' |+ worker2))
        tp2
        nnCfg2
        nrActs2
        nrAgents2
replaceTargetProxyFromTo (CombinedProxy px1 _ _) (CombinedProxy px2 outCol2 expOut2) = CombinedProxy (replaceTargetProxyFromTo px1 px2) outCol2 expOut2
replaceTargetProxyFromTo x1 x2 = error $ "Cannot replace proxies of differnt types: " ++ show (x1, x2)


mergeTables :: (Ord k, Num n) => M.Map k n -> M.Map k n -> M.Map k n
mergeTables = M.mergeWithKey (\_ v1 v2 -> Just (v1 + v2)) (M.map (*2)) (M.map (*2))

proxyScalar :: Traversal' Proxy (V.Vector Float)
proxyScalar f (Scalar x) = Scalar <$> f x
proxyScalar _ p          = pure p

proxyTable :: Traversal' Proxy (M.Map (StateFeatures, ActionIndex) (V.Vector Float))
proxyTable f (Table m d acts) = (\m' -> Table m' d acts) <$> f m
proxyTable  _ p               = pure p

proxyDefault :: Traversal' Proxy (V.Vector Float)
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

proxyExpectedOutput :: Traversal' Proxy [[((StateFeatures, [ActionIndex]), Value)]]
proxyExpectedOutput f (CombinedProxy p c out) = (CombinedProxy p c) <$> f out
proxyExpectedOutput _ p                       = pure p


instance Show Proxy where
  show (Scalar x)                  = "Scalar: " ++ show x
  show Table{}                     = "Table"
  show (Grenade _ _ t _ _ agents)  = "Grenade " ++ show t
  show (CombinedProxy p col _)     = "CombinedProxy of " ++ show p ++ " at column " ++ show col

prettyProxyType :: Proxy -> String
prettyProxyType Scalar{}              = "Scalar"
prettyProxyType Table{}               = "Tabular"
prettyProxyType Grenade{}             = "Grenade"
prettyProxyType (CombinedProxy p _ _) = "Combined Proxy built on " <> prettyProxyType p


instance NFData Proxy where
  rnf (Table x def acts) = rnf x `seq` rnf def `seq` rnf acts
  rnf (Grenade t w tp cfg nrActs agents) = rnf t `seq` rnf w `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs `seq` rnf agents
  rnf (Scalar x) = rnf x
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
