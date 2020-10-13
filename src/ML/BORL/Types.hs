{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE InstanceSigs        #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
{-# OPTIONS_GHC -fno-cse #-}
module ML.BORL.Types where

import qualified Data.Vector.Storable as V

type FilteredActionIndices = [V.Vector ActionIndex] -- ^ List of filtered action indices for each agent.
type ActionIndex = Int
type IsRandomAction = Bool
type IsOptimalValue = Bool
type ActionFilter s = s -> [V.Vector Bool] -- ^ List of action flags for each agent. Each Vector must be of the length
                                           -- of the actions.

-- | Agent type. There is only one main agent, but there could be multiple workers (configured via NNConfig).
data AgentType = MainAgent | WorkerAgent Int
  deriving (Show, Read, Eq, Ord)

isMainAgent :: AgentType -> Bool
isMainAgent MainAgent = True
isMainAgent _         = False

isWorkerAgent :: AgentType -> Bool
isWorkerAgent MainAgent = False
isWorkerAgent _         = True


instance Enum AgentType where
  fromEnum MainAgent        = 0
  fromEnum (WorkerAgent nr) = nr
  toEnum nr = (MainAgent : map WorkerAgent [1..]) !! nr

type NStep = Int
type Batchsize = Int
type EpisodeEnd = Bool
type InitialStateFun s = AgentType -> IO s         -- ^ Initial state

liftInitSt :: s -> InitialStateFun s
liftInitSt = const . return

type Period = Int
type State s = s
type StateNext s = s
type PsisOld = (Float, Float, Float)
type PsisNew = PsisOld
type RewardValue = Float


type FeatureExtractor s = s -> StateFeatures
type GammaLow = Float
type GammaHigh = Float
type GammaMiddle = Float
type Gamma = Float

-- data ValueType
--   = ValueSA
--   | ValuesFiltered
--   | ValuesUnfiltered

newtype Value = AgentValue [Float]
newtype Values = AgentValues [V.Vector Float]

mapValues :: (V.Vector Float -> V.Vector Float) -> Values -> Values
mapValues f (AgentValues vals) = AgentValues (fmap f vals)

fromValues :: Values -> [V.Vector Float]
fromValues (AgentValues xs) = xs

singleAgentValue :: Float -> Value
singleAgentValue = AgentValue . return

singleAgentValues :: V.Vector Float -> Values
singleAgentValues = AgentValues . return

multiAgentValue :: V.Vector Float -> Value
#ifdef DEBUG
multiAgentValue xs | V.null xs = error "empty input list in singleAgentValues in ML.BORL.Types"
#endif
multiAgentValue xs = AgentValue (V.toList xs)

multiAgentValues :: [V.Vector Float] -> Values
#ifdef DEBUG
multiAgentValues [] = error "empty input list in singleAgentValues in ML.BORL.Types"
multiAgentValues xs
  | any (/= l) ls = error $ "length in input list in singleAgentValues in ML.BORL.Types do not match: " ++ show (l : ls)
  where
    (l:ls) = V.length <$> xs
#endif
multiAgentValues xs = AgentValues xs


type StateFeatures = V.Vector Float
type StateNextFeatures = StateFeatures
type NetInputWoAction = StateFeatures
type NetInput = StateFeatures
type NetInputWithAction = StateFeatures
type NetOutput = V.Vector Float
type StateActionValuesFiltered = Values

type MSE = Float               -- ^ Mean squared error
type MaxValue n = n
type MinValue n = n


replace :: Int -> a -> [a] -> [a]
replace idx val ls = take idx ls ++ val : drop (idx+1) ls
