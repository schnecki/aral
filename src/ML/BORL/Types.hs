{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE InstanceSigs        #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
{-# OPTIONS_GHC -fno-cse #-}
module ML.BORL.Types where

import           Control.DeepSeq
import           Data.List            (transpose)
import           Data.Serialize
import qualified Data.Vector          as VB
import qualified Data.Vector.Storable as V
import           GHC.Generics

-- Agent
type AgentNumber = Int


-- Action
type Action as = as
type FilteredActions as = [VB.Vector (Action as)] -- ^ List of agents with possible actions for each agent.

-- ActionIndex
type ActionIndex = Int
type FilteredActionIndices = [V.Vector ActionIndex] -- ^ Allowed actions for each agent.


type ActionChoice = [(IsRandomAction, ActionIndex)]                       -- ^ One action per agent
type NextActions = (ActionChoice, [WorkerActionChoice])
type RandomNormValue = Float
type UseRand = Bool
type WorkerActionChoice = ActionChoice


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

newtype Value = AgentValue [Float] -- ^ One value for every agent
  deriving (Show, Generic, NFData, Serialize)

newtype Values = AgentValues [V.Vector Float] -- ^ A vector of values for every agent


instance Num Value where
  (AgentValue xs) + (AgentValue ys) = AgentValue (zipWith (+) xs ys)
  (AgentValue xs) - (AgentValue ys) = AgentValue (zipWith (-) xs ys)
  (AgentValue xs) * (AgentValue ys) = AgentValue (zipWith (*) xs ys)
  abs (AgentValue xs) = AgentValue (map abs xs)
  signum (AgentValue xs) = AgentValue (map signum xs)
  fromInteger nr = AgentValue (repeat $ fromIntegral nr)

toValue :: Int -> Float -> Value
toValue nr v = AgentValue $ replicate nr v

(.*) :: Float -> Value -> Value
x .* (AgentValue xs) = AgentValue (map (x*) xs)
infixl 7 .*

(*.) :: Value -> Float -> Value
(AgentValue xs) *. x = AgentValue (map (*x) xs)
infixl 7 *.

(.+) :: Float -> Value -> Value
x .+ (AgentValue xs) = AgentValue (map (x+) xs)
infixl 6 .+

(+.) :: Value -> Float -> Value
(AgentValue xs) +. x = AgentValue (map (+x) xs)
infixl 6 +.

(.-) :: Float -> Value -> Value
x .- (AgentValue xs) = AgentValue (map (x-) xs)
infixl 6 .-

(-.) :: Value -> Float -> Value
(AgentValue xs) -. x = AgentValue (map (subtract x) xs)
infixl 6 -.


fromValue :: Value -> [Float]
fromValue (AgentValue xs) = xs


mapValue :: (Float -> Float) -> Value -> Value
mapValue f (AgentValue vals) = AgentValue (fmap f vals)

reduceValue :: ([Float] -> Float) -> Value -> Float
reduceValue f (AgentValue vals) = f vals

selectIndex :: Int -> Value -> Float
selectIndex idx (AgentValue vals) = vals !! idx


zipWithValue :: (Float -> Float -> Float) -> Value -> Value -> Value
zipWithValue f (AgentValue xs) (AgentValue ys) = AgentValue $ zipWith f xs ys

mapValues :: (V.Vector Float -> V.Vector Float) -> Values -> Values
mapValues f (AgentValues vals) = AgentValues (fmap f vals)

selectIndices :: [ActionIndex] -> Values -> Value
selectIndices idxs (AgentValues vals) = AgentValue (zipWith (V.!) vals idxs)

reduceValues :: (V.Vector Float -> Float) -> Values -> Value
reduceValues f (AgentValues vals) = AgentValue (map f vals)

zipWithValues :: (a -> V.Vector Float -> b) -> [a] -> Values -> [b]
zipWithValues f as (AgentValues vals) = zipWith f as vals

fromValues :: Values -> [V.Vector Float]
fromValues (AgentValues xs) = xs

-- | Use with care! This function does not work properly on filtered Values.
toActionValue :: Values -> [Value]
toActionValue (AgentValues xs) = map AgentValue (transpose $ map V.toList xs)

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
