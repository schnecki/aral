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
type FilteredActions as = VB.Vector (VB.Vector (Action as)) -- ^ List of agents with possible actions for each agent.
type NumberOfActions = Int

-- ActionIndex
type ActionIndex = Int
type AgentActionIndices = VB.Vector ActionIndex               -- ^ One action index per agent
type FilteredActionIndices = VB.Vector (V.Vector ActionIndex) -- ^ Allowed actions for each agent.


type ActionChoice = VB.Vector (IsRandomAction, ActionIndex)                       -- ^ One action per agent


instance Serialize ActionChoice where
  put xs = put (VB.toList xs)
  get = VB.fromList <$> get

type NextActions = (ActionChoice, [WorkerActionChoice])
type RandomNormValue = Float
type UseRand = Bool
type WorkerActionChoice = ActionChoice


type IsRandomAction = Bool
type IsOptimalValue = Bool
type ActionFilter s = s -> [V.Vector Bool] -- ^ List of action flags for each agent. Each Vector must be of
                                                     -- the length of the actions.

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

newtype Value = AgentValue (V.Vector Float) -- ^ One value for every agent
  deriving (Show, Generic, NFData)

instance Serialize Value where
  put (AgentValue xs) = put (V.toList xs)
  get = AgentValue . V.fromList <$> get

newtype Values = AgentValues (VB.Vector (V.Vector Float)) -- ^ A vector of values for every agent
  deriving (Show)


instance Num Value where
  (AgentValue xs) + (AgentValue ys) = AgentValue (V.zipWith (+) xs ys)
  (AgentValue xs) - (AgentValue ys) = AgentValue (V.zipWith (-) xs ys)
  (AgentValue xs) * (AgentValue ys) = AgentValue (V.zipWith (*) xs ys)
  abs (AgentValue xs) = AgentValue (V.map abs xs)
  signum (AgentValue xs) = AgentValue (V.map signum xs)
  -- fromInteger nr = AgentValue (repeat $ fromIntegral nr)
  fromInteger _ = error "fromInteger is not implemented for Value!"
    -- AgentValue (replicate 1 $ fromIntegral nr)

toValue :: Int -> Float -> Value
toValue nr v = AgentValue $ V.replicate nr v

(.*) :: Float -> Value -> Value
x .* (AgentValue xs) = AgentValue (V.map (x*) xs)
infixl 7 .*

(*.) :: Value -> Float -> Value
(AgentValue xs) *. x = AgentValue (V.map (*x) xs)
infixl 7 *.

(.+) :: Float -> Value -> Value
x .+ (AgentValue xs) = AgentValue (V.map (x+) xs)
infixl 6 .+

(+.) :: Value -> Float -> Value
(AgentValue xs) +. x = AgentValue (V.map (+x) xs)
infixl 6 +.

(.-) :: Float -> Value -> Value
x .- (AgentValue xs) = AgentValue (V.map (x-) xs)
infixl 6 .-

(-.) :: Value -> Float -> Value
(AgentValue xs) -. x = AgentValue (V.map (subtract x) xs)
infixl 6 -.


fromValue :: Value -> [Float]
fromValue (AgentValue xs) = V.toList xs


mapValue :: (Float -> Float) -> Value -> Value
mapValue f (AgentValue vals) = AgentValue (V.map f vals)

reduceValue :: (V.Vector Float -> Float) -> Value -> Float
reduceValue f (AgentValue vals) = f vals

selectIndex :: Int -> Value -> Float
selectIndex idx (AgentValue vals) =
#ifdef DEBUG
  (if V.length vals <= idx then error ("selectIndex out of Bounds " ++ show (idx, vals)) else id)
#endif
  vals V.! idx


zipWithValue :: (Float -> Float -> Float) -> Value -> Value -> Value
zipWithValue f (AgentValue xs) (AgentValue ys) = AgentValue $ V.zipWith f xs ys

mapValues :: (V.Vector Float -> V.Vector Float) -> Values -> Values
mapValues f (AgentValues vals) = AgentValues (fmap f vals)

selectIndices :: AgentActionIndices -> Values -> Value
selectIndices idxs (AgentValues vals) = AgentValue (V.convert $ VB.zipWith (V.!) vals idxs)

reduceValues :: (V.Vector Float -> Float) -> Values -> Value
reduceValues f (AgentValues vals) = AgentValue (V.convert $ VB.map f vals)

zipWithValues :: (a -> V.Vector Float -> b) -> [a] -> Values -> VB.Vector b
zipWithValues f as (AgentValues vals) = VB.zipWith f (VB.fromList as) vals

fromValues :: Values -> VB.Vector (V.Vector Float)
fromValues (AgentValues xs) = xs

-- | Use with care! This function does not work properly on filtered Values.
toActionValue :: Values -> [Value]
toActionValue (AgentValues xs) = map (AgentValue . V.fromList) (transpose $ map V.toList (VB.toList xs))

multiAgentValue :: V.Vector Float -> Value
#ifdef DEBUG
multiAgentValue xs | V.null xs = error "empty input list in singleAgentValues in ML.BORL.Types"
#endif
multiAgentValue xs = AgentValue xs

multiAgentValues :: VB.Vector (V.Vector Float) -> Values
#ifdef DEBUG
multiAgentValues xs
  | VB.null xs = error "empty input list in singleAgentValues in ML.BORL.Types"
  | any (/= l) ls = error $ "length in input list in singleAgentValues in ML.BORL.Types do not match: " ++ show (l : ls)
  where
    l = V.length (VB.head xs)
    ls = map V.length (VB.toList $ VB.tail xs)
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
