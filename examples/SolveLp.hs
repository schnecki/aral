{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections       #-}
module SolveLp
    ( runBorlLp
    , BorlLp (..)
    , Policy
    , LpResult (..)
    ) where

import           ML.BORL

import           Control.Arrow
import           Control.Monad
import           Data.Function             (on)
import           Data.List
import qualified Data.Map.Strict           as M
import qualified Data.Text                 as T
import           Numeric.LinearProgramming

type EpisodeEnd = Bool
type Probability = Double
type NextState st = st


class (Ord st, Enum st, Bounded st, Show st) => BorlLp st where
  lpActions :: [Action st]
  lpActionFilter :: st -> [Bool]

data LpResult st = LpResult
  { givenPolicy     :: [((st, T.Text), [((st, T.Text), Probability)])]
  , inferredRewards :: [((st, T.Text), Double)]
  , gain            :: Double
  , bias            :: [((st, T.Text), Double)]
  , wValues         :: [((st, T.Text), Double)]
  }

instance (Show st) => Show (LpResult st) where
  show (LpResult pol rew g b w) =
    "Provided Policy:\n--------------------\n" <> unlines (map showPol pol) <>
    "\nInferred Rewards:\n--------------------\n" <> unlines (map show rew) <>
    "\nGain: " <> show g <>
    "\nBias values:\n--------------------\n" <> unlines (map show b) <>
    "\nW values:\n--------------------\n" <> unlines (map show w)
    where showPol (sa, sa') = show sa <> ": " <> show sa'

type Policy st = State st -> Action st -> [((NextState st, Action st), Probability)]

runBorlLp :: forall st . (BorlLp st) => Policy st -> IO (LpResult st)
runBorlLp policy = do
  let transitionProbs =
        map (\xs@(x:_) -> (fst x, map snd xs)) $
        groupBy ((==) `on` second actionName . fst) $
        sortBy (compare `on` second actionName . fst) $ filter ((/= 0) . snd . snd) $ concat [map ((s, a), ) (policy s a) | s <- states, a <- lpActions]
  -- print transitionProbs
  let stateActions = map fst transitionProbs
  let stateActionIndices = M.fromList $ zip (map (second actionName . fst) transitionProbs) [2 ..] -- start with nr 2, as 1 is g
  let obj = Maximize (1 : replicate (2 * length transitionProbs) 0)
  rewards <- concat <$> mapM makeReward states
  let rewards' = map (first (second actionName)) rewards
  let constr = map (makeConstraints stateActionIndices rewards) transitionProbs
  let constraints = Sparse (concat constr)
  let bounds = map Free [1 .. (2 * length transitionProbs + 1)]
  let sol = simplex obj constraints bounds
  let transProbs = map (second (map (first (second actionName))) . first (second actionName)) transitionProbs
  case sol of
    Optimal (g, vals) ->
      return $ LpResult transProbs rewards' g (zipWith mkResult stateActions (tail vals)) (zipWith mkResult stateActions (drop (length stateActions) (tail vals)))
  where
    states = [minBound .. maxBound] :: [st]
    mkResult (s, a) v = ((s, actionName a), v)


makeConstraints :: (BorlLp s) => M.Map (s, T.Text) Int -> [((State s, Action s), Double)] -> ((State s, Action s), [((State s, Action s), Probability)]) -> [Bound [(Double, Int)]]
makeConstraints stateActionIndices rewards (stAct, xs) =
  [ ([1# 1, 1# stateIndex stAct] ++ map (\(stateAction, prob) -> -prob # stateIndex stateAction) xs) :==: rewardValue stAct
  , ([1 # stateIndex stAct, 1 # wIndex stAct] ++ map (\(stateAction, prob) -> -prob # wIndex stateAction) xs) :==: 0
  ]
  where
    stateIndex state = M.findWithDefault (error $ "state " ++ show state ++ " not found in stateIndices") (second actionName state) stateActionIndices
    stateCount = M.size stateActionIndices
    wIndex state = stateCount + stateIndex state
    rewardValue k =
      case find ((== second actionName k) . second actionName . fst) rewards of
        Nothing -> error $ "Could not find reward for: " <> show (second actionName k)
        Just (_, r) -> r

makeReward :: (BorlLp s) => s -> IO [((State s, Action s), Double)]
makeReward s = do
  xss <- mapM ((\a -> replicateM nr (a s)) . actionFunction) acts
  return $ zipWith (\a xs -> ((s, a),sum (map (fromReward . fst3) xs) / fromIntegral (length xs)) ) acts xss
  where
    fst3 (x,_,_) = x
    snd3 (_,x,_) = x
    acts = map snd $ filter fst $ zip (lpActionFilter s) lpActions
    nr = 10000
    mkList :: [(Reward s, NextState s, Bool)] -> Double
    mkList [] = error "policy defines a transition which could not be inferred using the given actions! Aborting."
    mkList xs@((_,s',_):_) = sum (map (fromReward . fst3) xs) / fromIntegral (length xs)
    fromReward (Reward x) = x


