{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections       #-}
module SolveLp
    ( runBorlLp
    , runBorlLpInferWithRewardRepet
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
runBorlLp = runBorlLpInferWithRewardRepet 80000

runBorlLpInferWithRewardRepet :: forall st . (BorlLp st) => Int -> Policy st -> IO (LpResult st)
runBorlLpInferWithRewardRepet repetitionsReward policy = do
  let transitionProbs =
        map (\xs@(x:_) -> (fst x, map snd xs)) $
        groupBy ((==) `on` second actionName . fst) $
        sortBy (compare `on` second actionName . fst) $
        filter ((>= 0.001) . snd . snd) $ concat [map ((s, a), ) (policy s a) | s <- states, a <- map snd $ filter fst $ zip (lpActionFilter s) lpActions]
  -- print transitionProbs
  let stateActions = map fst transitionProbs
  let stateActionIndices = M.fromList $ zip (map (second actionName . fst) transitionProbs) [2 ..] -- start with nr 2, as 1 is g
  let obj = Maximize (1 : replicate (3 * length transitionProbs) 0)
  rewards <- concat <$> mapM (makeReward repetitionsReward) states
  let rewards' = map (first (second actionName)) rewards
  let constr = map (makeConstraints stateActionIndices rewards) transitionProbs
  let constraints = Sparse (concat constr)
  let bounds = map Free [1 .. (3 * length transitionProbs + 1)]
  let sol = simplex obj constraints bounds
  let transProbs = map (second (map (first (second actionName))) . first (second actionName)) transitionProbs
  case sol of
    Optimal (g, vals) ->
      return $ LpResult transProbs rewards' g (zipWith mkResult stateActions (tail vals)) (zipWith mkResult stateActions (drop (length stateActions) (tail vals)))
  where
    states = [minBound .. maxBound] :: [st]
    mkResult (s, a) v = ((s, actionName a), v)


makeConstraints :: (BorlLp s) => M.Map (s, T.Text) Int -> [((State s, Action s), Double)] -> ((State s, Action s), [((State s, Action s), Probability)]) -> [Bound [(Double, Int)]]
makeConstraints stateActionIndices rewards (stAct, xs)
  | stAct `elem` map fst xs =
    [ ([1 # 1] ++ map (\(stateAction, prob) -> (ite (stAct == stateAction) (1 - prob) (-prob)) # stateIndex stateAction) xs) :==: rewardValue stAct
    , ([1 # stateIndex stAct] ++ map (\(stateAction, prob) -> ite (stAct == stateAction) (1-prob) (-prob) # wIndex stateAction) xs) :==: 0
    , ([1 # wIndex stAct] ++ map (\(stateAction, prob) -> ite (stAct == stateAction) (1-prob) (-prob) # w2Index stateAction) xs) :==: 0
    ]
  | otherwise =
    [ ([1 # 1, 1 # stateIndex stAct] ++ map (\(stateAction, prob) -> -prob # stateIndex stateAction) xs) :==: rewardValue stAct
    , ([1 # stateIndex stAct, 1 # wIndex stAct] ++ map (\(stateAction, prob) -> -prob # wIndex stateAction) xs) :==: 0
    , ([1 # wIndex stAct, 1 # w2Index stAct] ++ map (\(stateAction, prob) -> -prob # w2Index stateAction) xs) :==: 0
    ]
  where
    ite True x _  = x
    ite False _ y = y
    stateIndex state =
      M.findWithDefault
        (error $ "state " ++ show state ++ " not found in stateIndices. Check your policy for unreachable state-action pairs!")
        (second actionName state)
        stateActionIndices
    stateCount = M.size stateActionIndices
    wIndex state = stateCount + stateIndex state
    w2Index state = 2*stateCount + stateIndex state
    rewardValue k =
      case find ((== second actionName k) . second actionName . fst) rewards of
        Nothing -> error $ "Could not find reward for: " <> show (second actionName k)
        Just (_, r) -> r

makeReward :: (BorlLp s) => Int -> s -> IO [((State s, Action s), Double)]
makeReward repetitionsReward s = do
  xss <- mapM ((\a -> replicateM repetitionsReward (a s)) . actionFunction) acts
  return $ zipWith (\a xs -> ((s, a),sum (map (fromReward . fst3) xs) / fromIntegral (length xs)) ) acts xss
  where
    fst3 (x,_,_) = x
    acts = map snd $ filter fst $ zip (lpActionFilter s) lpActions
    fromReward (Reward x) = x
    fromReward _ = error "Non materialised reward in makeReward. You must specify a scalar reward!"
    -- snd3 (_,x,_) = x
    -- mkList :: [(Reward s, NextState s, Bool)] -> Double
    -- mkList [] = error "policy defines a transition which could not be inferred using the given actions! Aborting."
    -- mkList xs@((_,s',_):_) = sum (map (fromReward . fst3) xs) / fromIntegral (length xs)


