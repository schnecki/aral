{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections       #-}
module SolveLp
    ( runBorlLp
    , runBorlLpInferWithRewardRepet
    , EpisodeEnd
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
import           Data.Maybe                (fromMaybe)
import qualified Data.Text                 as T
import           Numeric.LinearProgramming
import           System.IO                 (hFlush, stdout)

import           Debug.Trace

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
  , wValues         :: [[((st, T.Text), Double)]]
  -- , w2Values        :: [((st, T.Text), Double)]
  -- , w3Values        :: [((st, T.Text), Double)]
  }

instance (Show st) => Show (LpResult st) where
  show (LpResult pol rew g b w) =
    "Provided Policy:\n--------------------\n" <> unlines (map showPol pol) <>
    "\nInferred Rewards:\n--------------------\n" <> unlines (map show rew) <>
    "\nGain: " <> show g <>
    "\nBias values:\n--------------------\n" <> unlines (map show b) <>
    concatMap (\nr -> "\nW" ++ show nr ++ " values:\n--------------------\n" <> unlines (map show (w!!(nr-1)))) [1..wMax]
    -- "\nW values:\n--------------------\n" <> unlines (map show w) <>
    -- "\nW2 values:\n--------------------\n" <> unlines (map show w2) <>
    -- "\nW3 values:\n--------------------\n" <> unlines (map show w3)

showPol :: (Show a, Show a1) => (a, a1) -> String
showPol (sa, sa') = show sa <> ": " <> show sa'

type Policy st = State st -> Action st -> [((NextState st, Action st), Probability)]

runBorlLp :: forall st . (BorlLp st) => Policy st -> Maybe (st, ActionIndex) -> IO (LpResult st)
runBorlLp = runBorlLpInferWithRewardRepet 80000

wMax :: Int
wMax = 3


runBorlLpInferWithRewardRepet :: forall st . (BorlLp st) => Int -> Policy st -> Maybe (st, ActionIndex) -> IO (LpResult st)
runBorlLpInferWithRewardRepet repetitionsReward policy mRefStAct = do
  let mkPol s a =
        case policy s a of
          [] -> [((s, a), 0)]
          xs -> filter ((>= 0.001) . snd) xs
  let transitionProbs =
        map (\xs@(x:_) -> (fst x, map snd xs)) $
        groupBy ((==) `on` second actionName . fst) $
        sortBy (compare `on` second actionName . fst) $ concat [map ((s, a), ) (mkPol s a) | s <- states, a <- map snd $ filter fst $ zip (lpActionFilter s) lpActions]
      transProbSums = map (\(x, ps) -> (x, sum $ map snd ps)) transitionProbs
  mapM_ (\(a, p) -> when (abs (1 - p) > 0.001) $ error $ "transition probabilities do not sum up to 1 for state-action: " ++ show a) transProbSums
  let stateActions = map fst transitionProbs
  let stateActionIndices = M.fromList $ zip (map (second actionName . fst) transitionProbs) [2 ..] -- start with nr 2, as 1 is g
  let obj = Maximize (1 : replicate ((wMax + 1) * length transitionProbs) 0)
  putStr ("Inferring rewards using " <> show repetitionsReward <> " replications ...") >> hFlush stdout
  rewards <- concat <$> mapM (makeReward repetitionsReward) states
  putStrLn "\t[Done]"
  let rewards' = map (\(x, y, _) -> (second actionName x, y)) rewards
  let mRefStAct' = second (lpActions !!) <$> mRefStAct :: Maybe (st, Action st)
  let constr = map (makeConstraints mRefStAct' stateActionIndices rewards) transitionProbs
  let constraints = Sparse (concat constr)
  let bounds = map Free [1 .. ((wMax + 1) * length transitionProbs + 1)]
  let sol = simplex obj constraints bounds
  let transProbs = map (second (map (first (second actionName))) . first (second actionName)) transitionProbs
  let mkSol (g, vals) =
        LpResult
          transProbs
          rewards'
          g
          (zipWith mkResult stateActions (tail vals))
          (map (\nr -> (zipWith mkResult stateActions (drop (nr * length stateActions) (tail vals)))) [1..wMax])
  let parseSol mBound sol =
        case sol of
          Optimal xs -> return $ mkSol xs
          Feasible xs -> return $ mkSol xs
          Infeasible xs -> return $ mkSol xs
          _ ->
            case (sol, mBound) of
              (Unbounded, Nothing) -> do
                let initBounds = 1
                putStrLn $
                  "\n\nProvided Policy:\n--------------------\n" <> unlines (map showPol transProbs) <> "\n\nSolver returned: Unbounded! Introducing bound of " <> show initBounds <>
                  " and retrying..."
                let bounds = map (\x -> x :<=: 10) [1 .. ((wMax + 1) * length transitionProbs + 1)]
                parseSol (Just initBounds) (simplex obj constraints bounds)
              (res, Just bound) -> do
                let bound' = bound + 1
                putStrLn $ "Solver returned: " <> show res <> " for bound " <> show bound <> ". Increasing bound to " <> show bound' <> " and retrying..."
                let mkBounds b = map (\x -> x :<=: b) [1 .. ((wMax + 1) * length transitionProbs + 1)]
                let list = [bound,bound + 0.05 .. bound']
                sol <-
                  case simplex obj constraints (mkBounds bound') of
                    Optimal {} -> do
                      putStrLn $ "Found a solution with bound " <> show bound' <> ". Trying to find a smaller bound..."
                      let (b', res) = head $ filter (isOptimal . snd) $ zip list $ map (simplex obj constraints . mkBounds) list
                      putStrLn $ "Smallest bound I could find a solution for was " <> show b' <> "."
                      return res
                    x -> return x
                parseSol (Just bound') sol
              (Undefined, _) -> error $ unlines (map showPol transProbs) <> "\n\nSolver returned: Undefined"
              (NoFeasible, _) -> error $ unlines (map showPol transProbs) <> "\n\nSolver returned: NoFeasible"
  parseSol Nothing sol
  where
    states = [minBound .. maxBound] :: [st]
    mkResult (s, a) v = ((s, actionName a), v)
    isOptimal r =
      case r of
        Optimal {} -> True
        _          -> False

makeConstraints ::
     (BorlLp s)
  => Maybe (State s, Action s)
  -> M.Map (s, T.Text) Int
  -> [((State s, Action s), Double, EpisodeEnd)]
  -> ((State s, Action s), [((State s, Action s), Probability)])
  -> [Bound [(Double, Int)]]
makeConstraints mRefStAct stateActionIndices rewards (stAct, xs)
  | stAct `elem` map fst xs -- double occurance of variable is not allowed!
   =
    [ ([1 # 1] ++ map (\(stateAction, prob) -> ite (stAct == stateAction) (1 - prob) (-prob) # stateIndex stateAction) xs') :==: rewardValue stAct
    , ([1 # stateIndex stAct] ++ map (\(stateAction, prob) -> ite (stAct == stateAction) (1 - prob) (-prob) # wIndex stateAction) xs') :==: 0
    -- , ([1 # wIndex stAct] ++ map (\(stateAction, prob) -> ite (stAct == stateAction) (1 - prob) (-prob) # w2Index stateAction) xs') :==: 0
    -- , ([1 # w2Index stAct] ++ map (\(stateAction, prob) -> ite (stAct == stateAction) (1 - prob) (-prob) # w3Index stateAction) xs') :==: 0
    ] ++
    map (\nr -> ([1 # wNrIndex (nr - 1) stAct] ++ map (\(stateAction, prob) -> ite (stAct == stateAction) (1 - prob) (-prob) # wNrIndex nr stateAction) xs') :==: 0) [2 .. wMax] ++
    stActCtr
  | otherwise =
    [ ([1 # 1, 1 # stateIndex stAct] ++ map (\(stateAction, prob) -> -prob # stateIndex stateAction) xs') :==: rewardValue stAct
    , ([1 # stateIndex stAct, 1 # wIndex stAct] ++ map (\(stateAction, prob) -> -prob # wIndex stateAction) xs') :==: 0
    -- , ([1 # wIndex stAct, 1 # w2Index stAct] ++ map (\(stateAction, prob) -> -prob # w2Index stateAction) xs') :==: 0
    ] ++
    map (\nr -> ([1 # wNrIndex (nr - 1) stAct, 1 # wNrIndex nr stAct] ++ map (\(stateAction, prob) -> -prob # wNrIndex nr stateAction) xs') :==: 0) [2 .. wMax] ++ stActCtr
  where
    stActCtr = [[1 # w2Index stAct] :==: 0 | Just stAct == mRefStAct]
    xs' =
      case find ((== stAct) . fst) episodeEnds of
        Just (_, True) -> xs
        _              -> xs
    ite True x _  = x
    ite False _ y = y
    stateIndex state =
      M.findWithDefault
        (error $ "state " ++ show state ++ " not found in stateIndices. Check your policy for unreachable state-action pairs!")
        (second actionName state)
        stateActionIndices
    stateCount = M.size stateActionIndices
    wNrIndex nr state = nr * stateCount + stateIndex state
    wIndex state = stateCount + stateIndex state
    w2Index state = 2 * stateCount + stateIndex state
    w3Index state = 3 * stateCount + stateIndex state
    episodeEnds = map (fst3 &&& thd3) rewards
    rewardValue k =
      case find ((== second actionName k) . second actionName . fst3) rewards of
        Nothing -> error $ "Could not find reward for: " <> show (second actionName k)
        Just (_, r, _) -> r

fst3 (x,_,_) = x
thd3 (_,_,x) = x


makeReward :: (BorlLp s) => Int -> s -> IO [((State s, Action s), Double, EpisodeEnd)]
makeReward repetitionsReward s = do
  xss <- mapM ((\a -> replicateM repetitionsReward (a s)) . actionFunction) acts
  return $ zipWith (\a xs -> ((s, a), round' $ sum (map (fromReward . fst3) xs) / fromIntegral (length xs), getEpsEnd (map thd3 xs))) acts xss
  where
    round' x = (/100) . fromIntegral $ round (x * 100)
    acts = map snd $ filter fst $ zip (lpActionFilter s) lpActions
    getEpsEnd xs
      | length trues >= length false = True
      | otherwise = False
      where
        trues = filter id xs
        false = filter not xs
    fromReward (Reward x) = x
    fromReward _ = error "Non materialised reward in makeReward. You must specify a scalar reward!"
