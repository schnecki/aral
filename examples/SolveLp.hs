{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TupleSections         #-}
module SolveLp
    ( runBorlLp
    , runBorlLpInferWithRewardRepet
    , runBorlLpInferWithRewardRepetWMax
    , mkEstimatedVFile
    , mkEstimatedEFile
    , mkEstimatedVGammaFile
    , mkStateFile
    , EpisodeEnd
    , BorlLp (..)
    , Policy
    , LpResult (..)
    ) where

import           ML.ARAL

import           Control.Arrow
import           Control.Monad
import           Data.Function             (on)
import           Data.List
import qualified Data.Map.Strict           as M
import           Data.Maybe                (fromMaybe)
import qualified Data.Text                 as T
import qualified Data.Vector               as VB
import qualified Data.Vector.Storable      as V
import           Numeric.LinearProgramming
import           System.IO                 (hFlush, stdout)

import           Debug.Trace

type Probability = Double
type NextState st = st

-- | BorlLp is designed for a single agent only!
class (Ord st, Enum st, Bounded st, Show st) => BorlLp st as where
  lpActions        :: st -> [Action as] -- ^ Only one agent is allowed! First parameter is a dummy parameter.
  lpActionFunction :: ActionFunction st as
  lpActionFilter   :: as -> st -> V.Vector Bool

data LpResult st as = LpResult
  { givenPolicy     :: [((st, T.Text), [((st, T.Text), Probability)])]
  , inferredRewards :: [((st, T.Text), Double)]
  , gain            :: Double
  , bias            :: [((st, T.Text), Double)]
  , wValues         :: [[((st, T.Text), Double)]]
  }

instance (Eq st, Show st) => Show (LpResult st as) where
  show (LpResult pol rew g b w) =
    "Provided Policy:\n--------------------\n" <> unlines (map showPol pol) <>
    "\nInferred Rewards:\n--------------------\n" <> unlines (map show rew) <>
    "\nGain: " <> show g <>
    "\nBias values:\n--------------------\n" <> unlines (map show b) <>
    concatMap (\nr -> "\nW" ++ show nr ++ " values:\n--------------------\n" <> unlines (map show (w!!(nr-1)))) [1..min 5 (length w)] <>
    "\nW Sum values:\n--------------------\n" <> unlines (map (show . flip mkSum (concat w)) (map fst b))
    where mkSum k xs = (k, sum $ map snd $ filter ((==k).fst) xs)
    -- "\nW2 values:\n--------------------\n" <> unlines (map show w2) <>
    -- "\nW3 values:\n--------------------\n" <> unlines (map show w3)

showPol :: (Show a, Show a1) => (a, a1) -> String
showPol (sa, sa') = show sa <> ": " <> show sa'

type Policy st as = State st -> Action as -> [((NextState st, Action as), Probability)]

runBorlLp :: forall st as . (BorlLp st as, Show as, Eq as) => Policy st as -> Maybe (st, ActionIndex) -> IO (LpResult st as)
runBorlLp = runBorlLpInferWithRewardRepet 80000

tshow :: (Show a) => a -> T.Text
tshow = T.pack . show

mkEstimatedVFile :: (BorlLp st as) => LpResult st as -> IO ()
mkEstimatedVFile = mkStateFile 0.5 False True

mkEstimatedVGammaFile :: (BorlLp st as) => LpResult st as -> IO ()
mkEstimatedVGammaFile = mkStateFile 0.5 True True

mkEstimatedEFile :: (BorlLp st as) => LpResult st as -> IO ()
mkEstimatedEFile = mkStateFile 0.5 False False

mkStateFile :: (BorlLp st as) => Double -> Bool -> Bool -> LpResult st as -> IO ()
mkStateFile minGamma withGain withBias (LpResult _ _ gain bias ws) = do
  let file
        | withGain && withBias = "lp_v_gamma"
        | withGain = "lp_gain_and_e"
        | withBias = "lp_v"
        | otherwise = "lp_e"
  let deltaFile = file <> "_delta"
  let stateActions = map fst bias
      stateActionsTxtWith f = mkListStr (\(st, a) -> f $ T.unpack $ T.replace " " "_" $ tshow st <> "," <> a) stateActions
      stateActionsTxt = stateActionsTxtWith id
  writeFile "lp_state_nrs" (show $ length stateActions)
  writeFile file ("gamma\t" <> stateActionsTxt <> "\n")
  writeFile deltaFile ("gamma\t" <> stateActionsTxt <> "\t" <> stateActionsTxtWith ("d/d^2"<>) <> "\n")
  let maxGamma
        | withGain = 0.99
        | otherwise = 1.0
      step = 0.002
  forM_ [minGamma,minGamma + step .. maxGamma] $ \gam -> do
    let vals ga =
          flip map stateActions $ \sa ->
            let gainValue
                  | withGain = gain / (1 - ga)
                  | otherwise = 0
                biasValue
                  | withBias = snd $ head $ filter ((== sa) . fst) bias
                  | otherwise = 0
                wsValues = map (snd . head . filter ((== sa) . fst)) (init ws) -- last one is offset
                eFun n w = ((1 - ga) / ga) ^ n * w
                val = gainValue + biasValue + sum (zipWith eFun [(1 :: Integer) ..] wsValues)
             in val
    appendFile file (show gam <> "\t" <> mkListStr show (vals gam) <> "\n")
    let valsDelta = zipWith (-) (vals gam) (vals $ gam+step)
        valsDelta2 = zipWith (-) valsDelta (tail valsDelta)
    when (gam + step <= 1) $ appendFile deltaFile (show gam <> "\t" <> mkListStr show valsDelta <> "\t" <> mkListStr show valsDelta2 <> "\n")
  where
    mkListStr :: (a -> String) -> [a] -> String
    mkListStr f = intercalate "\t" . map f


runBorlLpInferWithRewardRepet :: forall st as . (BorlLp st as, Show as, Eq as) => Int -> Policy st as -> Maybe (st, ActionIndex) -> IO (LpResult st as)
runBorlLpInferWithRewardRepet = runBorlLpInferWithRewardRepetWMax 3

runBorlLpInferWithRewardRepetWMax :: forall st as . (BorlLp st as, Show as, Eq as) => Int -> Int -> Policy st as -> Maybe (st, ActionIndex) -> IO (LpResult st as)
runBorlLpInferWithRewardRepetWMax wMax repetitionsReward policy mRefStAct = do
  let allActions = lpActions (head states) :: [Action as]
  let mkPol :: st -> as -> [((NextState st, Action as), Probability)]
      mkPol s a =
        case policy s a of
          [] -> [((s, a), 0)]
          xs -> filter ((>= 0.001) . snd) xs
  let transitionProbs =
        map (\xs@(x:_) -> (fst x, map snd xs)) $
        groupBy ((==) `on` second tshow . fst) $
        sortBy (compare `on` second show . fst) $
        concat [map ((s, a), ) (mkPol s a) | s <- states, a <- map snd $ filter fst $ zip (V.toList $ lpActionFilter (head allActions) s) allActions]
      transProbSums = map (\(x, ps) -> (x, sum $ map snd ps)) transitionProbs
  mapM_ (\(a, p) -> when (abs (1 - p) > 0.001) $ error $ "transition probabilities do not sum up to 1 for state-action: " ++ show a) transProbSums
  let stateActions = map fst transitionProbs
  let stateActionIndices = M.fromList $ zip (map (second tshow . fst) transitionProbs) [2 ..] -- start with nr 2, as 1 is g
  let obj = Maximize (1 : replicate ((wMax + 1) * length transitionProbs) 0)
  putStr ("Inferring rewards using " <> show repetitionsReward <> " replications ...") >> hFlush stdout
  rewards <- concat <$> mapM (makeReward repetitionsReward) states
  putStrLn "\t[Done]"
  let rewards' = map (\(x, y, _) -> (second tshow x, y)) rewards
  let mRefStAct' = second (allActions !!) <$> mRefStAct :: Maybe (st, Action as)
  let constr = map (makeConstraints wMax mRefStAct' stateActionIndices rewards) transitionProbs
  let constraints = Sparse (concat constr)
  let bounds = map Free [1 .. ((wMax + 1) * length transitionProbs + 1)]
  let sol = simplex obj constraints bounds
  let transProbs = map (second (map (first (second tshow))) . first (second tshow)) transitionProbs
  let mkSol (g, vals) =
        LpResult
          transProbs
          rewards'
          g
          (zipWith mkResult stateActions (tail vals))
          (map (\nr -> (zipWith mkResult stateActions (drop (nr * length stateActions) (tail vals)))) [1 .. wMax])
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
                  "\n\nProvided Policy:\n--------------------\n" <>
                  unlines (map showPol transProbs) <> "\n\nSolver returned: Unbounded! Introducing bound of " <> show initBounds <> " and retrying..."
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
    mkResult (s, a) v = ((s, tshow a), v)
    isOptimal r =
      case r of
        Optimal {} -> True
        _          -> False

makeConstraints ::
     (BorlLp s as, Show as, Eq as)
  => Int
  -> Maybe (State s, Action as)
  -> M.Map (s, T.Text) Int
  -> [((State s, Action as), Probability, EpisodeEnd)]
  -> ((State s, Action as), [((State s, Action as), Probability)])
  -> [Bound [(Double, Int)]]
makeConstraints wMax mRefStAct stateActionIndices rewards (stAct, xs)
  | stAct `elem` map fst xs -- double occurance of variable is not allowed!
   =
    [ ([1 # 1] ++ map (\(stateAction, prob) -> ite (stAct == stateAction) (1 - prob) (-prob) # stateIndex stateAction) xs') :==: realToFrac (rewardValue stAct)
    , ([1 # stateIndex stAct] ++ map (\(stateAction, prob) -> ite (stAct == stateAction) (1 - prob) (-prob) # wIndex stateAction) xs') :==: 0
    ] ++
    map (\nr -> ([1 # wNrIndex (nr - 1) stAct] ++ map (\(stateAction, prob) -> ite (stAct == stateAction) (1 - prob) (-prob) # wNrIndex nr stateAction) xs') :==: 0) [2 .. wMax] ++
    stActCtr
  | otherwise =
    [ ([1 # 1, 1 # stateIndex stAct] ++ map (\(stateAction, prob) -> -prob # stateIndex stateAction) xs') :==: realToFrac (rewardValue stAct)
    , ([1 # stateIndex stAct, 1 # wIndex stAct] ++ map (\(stateAction, prob) -> -prob # wIndex stateAction) xs') :==: 0
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
        (second tshow state)
        stateActionIndices
    stateCount = M.size stateActionIndices
    wNrIndex nr state = nr * stateCount + stateIndex state
    wIndex state = stateCount + stateIndex state
    w2Index state = 2 * stateCount + stateIndex state
    w3Index state = 3 * stateCount + stateIndex state
    episodeEnds = map (fst3 &&& thd3) rewards
    rewardValue k =
      case find ((== second tshow k) . second tshow . fst3) rewards of
        Nothing        -> error $ "Could not find reward for: " <> show (second tshow k)
        Just (_, r, _) -> r

fst3 (x,_,_) = x
thd3 (_,_,x) = x


makeReward :: (BorlLp s as) => Int -> s -> IO [((State s, Action as), Double, EpisodeEnd)]
makeReward repetitionsReward s = do
  let actionFun tp s act = lpActionFunction tp s [act]
  xss <- mapM ((\a -> replicateM repetitionsReward (actionFun MainAgent s a))) acts
  return $ zipWith (\a xs -> ((s, a), round' $ sum (map (fromReward . fst3) xs) / fromIntegral (length xs), getEpsEnd (map thd3 xs))) acts xss
  where
    round' x = (/100) . fromIntegral $ round (x * 100)
    allActs = lpActions s
    acts = map snd $ filter fst $ zip (V.toList $ lpActionFilter (head allActs) s) allActs
    getEpsEnd xs
      | length trues >= length false = True
      | otherwise = False
      where
        trues = filter id xs
        false = filter not xs
    fromReward (Reward x) = x
    fromReward _          = error "Non materialised reward in makeReward. You must specify a scalar reward!"
