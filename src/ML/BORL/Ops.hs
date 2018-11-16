

module ML.BORL.Ops
    ( step
    , stepExecute
    , chooseNextAction
    ) where

import           ML.BORL.Parameters
import           ML.BORL.Type

import           Control.Applicative ((<|>))
import           Control.Lens
import           Control.Monad
import           Data.Function       (on)
import           Data.List           (foldl', groupBy, sortBy)
import qualified Data.Map.Strict     as M
import           Data.Maybe          (fromJust)
import           System.Random
import           Text.Printf


step :: (Show s, Ord s) => BORL s -> IO (BORL s)
step borl = chooseNextAction borl >>= stepExecute borl

stepExecute :: (Show s, Ord s) => BORL s -> (Bool, [(Probability, (Reward, s))]) -> IO (BORL s)
stepExecute borl (randomAction, possNxtStates) = do
  rand <- randomRIO (0, 1)
  when (null possNxtStates) (print "possNxtStates")
  let (_, (reward, stateNext)) = snd $ foldl' (\(ps, c) c'@(p, _) -> if ps <= rand && ps + p > rand then (ps + p, c') else (ps + p, c)) (0, head possNxtStates) possNxtStates
  let mv = borl ^. v
      mw = borl ^. w
      mPsiRho = borl ^. psiStates._1
      mPsiV = borl ^. psiStates._2
      mPsiW = borl ^. psiStates._3
      mr0 = borl ^. r0
      mr1 = borl ^. r1
      state = borl ^. s
  let bta = borl ^. parameters . beta
      alp = borl ^. parameters . alpha
      dlt = borl ^. parameters . delta
      (ga0, ga1) = borl ^. gammas
      period = borl ^. t
  let vValState = M.findWithDefault 0 state mv
      vValStateNext = M.findWithDefault 0 stateNext mv
      rhoVal = case borl ^. rho of
        Left v     -> v
        Right mrho -> M.findWithDefault 0 state mrho
      wValState = M.findWithDefault 0 state mw
      wValStateNext = M.findWithDefault 0 stateNext mw
      r0ValState = M.findWithDefault 0 state mr0
      r1ValState = M.findWithDefault 0 state mr1


  let rhoState | isUnichain borl  = reward + vValStateNext - vValState
               | otherwise = (100 * rhoValue borl stateNext + reward) / 101 -- approximation
      -- rhoState = reward
      rhoVal' = (1 - alp) * rhoVal + alp * rhoState

      -- multichain use exponential smoothing with g + Pg = 0
      rhoNew = case borl ^. rho of
        Left _  -> Left rhoVal'
        Right m -> Right (M.insert state rhoVal' m)
      psiRho = rhoVal' - rhoVal                                         -- should converge to 0

  let vValState'  = (1 - bta) * vValState + bta * (reward - rhoVal' + vValStateNext)
      psiV = - reward + rhoVal' + vValState' - vValStateNext             -- should converge to 0

  let wValState' = (1 - dlt) * wValState + dlt * (-vValState' + wValStateNext)
      psiW = vValState' + wValState' - wValStateNext                    -- should converge to 0

  let r0ValState' = (1 - bta) * r0ValState + bta * (reward + ga0 * M.findWithDefault 0 stateNext mr0)
      r1ValState' = (1 - bta) * r1ValState + bta * (reward + ga1 * M.findWithDefault 0 stateNext mr1)
      params' = (borl ^. decayFunction) (period + 1) (borl ^. parameters)

  let (psiValRho,psiValV,psiValW) = borl ^. psis
      psiValRho' = (1-0.03) * psiValRho + 0.03 * abs psiRho
      psiValV' = (1-0.03) * psiValV + 0.03 * abs psiV
      psiValW' = (1-0.03) * psiValW + 0.03 * abs psiW
      psiStateRho' = M.insert stateNext ((1-0.03) * M.findWithDefault 0 stateNext mPsiRho + 0.03 * psiRho) mPsiRho
      psiStateV'   = M.insert stateNext ((1-0.03) * M.findWithDefault 0 stateNext mPsiV + 0.03 * psiV) mPsiV
      psiStateW'   = M.insert stateNext ((1-0.03) * M.findWithDefault 0 stateNext mPsiW + 0.03 * psiW) mPsiW

  -- enforce values
  let vValStateNew = vValState' - if randomAction || psiValV' > borl ^. parameters.zeta then 0 else borl ^. parameters.xi * psiW
      -- wValStateNew = wValState' - if randomAction then 0 else (1-borl ^. parameters.xi) * psiW

  let borl' | randomAction && borl ^. parameters.exploration <= borl ^. parameters.learnRandomAbove = borl -- multichain ?
            | otherwise = set v (M.insert state vValStateNew mv) $ set w (M.insert state wValState' mw) $ set rho rhoNew $
                          set r0 (M.insert state r0ValState' mr0) $ set r1 (M.insert state r1ValState' mr1) borl

  -- update values
  return $
    set psiStates (psiStateRho', psiStateV',psiStateW') $
    set psis (psiValRho', psiValV', psiValW' ) $
    set visits (M.alter (\mV -> ((+1) <$> mV) <|> Just 1) state (borl ^. visits)) $
    set s stateNext $
    set t (period + 1) $
    set parameters params' borl'

-- | This function chooses the next action from the current state s and all possible actions.
chooseNextAction :: (Show s, Ord s) => BORL s -> IO (Bool, [(Probability, (Reward, s))])
chooseNextAction borl = do
  let expl = borl ^. parameters . exploration
  let state = borl ^. s
  let as = map snd $ filter fst $ zip ((borl ^. actionFilter) state) $ map snd (borl ^. actionList)
  when (null as) (error "Empty action list")
  possS <- mapM (\f -> f state) as
  rand <- randomRIO (0, 1)
  if rand < expl
    then do
      r <- randomRIO (0, length as - 1)
      return (True, possS !! r)
    else do
      when (any (\a -> sum (map fst a) >= 1.001 || sum (map fst a) <= 0.999) possS) (error $ "Transition probabilities must add up to 1 but are: " ++ show (map (map fst) possS))
      let bestRho | isUnichain borl = possS
                  | otherwise = head $ groupBy (epsCompare (==) `on` expectedRho borl) $ sortBy (epsCompare compare `on` expectedRho borl) possS
          bestV = head $ groupBy (epsCompare (==) `on` expectedV borl) $ sortBy (epsCompare compare `on` expectedV borl) bestRho
          bestE = sortBy (epsCompare compare `on` expectedE borl) bestV
      if length bestE > 1
        then do
          r <- randomRIO (0, length bestE - 1)
          return (False, bestE !! r)
        else return (False, head bestE)
  where
    eps = borl ^. parameters . epsilon
    epsCompare f x y
      | abs (x - y) <= eps = f 0 0
      | otherwise = y `f` x


-- | Expected average value of state s, that is y_{-1}(s).
expectedRho :: (Ord s) => BORL s -> [(Probability, (Reward, s))] -> Double
expectedRho borl as =
  case borl ^. rho of
    Left v  -> v
    Right m -> sum $ map (\(p,(r,s'))-> p * M.findWithDefault 0 s' m) as

rhoValue :: (Ord s) => BORL s -> s -> Double
rhoValue borl s' =   case borl ^. rho of
    Left v  -> v
    Right m -> M.findWithDefault 0 s' m


-- | Calculates the expected bias value for a given list of action outcomes.
expectedV :: (Ord s) => BORL s -> [(Probability, (Reward, s))] -> Double
expectedV borl as = sum $ map (\(p, (r, s')) -> p * (r + M.findWithDefault 0 s' mv)) as
  where
    mv = borl ^. v


-- | Used to select a discount factor.
data RSize
  = RSmall
  | RBig


-- | Calculates the expected discounted value with the provided gamma (small/big).
expectedR :: (Ord s) => BORL s -> RSize -> (Probability, (Reward, s)) -> Double
expectedR borl size (p, (r, s')) = p * (r + ga * M.findWithDefault 0 s' mr)
  where
    (mr, ga) =
      case size of
        RSmall -> (mr0, ga0)
        RBig   -> (mr1, ga1)
    mr0 = borl ^. r0
    mr1 = borl ^. r1
    (ga0, ga1) = borl ^. gammas


-- | Calculates the difference between the expected discounted values.
expectedE :: (Ord s) => BORL s -> [(Probability, (Reward, s))] -> Double
expectedE borl xs = sum (map (expectedR borl RBig) xs) - sum (map (expectedR borl RSmall) xs)
