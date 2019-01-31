module ML.BORL.Step
    ( step
    , stepExecute
    , nextAction
    ) where

import           ML.BORL.Action
import           ML.BORL.Fork
import           ML.BORL.Parameters
import           ML.BORL.Properties
import           ML.BORL.Proxy       as P
import           ML.BORL.Type

import           Control.Applicative ((<|>))
import           Control.DeepSeq     (NFData, force)
import           Control.Lens
import           Control.Monad
import           Data.Function       (on)
import           Data.List           (groupBy, sortBy)
import qualified Data.Map.Strict     as M
import           Data.Maybe          (isJust)
import           System.Random

expSmthPsi :: Double
expSmthPsi = 0.03

approxAvg :: Double
approxAvg = fromIntegral (100 :: Int)


step :: (NFData s, Ord s) => BORL s -> IO (BORL s)
step borl = fmap setRefState (nextAction borl) >>= stepExecute

setRefState :: (BORL s, Bool, ActionIndexed s) -> (BORL s, Bool, ActionIndexed s)
setRefState inp@(borl, b, as@(aNr,_))
  | True = inp
  | isJust (borl ^. sRef) = inp
  | otherwise = (sRef .~ Just (borl^.s, aNr) $ borl, b, as)

stepExecute :: (NFData s, Ord s) => (BORL s, Bool, ActionIndexed s) -> IO (BORL s)
stepExecute (borl, randomAction, act@(aNr, Action action _)) = do
  let state = borl ^. s
  (reward, stateNext) <- action state
  let mv = borl ^. v
      mw = borl ^. w
      mr0 = borl ^. r0
      mr1 = borl ^. r1
  let bta = borl ^. parameters . beta
      alp = borl ^. parameters . alpha
      dlt = borl ^. parameters . delta
      (ga0, ga1) = borl ^. gammas
      period = borl ^. t
  vValState <- vValue borl state act
  vValStateNext <- vStateValue borl stateNext
  rhoVal <- rhoValue borl state act
  wValState <- wValue borl state act
  wValStateNext <- wStateValue borl state
  r0ValState <- rValue borl RSmall state act
  r1ValState <- rValue borl RBig state act
  let label = (state, aNr)
  rhoState <- if isUnichain borl
    then return (reward + vValStateNext - vValState)
    -- reward                                              -- Alternative to above (estimating it from actual reward)
    else do
    rhoStateValNext <- rhoStateValue borl stateNext
    return $ (approxAvg * rhoStateValNext + reward) / (approxAvg + 1) -- approximation
  let rhoVal' = (1 - alp) * rhoVal + alp * rhoState
  rhoNew <-
    case borl ^. rho of
      Left _  -> return $ Left rhoVal'
      Right m -> Right . force <$> P.insert period label rhoVal' m
  let psiRho = rhoVal' - rhoVal -- should converge to 0
  let vValState' = (1 - bta) * vValState + bta * (reward - rhoVal' + vValStateNext)
      psiV = -reward + rhoVal' + vValState' - vValStateNext -- should converge to 0
  let wValState' | borl ^. sRef == Just (state, aNr) = 0
                 | otherwise = (1 - dlt) * wValState + dlt * (-vValState' + wValStateNext)
      psiW = vValState' + wValState' - wValStateNext -- should converge to 0
  forkMw' <- doFork $ P.insert period label wValState' mw
  let (psiValRho, psiValV, psiValW) = borl ^. psis -- Psis (exponentially smoothed)
      psiValRho' = (1 - expSmthPsi) * psiValRho + expSmthPsi * abs psiRho
      psiValV' = (1 - expSmthPsi) * psiValV + expSmthPsi * abs psiV
      psiValW' = (1 - expSmthPsi) * psiValW + expSmthPsi * abs psiW
  -- enforce values
  let vValStateNew
        | borl ^. sRef == Just (state, aNr) = 0
        | otherwise =
          vValState' -
          if randomAction || psiValV' > borl ^. parameters . zeta
            then 0
            else borl ^. parameters . xi * psiW
      -- wValStateNew = wValState' - if randomAction then 0 else (1-borl ^. parameters.xi) * psiW
  forkMv' <- doFork $ P.insert period label vValStateNew mv
  rSmall <- rStateValue borl RSmall stateNext
  let r0ValState' = (1 - bta) * r0ValState + bta * (reward + ga0 * rSmall)
  forkMr0' <- doFork $ P.insert period label r0ValState' mr0
  rBig <- rStateValue borl RBig stateNext
  let r1ValState' = (1 - bta) * r1ValState + bta * (reward + ga1 * rBig)
  forkMr1' <- doFork $ P.insert period label r1ValState' mr1
  let params' = (borl ^. decayFunction) (period + 1) (borl ^. parameters)
  mv' <- collectForkResult forkMv'
  mw' <- collectForkResult forkMw'
  mr0' <- collectForkResult forkMr0'
  mr1' <- collectForkResult forkMr1'
  let borl'
        | randomAction && borl ^. parameters . exploration <= borl ^. parameters . learnRandomAbove = borl -- multichain ?
        | otherwise = set v mv' $ set w mw' $ set rho rhoNew $ set r0 mr0' $ set r1 mr1' borl
  -- update values
  return $ force $ -- needed to ensure constant memory consumption
    set psis (psiValRho', psiValV', psiValW') $
    set visits (M.alter (\mV -> ((+ 1) <$> mV) <|> Just 1) state (borl ^. visits)) $
    set s stateNext $
    set t (period + 1) $
    set parameters params' borl'


-- | This function chooses the next action from the current state s and all possible actions.
nextAction :: (Ord s) => BORL s -> IO (BORL s, Bool, ActionIndexed s)
nextAction borl = do
  let explore = borl ^. parameters . exploration
  let state = borl ^. s
  let as = actionsIndexed borl state
  when (null as) (error "Empty action list")
  rand <- randomRIO (0, 1)
  if rand < explore
    then do
      r <- randomRIO (0, length as - 1)
      return (borl, True,  as !! r)
    else do
    bestRho <- if isUnichain borl
      then return as
      else do
      rhoVals <- mapM (rhoValue borl state) as
      return $ map snd $ head $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (zip rhoVals as)
    bestV <- do
      vVals <- mapM (vValue borl state) bestRho
      return $ map snd $ head $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (zip vVals bestRho)
    bestE <- do
      eVals <- mapM (eValue borl state) bestV
      return  $ map snd $ sortBy (epsCompare compare `on` fst) (zip eVals bestV)
    -- bestR <- sortBy (epsCompare compare `on` rValue borl RBig state) bestV
    -- return (False, head bestR)
    if length bestE > 1
        then do
          r <- randomRIO (0, length bestE - 1)
          return (borl, False, bestE !! r)
        else return (borl, False, head bestE)
  where
    eps = borl ^. parameters . epsilon
    epsCompare f x y
      | abs (x - y) <= eps = f 0 0
      | otherwise = y `f` x

-- actions :: BORL s -> s -> [Action s]
-- actions borl state = map snd (actionsIndexed borl state)

actionsIndexed :: BORL s -> s -> [ActionIndexed s]
actionsIndexed borl state = map snd $ filter fst $ zip ((borl ^. actionFilter) state) (borl ^. actionList)


reduce :: [Double] -> Double
reduce = maximum
-- reduce xs = sum xs / fromIntegral (length xs)
{-# INLINE reduce #-}


-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
rhoValue :: (Ord s) => BORL s -> s -> ActionIndexed s -> IO Double
rhoValue = rhoValueWith Worker

rhoValueWith :: (Ord s) => LookupType -> BORL s -> s -> ActionIndexed s -> IO Double
rhoValueWith lkTp borl state (a,_) =
  case borl ^. rho of
    Left r  -> return r
    Right m -> P.lookupProxy (borl ^. t) lkTp (state,a) m

rhoStateValue :: (Ord s) => BORL s -> s -> IO Double
rhoStateValue borl state = case borl ^. rho of
  Left r  -> return r
  Right _ -> reduce <$> mapM (rhoValueWith Target borl state) (actionsIndexed borl state)

vValue :: (Ord s) => BORL s -> s -> ActionIndexed s -> IO Double
vValue = vValueWith Worker

vValueWith :: (Ord s) => LookupType -> BORL s -> s -> ActionIndexed s -> IO Double
vValueWith lkTp borl state (a,_) = P.lookupProxy (borl ^. t) lkTp (state, a) mv
  where
    mv = borl ^. v

vStateValue :: (Ord s) => BORL s -> s -> IO Double
vStateValue borl state = reduce <$> mapM (vValueWith Target borl state) (actionsIndexed borl state)


wValue :: (Ord s) => BORL s -> s -> ActionIndexed s -> IO Double
wValue = wValueWith Worker

wValueWith :: (Ord s) => LookupType -> BORL s -> s -> ActionIndexed s -> IO Double
wValueWith lkTp borl state (a,_) = P.lookupProxy (borl ^. t) lkTp (state, a) mw
  where
    mw = borl ^. w

wStateValue :: (Ord s) => BORL s -> s -> IO Double
wStateValue borl state = reduce <$> mapM (wValueWith Target borl state) (actionsIndexed borl state)


-- | Used to select a discount factor.
data RSize
  = RSmall
  | RBig


-- | Calculates the expected discounted value with the provided gamma (small/big).
rValue :: (Ord s) => BORL s -> RSize -> s -> ActionIndexed s -> IO Double
rValue = rValueWith Worker

-- | Calculates the expected discounted value with the provided gamma (small/big).
rValueWith :: (Ord s) => LookupType -> BORL s -> RSize -> s -> ActionIndexed s -> IO Double
rValueWith lkTp borl size state (a, _) = P.lookupProxy (borl ^. t) lkTp (state, a) mr
  where
    mr =
      case size of
        RSmall -> borl ^. r0
        RBig   -> borl ^. r1

rStateValue :: (Ord s) => BORL s -> RSize -> s -> IO Double
rStateValue borl size state = reduce <$> mapM (rValueWith Target borl size state) (actionsIndexed borl state)

-- | Calculates the difference between the expected discounted values.
eValue :: (Ord s) => BORL s -> s -> ActionIndexed s -> IO Double
eValue borl state act = do
  big <- rValueWith Target borl RBig state act
  small <- rValueWith Target borl RSmall state act
  return $ big - small

--  | Calculates the difference between the expected discounted values.
-- eStateValue :: (Ord s) => BORL s -> s -> Double
-- eStateValue borl state = reduce (map (rValueWith Target borl RBig state) as) - reduce (map (rValueWith Target borl RSmall state) as)
--   where as = actionsIndexed borl state

