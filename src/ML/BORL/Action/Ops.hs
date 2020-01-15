module ML.BORL.Action.Ops
    ( nextAction
    , epsCompareWith
    ) where


import           ML.BORL.Algorithm
import           ML.BORL.Calculation
import           ML.BORL.Exploration
import           ML.BORL.Parameters
import           ML.BORL.Properties
import           ML.BORL.Type
import           ML.BORL.Types


import           Control.Arrow          ((***))
import           Control.Lens
import           Control.Monad.IO.Class (liftIO)
import           Data.Function          (on)
import           Data.List              (groupBy, partition, sortBy)
import           System.Random

import           Debug.Trace


-- | This function chooses the next action from the current state s and all possible actions.
nextAction :: (MonadBorl' m) => BORL s -> m (BORL s, Bool, ActionIndexed s)
nextAction borl
  | null as = error "Empty action list"
  | length as == 1 = return (borl, False, head as)
  | otherwise =
    case borl ^. parameters . explorationStrategy of
      EpsilonGreedy -> chooseAction borl True (return . head)
      SoftmaxBoltzmann t0 -> chooseAction borl False (chooseBySoftmax (t0 * params' ^. exploration))
  where
    as = actionsIndexed borl state
    state = borl ^. s
    params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)


type UseRand = Bool
type ActionSelection s = [[(Double, ActionIndexed s)]] -> IO [(Double, ActionIndexed s)]
type RandomNormValue = Double

chooseBySoftmax :: TemperatureInitFactor -> ActionSelection s
chooseBySoftmax temp xs = do
  r <- liftIO $ randomRIO (0 :: Double, 1)
  return $ xs !! chooseByProbability r 0 0 probs
  where
    probs = softmax temp $ map (fst . head) xs

chooseByProbability :: RandomNormValue -> ActionIndex -> Double -> [Double] -> Int
chooseByProbability r idx acc [] = error $ "no more options in chooseByProbability in Action.Ops: " ++ show (r, idx, acc)
chooseByProbability r idx acc (v:vs)
  | acc + v >= r = idx
  | otherwise = chooseByProbability r (idx + 1) (acc + v) vs

chooseAction :: (MonadBorl' m) => BORL s -> UseRand -> ActionSelection s -> m (BORL s, Bool, ActionIndexed s)
chooseAction borl useRand selFromList = do
  rand <- liftIO $ randomRIO (0, 1)
  if useRand && rand < explore
    then do
      r <- liftIO $ randomRIO (0, length as - 1)
      return (borl, True, as !! r)
    else case borl ^. algorithm of
           AlgBORL {} -> do
             bestRho <-
               if isUnichain borl
                 then return as
                 else do
                   rhoVals <- mapM (rhoValue borl state . fst) as
                   map snd <$> liftIO (selFromList $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (zip rhoVals as))
             bestV <-
               do vVals <- mapM (vValue borl state . fst) bestRho
                  map snd <$> liftIO (selFromList $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (zip vVals bestRho))
             if length bestV == 1
               then return (borl, False, head bestV)
               else do
                 bestE <-
                   do eVals <- mapM (eValueAvgCleaned borl state . fst) bestV
                      let (increasing, decreasing) = partition ((0 <) . fst) (zip eVals bestV)
                          actionsToChooseFrom
                            | null decreasing = increasing
                            | otherwise = decreasing
                      map snd <$> liftIO (selFromList $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) actionsToChooseFrom)
                 -- other way of doing it:
                 -- ----------------------
                 -- do eVals <- mapM (eValue borl state . fst) bestV
                 --    rhoVal <- rhoValue borl state (fst $ head bestRho)
                 --    vVal <- vValue decideVPlusPsi borl state (fst $ head bestV) -- all a have the same V(s,a) value!
                 --    r0Values <- mapM (rValue borl RSmall state . fst) bestV
                 --    let rhoPlusV = rhoVal / (1-gamma0) + vVal
                 --        (posErr,negErr) = (map snd *** map snd) $ partition ((rhoPlusV<) . fst) (zip r0Values (zip eVals bestV))
                 --    return $ map snd $ head $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (if null posErr then negErr else posErr)
                 ----
                 if length bestE > 1 -- Uniform distributed as all actions are considered as having the same value!
                   then do
                     r <- liftIO $ randomRIO (0, length bestE - 1)
                     return (borl, False, bestE !! r)
                   else return (borl, False, headE bestE)
           AlgBORLVOnly {} -> singleValueNextAction EpsilonSensitive (vValue borl state . fst)
           AlgDQN _ cmp -> singleValueNextAction cmp (rValue borl RBig state . fst)
           AlgDQNAvgRewAdjusted {} -> do
             r1Values <- mapM (rValue borl RBig state . fst) as
             bestR1ValueActions <- liftIO $ selFromList $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (zip r1Values as)
             let bestR1 = map snd bestR1ValueActions
             r0Values <- mapM (rValue borl RSmall state . fst) bestR1
             let r1Value = fst $ headR1 bestR1ValueActions
                 group = groupBy (epsCompare (==) `on` fst) . sortBy (epsCompare compare `on` fst)
                 (posErr, negErr) = (group *** group) $ partition ((r1Value <) . fst) (zip r0Values bestR1)
             let bestR0 =
                   map snd $
                   head $
                   groupBy (epsCompare (==) `on` fst) $
                   sortBy
                     (epsCompare compare `on` fst)
                     (headR0 $
                      if null posErr
                        then negErr
                        else posErr)
             if length bestR1 == 1
               then return (borl, False, head bestR1)
               else if length bestR0 > 1
                      then do
                        r <- liftIO $ randomRIO (0, length bestR0 - 1)
                        return (borl, False, bestR0 !! r)
                      else return (borl, False, headDqnAvgRewFree bestR0)
               -- singleValueNextAction
  where
    headE []    = error "head: empty input data in nextAction on E Value"
    headE (x:_) = x
    headR0 []    = error "head: empty input data in nextAction on R0 Value"
    headR0 (x:_) = x
    headR1 []    = error "head: empty input data in nextAction on R1 Value"
    headR1 (x:_) = x
    headDqn []    = error "head: empty input data in nextAction on Dqn Value"
    headDqn (x:_) = x
    headDqnAvgRewFree [] = error "head: empty input data in nextAction on DqnAvgRewFree Value"
    headDqnAvgRewFree (x:_) = x
    params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
    eps = params' ^. epsilon
    explore = params' ^. exploration
    state = borl ^. s
    as = actionsIndexed borl state
    epsCompare = epsCompareWith eps
    singleValueNextAction cmp f = do
      rValues <- mapM f as
      let groupValues = case cmp of
            EpsilonSensitive -> groupBy (epsCompare (==) `on` fst) . sortBy (epsCompare compare `on` fst)
            Exact -> groupBy ((==) `on` fst) . sortBy (compare `on` fst)
      bestR <- liftIO $ selFromList $ groupValues (zip rValues as)
      if length bestR == 1
        then return (borl, False, snd $ headDqn bestR)
        else do
          r <- liftIO $ randomRIO (0, length bestR - 1)
          return (borl, False, snd $ bestR !! r)

epsCompareWith :: (Ord t, Num t) => t -> (t -> t -> p) -> t -> t -> p
epsCompareWith eps f x y
  | abs (x - y) <= eps = f 0 0
  | otherwise = y `f` x
