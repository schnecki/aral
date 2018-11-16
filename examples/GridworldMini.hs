module Main where

import           ML.BORL

import           Helper

import           Control.Arrow (first, second)
import           Control.Lens  (set, (^.))
import           Control.Monad (foldM, unless, when)
import           System.IO
import           System.Random

-- | Define the grid size.
maxX,maxY :: Int
maxX = 2                        -- [0..maxX]
maxY = 2                        -- [0..maxY]

-- | Defines the goal states
goalState :: Action St -> Action St
goalState f st = do
  x <- randomRIO (0, maxX :: Int)
  y <- randomRIO (0, maxY :: Int)
  case getCurrentIdx st of
    (0, 2) -> return [(1, (10, fromIdx (maxX, 0)))]
    _      -> stepRew <$> f st
  where
    stepRew = map (second (first (+ 1)))

main :: IO ()
main = do
  let rl = mkBORLUnichain initState actions (const $ repeat True) params decay
  askUser True usage cmds rl -- maybe increase learning by setting estimate of rho
  where
    cmds = map (second goalState) [("i", moveUp), ("j", moveLeft), ("k", moveDown), ("l", moveRight)]
    usage = [("i", "Move up"), ("j", "Move left"), ("k", "Move down"), ("l", "Move right")]


initState :: St
initState = fromIdx (0,0)


-- | BORL Parameters.
params :: Parameters
params = Parameters 0.001 0.001 0.001 1.0 0.1 0.0 0.2 0.2


-- | Decay function of parameters.
decay :: Period -> Parameters -> Parameters
decay t p@(Parameters alp bet del eps exp rand zeta xi)
  | t `mod` 1000 == 0 = Parameters alp (f $ slower * bet) (f $ slower * del) (max 0.1 $ slower * eps) (max 0.01 $ slower * exp) rand zeta xi -- (1 - slower * (1-frc)) mRho
  | otherwise = p
  where
    slower = 0.995
    slow = 0.95
    faster = 1.0 / 0.995
    f = max 0.001


-- State
newtype St = St [[Integer]] deriving (Eq)

instance Ord St where
  x <= y = fst (getCurrentIdx x) < fst (getCurrentIdx y) || (fst (getCurrentIdx x) == fst (getCurrentIdx y) && snd (getCurrentIdx x) < snd (getCurrentIdx y))

instance Show St where
  show xs = show (getCurrentIdx xs)


-- Actions
actions :: [St -> IO [(Probability, (Reward, St))]]
actions = map goalState [moveUp, moveDown, moveLeft, moveRight]


moveUp :: St -> IO [(Probability, (Reward, St))]
moveUp st
    | m == 0 = return [(1, (-1, st))]
    | otherwise = return [(1, (0, fromIdx (m-1,n)))]
  where (m,n) = getCurrentIdx st

moveDown :: St -> IO [(Probability, (Reward, St))]
moveDown st
    | m == maxX = return [(1,(-1, st))]
    | otherwise = return [(1, (0, fromIdx (m+1,n)))]
  where (m,n) = getCurrentIdx st

moveLeft :: St -> IO [(Probability, (Reward, St))]
moveLeft st
    | n == 0 = return [(1, (-1, st))]
    | otherwise = return [(1, (0, fromIdx (m,n-1)))]
  where (m,n) = getCurrentIdx st

moveRight :: St -> IO [(Probability, (Reward, St))]
moveRight st
    | n == maxY = return [(1, (-1, st))]
    | otherwise = return [(1, (0, fromIdx (m,n+1)))]
  where (m,n) = getCurrentIdx st


-- Conversion from/to index for state

fromIdx :: (Int, Int) -> St
fromIdx (m,n) = St $ zipWith (\nr xs -> zipWith (\nr' ys -> if m == nr && n == nr' then 1 else 0) [0..] xs) [0..] base
  where base = replicate 5 [0,0,0,0,0]


getCurrentIdx :: St -> (Int,Int)
getCurrentIdx (St st) = second (fst . head . filter ((==1) . snd)) $
  head $ filter ((1 `elem`) . map snd . snd) $
  zip [0..] $ map (zip [0..]) st


