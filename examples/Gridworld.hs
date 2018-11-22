{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE OverloadedStrings #-}
module Main where

import           ML.BORL

import           Helper

import           Control.Arrow (first, second)
import           Control.Lens  (set, (^.))
import           Control.Monad (foldM, unless, when)

import           Grenade
import           System.IO
import           System.Random

maxX,maxY :: Int
maxX = 4                        -- [0..maxX]
maxY = 4                        -- [0..maxY]


type NN = Network  '[ FullyConnected 3 6, Relu, FullyConnected 6 4, Relu, FullyConnected 4 1, Tanh] '[ 'D1 3, 'D1 6, 'D1 6, 'D1 4, 'D1 4, 'D1 1, 'D1 1]

nnConfig :: NNConfig St
nnConfig = NNConfig netInp [] 128 (LearningParameters 0.01 0.5 0.0001) ([minBound .. maxBound] :: [St]) (scalingByMaxReward 10) 10000

netInp :: St -> [Double]
netInp st = [scaleNegPosOne (0, fromIntegral maxX) $ fromIntegral $ fst (getCurrentIdx st), scaleNegPosOne (0, fromIntegral maxY) $ fromIntegral $ snd (getCurrentIdx st)]


main :: IO ()
main = do

  -- let rl = mkBORLUnichainTabular initState actions actFilter params decay
  net <- randomNetworkInitWith HeEtAl :: IO NN
  let rl = mkBORLUnichain initState actions actFilter params decay net nnConfig
  askUser True usage cmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = zipWith3 (\n (s,a) na -> (s, (n, Action a na))) [0..] [("i",moveUp),("j",moveDown), ("k",moveLeft), ("l", moveRight) ] (tail names)
        usage = [("i","Move up") , ("j","Move left") , ("k","Move down") , ("l","Move right")]

names = ["random", "up   ", "down ", "left ", "right"]

params :: Parameters
params = Parameters 0.2 1.0 1.0 1.0 1.0 0.1 0.5 0.2


initState :: St
initState = fromIdx (0,0)

decay :: Period -> Parameters -> Parameters
decay t p@(Parameters alp bet del eps exp rand zeta xi)
  | t `mod` 200 == 0 = Parameters (f $ slow * alp) (f $ slow * bet) (f $ slow * del) (max 0.1 $ slow * eps) (f $ slow * exp) rand zeta xi -- (1 - slow * (1-frc)) mRho
  | otherwise = p

  where slow = 0.95
        faster = 1.0/0.995
        f = max 0.001


-- State
newtype St = St [[Integer]] deriving (Eq)

instance Ord St where
  x <= y = fst (getCurrentIdx x) < fst (getCurrentIdx y) || (fst (getCurrentIdx x) == fst (getCurrentIdx y) && snd (getCurrentIdx x) < snd (getCurrentIdx y))

instance Show St where
  show xs = show (getCurrentIdx xs)

instance Enum St where
  fromEnum st = let (x,y) = getCurrentIdx st
                in x * (maxX + 1) + y
  toEnum x = fromIdx (x `div` (maxX+1), x `mod` (maxX+1))

instance Bounded St where
  minBound = fromIdx (0,0)
  maxBound = fromIdx (maxX, maxY)


-- Actions
actions :: [Action St]
actions = zipWith Action
  (map goalState [moveRand, moveUp, moveDown, moveLeft, moveRight])
  names

actFilter :: St -> [Bool]
actFilter st | st == fromIdx (0,2) = True : repeat False
actFilter _  = False : repeat True


moveRand :: St -> IO (Reward, St)
moveRand = moveUp


-- goalState :: Action St -> Action St
goalState f st = do
  x <- randomRIO (0, maxX :: Int)
  y <- randomRIO (0, maxY :: Int)
  case getCurrentIdx st of
    -- (0, 1) -> return [(1, (10, fromIdx (x,y)))]
    (0, 2) -> return (10, fromIdx (x,y))
    -- (0, 3) -> return [(1, (5, fromIdx (x,y)))]
    _      -> stepRew <$> f st
  where stepRew = first (+ 1)


moveUp :: St -> IO (Reward,St)
moveUp st
    | m == 0 = return (-1, st)
    | otherwise = return (0, fromIdx (m-1,n))
  where (m,n) = getCurrentIdx st

moveDown :: St -> IO (Reward,St)
moveDown st
    | m == maxX = return (-1, st)
    | otherwise = return (0, fromIdx (m+1,n))
  where (m,n) = getCurrentIdx st

moveLeft :: St -> IO (Reward,St)
moveLeft st
    | n == 0 = return (-1, st)
    | otherwise = return (0, fromIdx (m,n-1))
  where (m,n) = getCurrentIdx st

moveRight :: St -> IO (Reward,St)
moveRight st
    | n == maxY = return (-1, st)
    | otherwise = return (0, fromIdx (m,n+1))
  where (m,n) = getCurrentIdx st


-- Conversion from/to index for state

fromIdx :: (Int, Int) -> St
fromIdx (m,n) = St $ zipWith (\nr xs -> zipWith (\nr' ys -> if m == nr && n == nr' then 1 else 0) [0..] xs) [0..] base
  where base = replicate 5 [0,0,0,0,0]


getCurrentIdx :: St -> (Int,Int)
getCurrentIdx (St st) = second (fst . head . filter ((==1) . snd)) $
  head $ filter ((1 `elem`) . map snd . snd) $
  zip [0..] $ map (zip [0..]) st


