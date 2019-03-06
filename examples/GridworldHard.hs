{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedLists            #-}
{-# LANGUAGE OverloadedStrings          #-}
module Main where

import           ML.BORL

import           Helper

import           Control.Arrow                        (first, second)
import           Control.DeepSeq                      (NFData)
import           Control.Lens
import           Control.Lens                         (set, (^.))
import           Control.Monad                        (foldM, unless, when)
import           Control.Monad.IO.Class               (liftIO)
import           Data.List                            (genericLength)
import           GHC.Generics
import           GHC.Int                              (Int32, Int64)
import           Grenade
import           System.IO
import           System.Random

import           Data.Random                          (StdRandom, runRVar, sample)
import           Data.Random.Distribution.Exponential (exponential)


import qualified TensorFlow.Build                     as TF (addNewOp, evalBuildT,
                                                             explicitName, opDef,
                                                             opDefWithName, opType,
                                                             runBuildT, summaries)
import qualified TensorFlow.Core                      as TF hiding (value)
-- import qualified TensorFlow.GenOps.Core                         as TF (square)
import qualified TensorFlow.GenOps.Core               as TF (abs, add, approximateEqual,
                                                             approximateEqual, assign,
                                                             cast, getSessionHandle,
                                                             getSessionTensor, identity',
                                                             lessEqual, matMul, mul,
                                                             readerSerializeState, relu,
                                                             relu', shape, square, sub,
                                                             tanh, tanh', truncatedNormal)
import qualified TensorFlow.Minimize                  as TF
-- import qualified TensorFlow.Ops                                 as TF (abs, add, assign,
--                                                                        cast, identity',
--                                                                        matMul, mul, relu,
--                                                                        sub,
--                                                                        truncatedNormal)
import qualified TensorFlow.Ops                       as TF (initializedVariable,
                                                             initializedVariable',
                                                             placeholder, placeholder',
                                                             reduceMean, reduceSum,
                                                             restore, save, scalar, vector,
                                                             zeroInitializedVariable,
                                                             zeroInitializedVariable')
import qualified TensorFlow.Tensor                    as TF (Ref (..), collectAllSummaries,
                                                             tensorNodeName,
                                                             tensorRefFromName,
                                                             tensorValueFromName)


maxX,maxY :: Int
maxX = 4                        -- [0..maxX]
maxY = 4                        -- [0..maxY]


type NN = Network  '[ FullyConnected 2 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 10, Relu, FullyConnected 10 5, Tanh] '[ 'D1 2, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 10, 'D1 10, 'D1 5, 'D1 5]

nnConfig :: NNConfig St
nnConfig = NNConfig
  { _toNetInp             = netInp
  , _replayMemoryMaxSize  = 10000
  , _trainBatchSize       = 32
  , _grenadeLearningParams       = LearningParameters 0.01 0.9 0.0001
  , _prettyPrintElems     = [minBound .. maxBound] :: [St]
  , _scaleParameters      = scalingByMaxAbsReward False 6
  , _updateTargetInterval = 10000
  , _trainMSEMax          = Just 0.00125
  }

netInp :: St -> [Double]
netInp st = [scaleNegPosOne (0, fromIntegral maxX) $ fromIntegral $ fst (getCurrentIdx st),
             scaleNegPosOne (0, fromIntegral maxY) $ fromIntegral $ snd (getCurrentIdx st)]


modelBuilder :: (TF.MonadBuild m) => m TensorflowModel
modelBuilder =
  buildModel $
  inputLayer1D (genericLength (netInp initState)) >> fullyConnected1D 10 TF.relu' >> fullyConnected1D 7 TF.relu' >> fullyConnected1D (genericLength actions) TF.tanh' >>
  trainingByAdam1DWith TF.AdamConfig {TF.adamLearningRate = 0.001, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}


main :: IO ()
main = do

  nn <- randomNetworkInitWith UniformInit :: IO NN
  -- rl <- mkBORLUnichainGrenade initState actions actFilter params decay nn nnConfig
  -- rl <- mkBORLUnichainTensorflow initState actions actFilter params decay modelBuilder nnConfig Nothing
  let rl = mkBORLUnichainTabular initState actions actFilter params decay
  askUser True usage cmds rl   -- maybe increase learning by setting estimate of rho

  where cmds = zipWith3 (\n (s,a) na -> (s, (n, Action a na))) [0..] [("i",moveUp),("j",moveDown), ("k",moveLeft), ("l", moveRight) ] (tail names)
        usage = [("i","Move up") , ("j","Move left") , ("k","Move down") , ("l","Move right")]

names = ["random", "up   ", "down ", "left ", "right"]

-- | BORL Parameters.
params :: Parameters
params = Parameters
  { _alpha            = 0.5
  , _beta             = 0.05
  , _delta            = 0.04
  , _gamma            = 0.30
  , _epsilon          = 0.075
  , _exploration      = 0.8
  , _learnRandomAbove = 0.0
  , _zeta             = 1.0
  , _xi               = 0.2
  }

-- | Decay function of parameters.
decay :: Decay
decay t (psiRhoOld, psiVOld, psiWOld) (psiRhoNew, psiVNew, psiWNew) p@(Parameters alp bet del ga eps exp rand zeta xi)
  | t `mod` 200 == 0 =
    Parameters
      (max 0.03 $ slow * alp)
      (max 0.015 $ slow * bet)
      (max 0.015 $ slow * del)
      (max 0.01 $ slow * ga)
      (max 0.05 $ slow * eps) -- (0.5*bet)
      (max 0.01 $ slow * exp)
      rand
      zeta -- zeta
      -- (max 0.075 $ slower * xi)
      (0.5*bet)
  | otherwise = p
  where
    slower = 0.995
    slow = 0.98
    faster = 1.0 / 0.99
    f = max 0.01


initState :: St
initState = fromIdx (2,2)


-- State
newtype St = St [[Integer]] deriving (Eq,NFData,Generic)

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
actions = zipWith Action [goalState moveRand, moveUp, moveDown, moveLeft, moveRight] names

actFilter :: St -> [Bool]
actFilter st | fromEnum st `elem` goalStates  = repeat True
  where goalStates = map fromEnum [fromIdx (0,0), fromIdx (0,4),fromIdx (4,0),fromIdx (4,4)]
actFilter _  = False : repeat True


moveRand :: St -> IO (Reward, St)
moveRand = moveUp


goalState :: (St -> IO (Reward, St)) -> St -> IO (Reward, St)
goalState f st = do
  x <- randomRIO (0, maxX :: Int)
  y <- randomRIO (0, maxY :: Int)
  xG <- randomRIO (0, 1 :: Int)
  r <- randomRIO (0, 8 :: Double)
  let stepRew = first (subtract r)
  case getCurrentIdx st of
    (0, 0) | True || xG == 0 -> return (10, fromIdx (x,y))
    (4, 4) | True || xG == 1 -> return (10, fromIdx (x,y))
    _      -> stepRew <$> f st


stepWidth :: IO Int
stepWidth = do
  x <- sample (exponential 1)
  return $ ceiling (x :: Float)


moveUp :: St -> IO (Reward,St)
moveUp st = stepWidth >>= \w -> return (fromIntegral $ 5 * min 0 (m-w), fromIdx (max 0 (m-w),n))
  where (m,n) = getCurrentIdx st

moveDown :: St -> IO (Reward,St)
moveDown st = stepWidth >>= \w -> return (fromIntegral $ 5 * min 0 (-(m+w-4)), fromIdx (min 4 (m+w),n))
  where (m,n) = getCurrentIdx st

moveLeft :: St -> IO (Reward,St)
moveLeft st = stepWidth >>= \w -> return (fromIntegral $ 5 * min 0 (n-w), fromIdx (m,max 0 (n-w)))
  where (m,n) = getCurrentIdx st

moveRight :: St -> IO (Reward,St)
moveRight st = stepWidth >>= \w -> return (fromIntegral $ 5 * min 0 (-(n+w-4)), fromIdx (m,min 4 (n+w)))
  where (m,n) = getCurrentIdx st


-- Conversion from/to index for state

fromIdx :: (Int, Int) -> St
fromIdx (m,n) = St $ zipWith (\nr xs -> zipWith (\nr' ys -> if m == nr && n == nr' then 1 else 0) [0..] xs) [0..] base
  where base = replicate 5 [0,0,0,0,0]


getCurrentIdx :: St -> (Int,Int)
getCurrentIdx (St st) = second (fst . head . filter ((==1) . snd)) $
  head $ filter ((1 `elem`) . map snd . snd) $
  zip [0..] $ map (zip [0..]) st


