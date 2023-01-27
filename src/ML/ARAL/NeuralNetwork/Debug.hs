{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE TupleSections     #-}
module ML.ARAL.NeuralNetwork.Debug
    ( plotProxyFunction
    , calcModelCertainty
    ) where

import           Control.Lens
import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Function                               (on)
import           Data.List                                   (foldl', intercalate, sortBy)
import qualified Data.Map.Strict                             as M
import           Data.Maybe                                  (maybe)
import           Data.Ord
import qualified Data.Text                                   as T
import qualified Data.Vector                                 as VB
import           Data.Vector.Algorithms.Intro                as VB
import qualified Data.Vector.Storable                        as VS
import           EasyLogger
import           Statistics.Sample.WelfordOnlineMeanVariance
import           System.Environment                          (getProgName)
import           System.IO
import           System.Process

import           ML.ARAL.Algorithm
import           ML.ARAL.Calculation.Ops
import           ML.ARAL.Calculation.Type
import           ML.ARAL.NeuralNetwork.Hasktorch
import           ML.ARAL.NeuralNetwork.NNConfig
import           ML.ARAL.NeuralNetwork.Normalisation
import           ML.ARAL.NeuralNetwork.ReplayMemory
import           ML.ARAL.NeuralNetwork.Scaling
import           ML.ARAL.Proxy
import           ML.ARAL.Settings
import           ML.ARAL.Step                                (setDropoutValue)
import           ML.ARAL.Type
import           ML.ARAL.Types

-- Model uncertainty for Dropout ANNS
calcModelCertainty :: (MonadIO m) => ARAL s as -> m [Double]
calcModelCertainty aral = fmap (computeProbs . processOuts . mWarn) . replicateM n . runANN . getPx . setDropoutValue True $ aral
  where
    mWarn xs@(v:vals)
      | all (== v) vals = $(pureLogPrintErrorText) ("calcModelCertainty: Elements are all equal. Are you using Dropout?") xs
      | otherwise = xs
    runANN :: (MonadIO m) => Proxy -> m Values
    runANN px = do
      -- preds <- forM [1.n] $ \_ -> mlp model stochastic testImg
      emptyCache
      lookupActionsNeuralNetwork Target sFeats px
    processOuts :: [Values] -> [VB.Vector (VB.Vector (Int, Double))]
    processOuts = map (filterActions . indexActions . fromValues)
    filterActions = VB.zipWith (\disAsIdx -> VB.ifilter (\i _ -> i `notElem` VS.toList disAsIdx)) disallowedActionIndices
    indexActions = VB.map (\xs -> VB.zip (VB.generate nrAs id) . VB.convert $ xs)
    nrAs = VB.length (aral ^. actionList)
    sFeats :: StateFeatures
    sFeats = (aral ^. featureExtractor) (aral ^. s)
    DisallowedActionIndicies disallowedActionIndices = actionIndicesDisallowed aral (aral ^. s)
    n = 100
    getPx rl =
      case rl ^. algorithm of
        AlgARAL {}      -> rl ^. proxies . r1
        AlgNBORL {}     -> rl ^. proxies . v
        AlgARALVOnly {} -> rl ^. proxies . v
        AlgRLearning    -> rl ^. proxies . v
        AlgDQN {}       -> rl ^. proxies . r1
    mEmptyConts :: M.Map Int Int
    mEmptyConts = M.fromList . map (, 0) $ [0 .. nrAs - 1]
    computeProbs :: [VB.Vector (VB.Vector (Int, Double))] -> [Double]
    computeProbs = calcProbs . foldl' accumCounts mEmptyConts . map computMaxIdx
    computMaxIdx :: VB.Vector (VB.Vector (Int, Double)) -> Int
    computMaxIdx xs
      | VB.length xs == 1 = fst . VB.maximumBy (compare `on` snd) . VB.head $ xs
      | otherwise = error "calcModelCertainty currently not implemented for nrAgents > 1 !"
    accumCounts m aIdx = M.insertWith (+) aIdx 1 m
    calcProbs :: M.Map Int Int -> [Double]
    calcProbs = M.elems . M.map ((/ fromIntegral n) . fromIntegral)


plotProxyFunction :: Bool -> Int -> Int -> ARAL s as -> LookupType -> Proxy -> IO ()
plotProxyFunction  inclIterToFilenames dim1 dim2 rl lookupType px@Hasktorch{} =
  mapM_ (\aIdx -> plotHasktorchAction inclIterToFilenames aIdx dim1 dim2 rl lookupType px) [0.. px^?!proxyNrActions - 1]

plotProxyFunction  inclIterToFilenames dim1 dim2 _ lookupType _ =
  $(logPrintErrorText) "plotProxyFunction not implemented for specified proxy type"


-- | Takes node index, two dimension dim1 and dim2 (which both must be < nr of Coefficients) for x axis (input dim) and -/+ StdDev to draw (dim2) to plot and model.
plotHasktorchAction :: Bool -> Int -> Int -> Int -> ARAL s as -> LookupType -> Proxy -> IO ()
plotHasktorchAction inclIterToFilenames nodeIdx dim1 dim2 rl lookupType px@(Hasktorch netT netW tp nnCfg nrNodes 1 opt mdl wel _)
  | isWelfordExistingAggregateEmpty wel = $(logPrintErrorText) $ "Cannot plot regression node " <> tshow nodeIdx <> ". Add data first!"
  | nodeIdx >= nrNodes = $(logPrintErrorText) $ "Node index out of range: " <> tshow nodeIdx <> "<" <> tshow nrNodes
  | dim1 >= len = $(logPrintErrorText) $ "Dimension 1 out of range: " <> tshow dim1 <> "<" <> tshow len
  | dim2 >= len = $(logPrintErrorText) $ "Dimension 2 out of range: " <> tshow dim2 <> "<" <> tshow len
  | not (nnCfg ^?! autoNormaliseInput) = $(logPrintErrorText) "Cannot plot nets that are not auto-normalised!"
  | otherwise = do
    progName <- getProgName
    let iter = welfordCount wel
        (means, _, variances) = finalize wel
    let iterName
          | inclIterToFilenames = "_" ++ show iter
          | otherwise = ""
    let file = "/tmp/output_" ++ progName ++ "_" ++ show nodeIdx ++ iterName
        fileObs = "/tmp/observations_" ++ progName ++ "_" ++ show nodeIdx ++ iterName
        mean1 = means VS.! dim1
        stdDev1 = sqrt (variances VS.! dim1)
        leftBorder1 = mean1 - 8 * stdDev1
        rigthBorder1 = mean1 + 8 * stdDev1
        dist1 = max 0.01 (rigthBorder1 - leftBorder1)
        mean2 = means VS.! dim2
        stdDev2 = sqrt (variances VS.! dim2)
        leftBorder2 = mean2 - 8 * stdDev2
        rigthBorder2 = mean2 + 8 * stdDev2
        dist2 = max 0.01 (rigthBorder2 - leftBorder2)
        scaleIn = normaliseStateFeature wel
        scaleOut = scaleValue scaleAlg (getMinMaxVal px)
        unscaleOut = unscaleDouble scaleAlg (getMinMaxVal px)
        scaleAlg = nnCfg ^. scaleOutputAlgorithm
    writeFile file "x\ty\tzModel\n"
    fh <- openFile file AppendMode
    let xs = [leftBorder1,leftBorder1 + 0.025 * dist1 .. rigthBorder1]
        ys = [leftBorder2,leftBorder2 + 0.025 * dist2 .. rigthBorder2]
    forM_ xs $ \x -> do
      forM_ ys $ \y -> do
        let vecOrigScale = welfordMeanUnsafe wel VS.// [(dim1, x), (dim2, y)]
            vec = scaleIn vecOrigScale
            mdl = case lookupType of
              Target -> netT
              Worker -> netW
        zHat <- unscaleOut . (VS.! nodeIdx) . VB.head . fromValues . head <$> runHasktorch mdl nrNodes 1 (Just wel) vec
        -- hPutStrLn fh (show (vec VS.! dim1) ++ "\t" ++ show (vec VS.! dim2) ++ "\t" ++ show zHat)
        hPutStrLn fh (show x ++ "\t" ++ show y ++ "\t" ++ show zHat)
    hClose fh
    -- Example Learn Points
    case rl ^. proxies . replayMemory of
      Nothing -> return ()
      Just repMem -> do
        mems <- getRandomReplayMemoriesElements (rl ^. settings . nStep) (nnCfg ^. trainBatchSize) repMem
        -- let nsLearnObs = sortBy (comparing ((VS.! dim1) . getStateFeats)) $ mems
        writeFile fileObs "x\ty\tzObs\n"
        let mkCalc (s, idx, rew, s', epiEnd) = (mkCalculation MainAgent rl) s idx rew s' epiEnd
        calcs <- concat <$> mapM (executeAndCombineCalculations mkCalc) mems
        fhObs <- openFile fileObs AppendMode
        let getFeat ((sf, _, _, _), _) = sf
            getCalc ((_, _, _, _), calc) = calc
            getOut = case tp of
              R1Table -> fmap ((VS.! 0) . unpackValue) . getR1ValState' . getCalc
              R0Table -> fmap ((VS.! 0) . unpackValue) . getR0ValState' . getCalc
              _       -> fmap ((VS.! 0) . unpackValue) . getVValState'  . getCalc
        forM_ calcs $ \obs -> do
          hPutStrLn fhObs $ show (getFeat obs VS.! dim1) ++ "\t" ++ show (getFeat obs VS.! dim2) ++ "\t" ++ (maybe "-" show . getOut $ obs)
        hClose fhObs
  --
  -- Open
    let cmd =
          "gnuplot -e 'set terminal wxt lw 1; set terminal wxt fontscale 1; set key autotitle columnhead; set title \"Hasktorch Action " ++
          show nodeIdx ++ " " ++ show tp ++ " " ++ show lookupType ++
          ": Iteration " ++ show iter ++ "\"; " ++
          "set xlabel \"Dim " ++ show dim1 ++ "\";" ++ "set ylabel \"Dim " ++ show dim2 ++ "\"; " ++
        -- "set xrange [" ++ show leftBorder1 ++ ":" ++ show rigthBorder1 ++ "]; " ++
          -- maybe "" (\(minV, maxV) -> "set zrange [" ++ show minV ++ ":" ++ show maxV ++ "]; ") (regConfigClipOutput $ regNodeConfig node) ++
        -- "set dgrid3d " ++ show (length xs) ++ "," ++ show (length ys) ++ "; " ++
        -- "set hidden3d; " ++
          "splot \"" ++ file ++ "\" u 1:2:3 with points, \"" ++ file ++ "\" u 1:2:4 with points" ++
          ", \"" ++ fileObs ++ "\" u 1:2:3 with points" ++
          "; pause mouse close;' & "
    pss <- filter (not . T.isInfixOf "grep") . map T.pack . lines <$> readCreateProcess (shell "ps aux | grep gnuplot") ""
    -- let ps = filter (\x -> T.isInfixOf (T.pack file) x && T.isInfixOf (T.pack fileObs) x) pss
    let ps = filter (\x -> T.isInfixOf (T.pack file) x) pss
    let exists = not . null $ ps
    if exists
      then putStrLn "Gnuplot already open!"
      else void $ createProcess $ shell cmd
    putStrLn cmd
  where
    len = VS.length (welfordMeanUnsafe wel)
    tshow = T.pack . show
plotHasktorchAction inclIterToFilenames nodeIdx dim1 dim2 _ lookupType px@(Hasktorch netT netW tp nnCfg nrNodes _ opt mdl wel _) =
  $(logPrintErrorText) "plotHasktorchAction: Not defined for nrAgents > 1"
plotHasktorchAction inclIterToFilenames nodeIdx dim1 dim2 _ lookupType _ = error "plotHasktorchAction: Should not be called on non-Hasktorch Proxy"

