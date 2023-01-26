{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell   #-}
module ML.ARAL.NeuralNetwork.Debug
    ( plotProxyFunction

    ) where

import           Control.Lens
import           Control.Monad
import           Data.List                                   (intercalate, sortBy)
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

import           ML.ARAL.Calculation.Ops
import           ML.ARAL.Calculation.Type
import           ML.ARAL.NeuralNetwork.Hasktorch
import           ML.ARAL.NeuralNetwork.NNConfig
import           ML.ARAL.NeuralNetwork.Normalisation
import           ML.ARAL.NeuralNetwork.ReplayMemory
import           ML.ARAL.NeuralNetwork.Scaling
import           ML.ARAL.Proxy
import           ML.ARAL.Settings
import           ML.ARAL.Type
import           ML.ARAL.Types


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
    writeFile file "x\ty\tzModel\n"
    fh <- openFile file AppendMode
    let xs = [leftBorder1,leftBorder1 + 0.025 * dist1 .. rigthBorder1]
        ys = [leftBorder2,leftBorder2 + 0.025 * dist2 .. rigthBorder2]
    forM_ xs $ \x -> do
      forM_ ys $ \y -> do
        let vec = welfordMeanUnsafe wel VS.// [(dim1, x), (dim2, y)]
            -- mGrTruth' vec' = maybe "-" (\f -> show (f vec')) mGrTruth
            -- z = mGrTruth' vec
            mdl = case lookupType of
              Target -> netT
              Worker -> netW
            zHat = (VS.! nodeIdx) . VB.head . fromValues . head $ runHasktorch mdl nrNodes 1 (Just wel) vec
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
        let getFeat ((sf, _, _, _), _) = scaleIn sf
            getCalc ((_, _, _, _), calc) = calc
            scaleAlg = nnCfg ^. scaleOutputAlgorithm
            scaleIn = normaliseStateFeature wel
            scaleOut = scaleValue scaleAlg (getMinMaxVal px)
            getOut = case tp of
              R1Table -> fmap ((VS.! 0) . unpackValue . scaleOut) . getR1ValState' . getCalc
              R0Table -> fmap ((VS.! 0) . unpackValue . scaleOut) . getR0ValState' . getCalc
              _       -> fmap ((VS.! 0) . unpackValue . scaleOut) . getVValState'  . getCalc
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

