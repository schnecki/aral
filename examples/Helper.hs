{-# LANGUAGE BangPatterns #-}
module Helper
    ( askUser

    ) where

import           ML.BORL

import           Control.Arrow
import           Control.DeepSeq        (NFData, force)
import           Control.Lens           (over, set, traversed, (^.))
import           Control.Monad          (foldM, unless, when)
import           Control.Monad.IO.Class (liftIO)
import           Data.Function          (on)
import           Data.List              (find, sortBy)
import           Data.Time.Clock
import           System.CPUTime
import           System.IO
import           System.Random
import           Text.Printf

askUser :: (NFData s, Ord s, Show s, RewardFuture s) => Maybe (NetInputWoAction -> Maybe s) -> Bool -> [(String,String)] -> [(String, ActionIndexed s)] -> BORL s -> IO ()
askUser mInverse showHelp addUsage cmds ql = do
  let usage =
        sortBy (compare `on` fst) $
        [ ("v", "Print V+W tables")
        , ("p", "Print everything")
        , ("q", "Exit program (unsaved state will be lost)")
        , ("r", "Run for X times")
        -- , ("s" "Save to file save.dat (overwrites the file if it exists)")
        -- , ("l" "Load from file save.dat")
        , ("_", "Any other input starts another learning round\n")
        ] ++
        addUsage
  putStrLn ""
  when showHelp $ putStrLn $ unlines $ map (\(c, h) -> c ++ ": " ++ h) usage
  putStr "Enter value (h for help): " >> hFlush stdout
  c <- getLine
  case c of
    "h" -> askUser mInverse True addUsage cmds ql
    "?" -> askUser mInverse True addUsage cmds ql
    -- "s" -> do
    --   saveQL ql "save.dat"
    --   askUser ql addUsage cmds
    -- "l" -> do
    --   ql' <- loadQL ql "save.dat"
    --   print (prettyQLearner prettyState (text . show) ql')
    --   askUser ql addUsage cmds'
    "r" -> do
      putStr "How many learning rounds should I execute: " >> hFlush stdout
      l <- getLine
      case reads l :: [(Integer, String)] of
        [(nr, _)] -> do
          putStr "How often shall I repeat this? [1] " >> hFlush stdout
          l <- getLine
          case reads l :: [(Integer, String)] of
            [(often, _)] -> do
              ql' <-
                runMonadBorlTF $ do
                  restoreTensorflowModels True ql
                  borl' <-
                    foldM
                      (\q _ -> do
                         q' <- stepsM q nr
                         output <- prettyBORLMWithStateInverse mInverse q'
                         liftIO $ print output >> hFlush stdout
                         return q')
                      ql
                      [1 .. often]
                  saveTensorflowModels borl'
              askUser mInverse False addUsage cmds ql'
              -- ql' <- foldM (\q _ -> do
              --           q' <- time (steps q nr)
              --           liftIO $ prettyBORLWithStInverse mInverse q' >>= print >> hFlush stdout
              --           return q'
              --       ) ql [1 .. often]
              -- askUser mInverse False addUsage cmds ql'

            _ -> time (steps ql nr) >>= askUser mInverse False addUsage cmds
        _ -> do
          putStr "Could not read your input :( You are supposed to enter an Integer.\n"
          askUser mInverse False addUsage cmds ql
    "p" -> do
      prettyBORLWithStInverse mInverse ql >>= print >> hFlush stdout
      askUser mInverse False addUsage cmds ql
    "v" -> do
      case find isTensorflow (allProxies $ ql ^. proxies) of
        Nothing -> runMonadBorlIO $ prettyBORLTables mInverse True False False ql >>= print
        Just _ -> runMonadBorlTF (restoreTensorflowModels True ql >> prettyBORLTables mInverse True False False ql) >>= print
      askUser mInverse False addUsage cmds ql
    _ ->
      case find ((== c) . fst) cmds of
        Nothing ->
          unless
            (c == "q")
            (step ql >>= \x ->
               case find isTensorflow (allProxies $ ql ^. proxies) of
                 Nothing -> runMonadBorlIO $ prettyBORLTables mInverse True False False x >>= print >> askUser mInverse False addUsage cmds x
                 Just _ -> runMonadBorlTF (restoreTensorflowModels True x >> prettyBORLTables mInverse True False True x) >>= print >> askUser mInverse False addUsage cmds x)
        Just (_, cmd) ->
          case find isTensorflow (allProxies $ ql ^. proxies) of
            Nothing -> runMonadBorlIO $ stepExecute (ql, False, cmd) >>= askUser mInverse False addUsage cmds
            Just _ -> runMonadBorlTF (restoreTensorflowModels True ql >> stepExecute (ql, False, cmd) >>= saveTensorflowModels) >>= askUser mInverse False addUsage cmds


time :: NFData t => IO t -> IO t
time a = do
    start <- getCurrentTime
    !val <- force <$> a
    end   <- getCurrentTime
    putStrLn ("Computation Time: " ++ show (diffUTCTime end start))
    return val
