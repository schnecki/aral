{-# LANGUAGE BangPatterns #-}
module Helper
    ( askUser

    ) where

import           ML.BORL

import           Control.Arrow
import           Control.DeepSeq (NFData, force)
import           Control.Lens    (over, set, traversed, (^.))
import           Control.Monad   (foldM, unless, when)
import           Data.Function   (on)
import           Data.List       (find, sortBy)
import           Data.Time.Clock
import           System.CPUTime
import           System.IO
import           System.Random
import           Text.Printf

askUser :: (NFData s, Ord s, Show s, RewardFuture s) => Bool -> [(String,String)] -> [(String, ActionIndexed s)] -> BORL s -> IO ()
askUser showHelp addUsage cmds ql = do
  let usage =
        sortBy (compare `on` fst) $
        [ ("v", "Print V+W tables")
        , ("p", "Print everything")
        , ("q", "Exit program (unsaved state will be lost)")
        , ("r", "Run for X times")
        , ("m", "Multiply all state values by X")
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
    "h" -> askUser True addUsage cmds ql
    "?" -> askUser True addUsage cmds ql
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
        [(nr, _)] -> time (steps ql nr) >>= askUser False addUsage cmds
        _ -> do
          putStr "Could not read your input :( You are supposed to enter an Integer.\n"
          askUser False addUsage cmds ql
    "p" -> do
      prettyBORL ql >>= print
      askUser False addUsage cmds ql
    "m" -> do
      putStr "Multiply by: " >> hFlush stdout
      l <- getLine
      case reads l :: [(Double, String)] of
        [(nr, _)] -> askUser False addUsage cmds (foldl (\q f -> over (proxies . f) (multiplyProxy nr) q) ql [psiV, v, w])
        _ -> do
          putStr "Could not read your input :( You are supposed to enter an Integer.\n"
          askUser False addUsage cmds ql
    "v" -> do
      case find isTensorflow (allProxies $ ql ^. proxies) of
        Nothing -> runMonadBorlIO $ prettyBORLTables True False False ql >>= print
        Just _ -> runMonadBorlTF (restoreTensorflowModels True ql >> prettyBORLTables True False False ql) >>= print
      askUser False addUsage cmds ql
    _ ->
      case find ((== c) . fst) cmds of
        Nothing ->
          unless
            (c == "q")
            (step ql >>= \x ->
               case find isTensorflow (allProxies $ ql ^. proxies) of
                 Nothing -> runMonadBorlIO $ prettyBORLTables True False False x >>= print >> askUser False addUsage cmds x
                 Just _ -> runMonadBorlTF (restoreTensorflowModels True x >> prettyBORLTables True False True x) >>= print >> askUser False addUsage cmds x)
        Just (_, cmd) ->
          case find isTensorflow (allProxies $ ql ^. proxies) of
            Nothing -> runMonadBorlIO $ stepExecute (ql, False, cmd) >>= askUser False addUsage cmds
            Just _ -> runMonadBorlTF (restoreTensorflowModels True ql >> stepExecute (ql, False, cmd) >>= saveTensorflowModels) >>= askUser False addUsage cmds


time :: NFData t => IO t -> IO t
time a = do
    start <- getCurrentTime
    !val <- force <$> a
    end   <- getCurrentTime
    putStrLn ("Computation Time: " ++ show (diffUTCTime end start))
    return val
