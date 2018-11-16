module Helper
    ( askUser

    ) where

import           ML.BORL

import           Control.Arrow
import           Control.Lens  (set, (^.))
import           Control.Monad (foldM, unless, when)
import           Data.Function (on)
import           Data.List     (find, sortBy)
import           System.IO
import           System.Random

askUser :: (Ord s, Show s) => Bool -> [(String,String)] -> [(String, ActionIndexed s)] -> BORL s -> IO ()
askUser showHelp addUsage cmds ql = do
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
        [(nr, _)] -> foldM (\q _ -> step q) ql [0 .. nr - 1] >>= askUser False addUsage cmds
        _ -> do
          putStr "Could not read your input :( You are supposed to enter an Integer.\n"
          askUser False addUsage cmds ql
    "p" -> do
      print (prettyBORL ql)
      askUser False addUsage cmds ql
    "v" -> do
      print (prettyBORLTables True False False ql)
      askUser False addUsage cmds ql
    _ ->
      case find ((== c) . fst) cmds of
        Nothing -> unless (c == "q") (step ql >>= \x -> print (prettyBORLTables True False True x) >> return x >>= askUser False addUsage cmds)
        Just (_, cmd) -> stepExecute ql (False, cmd) >>= askUser False addUsage cmds
