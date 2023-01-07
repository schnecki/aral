{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections       #-}
module Helper
    ( askUser
    , getIOMWithDefault
    , getIOWithDefault
    , chooseAlg
    ) where

import           Grenade
import           ML.ARAL
import           ML.ARAL.InftyVector
import           RegNet

import           Control.Arrow
import           Control.DeepSeq        (NFData, force)
import           Control.Lens
import           Control.Lens           (over, set, traversed, (^.))
import           Control.Monad          (foldM, unless, when)
import           Control.Monad.IO.Class (liftIO)
import qualified Data.ByteString        as BS
import           Data.Function          (on)
import           Data.List              (find, sortBy)
import           Data.Maybe             (fromMaybe)
import           Data.Serialize         as S
import           Data.Time.Clock
import qualified Data.Vector            as VB
import           System.CPUTime
import           System.IO
import           System.Random
import           Text.Printf

import           Debug.Trace

askUser ::
     (NFData s, Ord s, Show s, Serialize s, NFData as, Serialize as, Show as, RewardFuture s, Eq as)
  => Maybe (NetInputWoAction -> Maybe (Either String s))
  -> Bool
  -> [(String, String)]
  -> [(String, [ActionIndex])]
  -> [(String, String, ARAL s as -> IO (ARAL s as))]
  -> ARAL s as
  -> IO ()
askUser mInverse showHelp addUsage cmds qlCmds ql = do
  let usage =
        sortBy (compare `on` fst) $
        [ ("h", "Print head")
        , ("i", "Print info head only")
        , ("l", "Load from file save.dat")
        , ("p", "Print everything")
        , ("param", "Change parameters")
        , ("pr", "Plot RegNet")
        , ("q", "Exit program (unsaved state will be lost)")
        , ("r", "Run for X times")
        , ("s", "Save to file save.dat (overwrites the file if it exists)")
        , ("v", "Print V+W tables")
        , ("_", "Any other input starts another learning round\n")
        ] ++
        addUsage ++
        map (\(c, h, _) -> (c, h)) qlCmds
  putStrLn ""
  when showHelp $ putStrLn $ unlines $ map (\(c, h) -> c ++ ": " ++ h) usage
  putStr "Enter value ('help' for help): " >> hFlush stdout
  c <- getLine
  case c of
    "help" -> askUser mInverse True addUsage cmds qlCmds ql
    "?" -> askUser mInverse True addUsage cmds qlCmds ql
    "s" -> do
      let path = "save.dat"
      res <- toSerialisableWith id id ql
      BS.writeFile path (runPut $ put res)
      putStrLn $ "Saved to " ++ path
      askUser mInverse True addUsage cmds qlCmds ql
    "l" -> do
      let path = "save.dat"
      bs <- liftIO $ BS.readFile path
      ql' <- case S.runGet S.get bs of
        Left err  -> error ("Could not deserialize to Serializable ARAL. Error: " ++ err)
        Right ser -> fromSerialisableWith id id (ql ^. actionFunction) (ql ^. actionFilter) (ql ^. featureExtractor) ser
      let ql'' = overAllProxies (proxyNNConfig . prettyPrintElems) (\pp -> pp ++ [(ql' ^. featureExtractor) (ql' ^. s)]) ql'
      prettyARALWithStInverse mInverse ql'' >>= print >> hFlush stdout
      putStrLn $ "Loaded from " ++ path
      askUser mInverse True addUsage cmds qlCmds ql''
    "r" -> do
      putStr "How many learning rounds should I execute: " >> hFlush stdout
      l <- getLine
      case reads l :: [(Integer, String)] of
        [(nr, _)] -> do
          putStr "How often shall I repeat this? [1] " >> hFlush stdout
          l <- getLine
          case reads l :: [(Integer, String)] of
            [(often, _)] ->
              let runner borl = do
                    foldM
                      (\q _ -> do
                         q' <- stepsM q nr
                         let qPP = overAllProxies (proxyNNConfig . prettyPrintElems) (\pp -> pp ++ [(q' ^. featureExtractor) (borl ^. s), (q' ^. featureExtractor) (q' ^. s)]) q'
                         output <- prettyARALMWithStInverse mInverse qPP
                         liftIO $ print output >> hFlush stdout
                         liftIO $ case q' ^. proxies . r1 of
                           RegressionProxy lay _ _ | q' ^. t > 10 -> mapM_ (\i -> plotRegressionNode i 0 1 Nothing lay) [0..regNetNodes lay - 1]
                           _                                      -> return ()
                         return q')
                      borl
                      [1 .. often]
               in liftIO $ time (runner ql) >>= askUser mInverse False addUsage cmds qlCmds
             -- -> do
             --  ql' <-
             --    foldM
             --      (\q _ -> do
             --         q' <- time (steps q nr)
             --         let qPP = overAllProxies (proxyNNConfig . prettyPrintElems) (\pp -> pp ++ [(q' ^. featureExtractor) (ql ^. s), (q' ^. featureExtractor) (q' ^. s)]) q'
             --         liftIO $ prettyARALWithStInverse mInverse qPP >>= print >> hFlush stdout
             --         return q')
             --      ql
             --      [1 .. often]
             --  askUser mInverse False addUsage cmds qlCmds ql'
            _ -> time (steps ql nr) >>= askUser mInverse False addUsage cmds qlCmds
        _ -> do
          putStr "Could not read your input :( You are supposed to enter an Integer.\n"
          askUser mInverse False addUsage cmds qlCmds ql
    "p" -> do
      let ql' = overAllProxies (proxyNNConfig . prettyPrintElems) (\pp -> pp ++ [(ql ^. featureExtractor) (ql ^. s)]) ql
      prettyARALWithStInverse mInverse ql' >>= print >> hFlush stdout
      askUser mInverse False addUsage cmds qlCmds ql
    "pr" -> do
      case ql ^. proxies . r1 of
        RegressionProxy lay _ _ | ql ^. t > 10 -> mapM_ (\i -> plotRegressionNode i 0 1 Nothing lay) [0..regNetNodes lay - 1]
        _                                      -> return ()
      askUser mInverse False addUsage cmds qlCmds ql
    "rp" -> do
      let doc = maybe mempty  prettyRegressionLayer (ql ^? proxies . r1 . proxyRegressionLayer)
      print doc
      askUser mInverse False addUsage cmds qlCmds ql
    "h" -> do
      prettyARALHead True mInverse ql >>= print >> hFlush stdout
      askUser mInverse False addUsage cmds qlCmds ql
    "i" -> do
      prettyARALHead True mInverse ql >>= print >> hFlush stdout
      askUser mInverse False addUsage cmds qlCmds ql
    "v" -> do
      liftIO $ prettyARALTables mInverse True False False ql >>= print
      askUser mInverse False addUsage cmds qlCmds ql
    "param" -> do
      e <-
        liftIO $ do
          putStrLn "Which settings to change:"
          putStrLn $ unlines $ map (\(c, h) -> c ++ ": " ++ h) $
            sortBy
              (compare `on` fst)
              [ ("alpha", "alpha")
              , ("exp", "exploration rate")
              , ("eps", "epsilon")
              , ("lr", "learning rate")
              , ("dislearn", "Disable/Enable all learning")
              , ("drop", "flip Grenade dropout layer active flag")
              ]
          liftIO $ putStr "Enter value: " >> hFlush stdout >> getLine
      ql' <-
        case e of
          "drop" -> do
            liftIO $ putStr "New value (True or False): " >> hFlush stdout
            liftIO $
              maybe
                ql
                (\(v' :: Bool) ->
                   overAllProxies
                     (filtered isGrenade)
                     (\(Grenade tar wor tp cfg act agents wel) -> Grenade (runSettingsUpdate (NetworkSettings v') tar) (runSettingsUpdate (NetworkSettings v') wor) tp cfg act agents wel)
                     ql) <$>
              getIOMWithDefault Nothing
          "alpha" -> do
            liftIO $ putStr "New value: " >> hFlush stdout
            liftIO $ maybe ql (\v' -> decaySetting . alpha .~ NoDecay $ parameters . alpha .~ v' $ ql) <$> getIOMWithDefault Nothing
          "exp" -> do
            liftIO $ putStr "New value: " >> hFlush stdout
            liftIO $ maybe ql (\v' -> decaySetting . exploration .~ NoDecay $ parameters . exploration .~ v' $ ql) <$> getIOMWithDefault Nothing
          "eps" -> do
            liftIO $ putStr "New value: " >> hFlush stdout
            liftIO $ maybe ql (\v' -> decaySetting . epsilon .~ Last NoDecay $ parameters . epsilon .~ Last (v' :: Double) $ ql) <$> getIOMWithDefault Nothing
          "lr" -> do
            liftIO $ putStr "New value: " >> hFlush stdout
            liftIO $
              maybe
                ql
                (\v' -> overAllProxies (proxyNNConfig . grenadeLearningParams) (setLearningRate v') $ overAllProxies (proxyNNConfig . learningParamsDecay) (const NoDecay) ql) <$>
              getIOMWithDefault Nothing
          "dislearn" -> do
            liftIO $ putStr "New value (True or False): " >> hFlush stdout
            liftIO $ maybe ql (\v' -> settings . disableAllLearning .~ v' $ ql) <$> getIOMWithDefault Nothing
          _ -> liftIO $ putStrLn "Did not understand the input" >> return ql
      askUser mInverse False addUsage cmds qlCmds ql'
    _ ->
      case find ((== c) . fst) cmds of
        Nothing ->
          case find ((== c) . fst) (map (\(c, _, f) -> (c, f)) qlCmds) of
            Nothing ->
              unless
                (c == "q")
                (step ql >>= \x -> do
                   let ppQl = setAllProxies (proxyNNConfig . prettyPrintElems) [(ql ^. featureExtractor) (ql ^. s)] x
                   liftIO $ prettyARALTables mInverse True False False ppQl >>= print >> askUser mInverse False addUsage cmds qlCmds x)
            Just (_, f) -> f ql >>= askUser mInverse False addUsage cmds qlCmds
        Just (_, cmd) ->
          liftIO $ stepExecute ql (VB.map (False,) (VB.fromList cmd), []) >>= askUser mInverse False addUsage cmds qlCmds


time :: NFData t => IO t -> IO t
time a = do
    start <- getCurrentTime
    !val <- force <$> a
    end   <- getCurrentTime
    putStrLn ("Computation Time: " ++ show (diffUTCTime end start))
    return val

getIOMWithDefault :: forall m a . (Monad m, Read a) => m a -> IO (m a)
getIOMWithDefault def = do
  line <- getLine
  case reads line :: [(a, String)] of
    [(x, _)] -> return $ return x
    _        -> return def


getIOWithDefault :: forall a . (Read a) => a -> IO a
getIOWithDefault def = fromMaybe def <$> getIOMWithDefault (Just def)


chooseAlg :: Maybe (s, ActionIndex) -> IO (Algorithm s)
chooseAlg mRefState = do
  putStrLn "\nChoose Algorithm:"
  putStrLn "------------------------------"
  putStrLn $ unlines
    [ "0: AlgARAL 0.8 1.0 ByStateValues (DEFAULT)"
    , "1: AlgARAL 0.8 0.99 ByStateValues"
    , "2: AlgDQN 0.99 Exact"
    , "3: AlgRLearning"
    , "4: AlgARAL 0.9 0.99 ByStateValues"
    , "5: Q-Learning with gamma=0.99"
    ]
  putStr "Enter number [0]: "
  hFlush stdout
  nr <- getIOWithDefault 0
  return $
    case nr of
      1 -> AlgARAL 0.8 0.99 ByStateValues
      2 -> AlgDQN 0.99 Exact
      3 -> AlgRLearning
      4 -> AlgARAL 0.9 0.99 ByStateValues
      5 -> AlgDQN 0.99 Exact
      _ -> AlgARAL 0.8 1.0 ByStateValues
