{-# LANGUAGE CPP               #-}
{-# LANGUAGE OverloadedStrings #-}
module ML.BORL.Pretty
    ( prettyTable
    , prettyBORL
    , prettyBORLM
    , prettyBORLMWithStateInverse
    , prettyBORLWithStInverse
    , prettyBORLHead
    , prettyBORLTables
    , wideStyle
    ) where


import           ML.BORL.Action
import           ML.BORL.Algorithm
import           ML.BORL.Decay
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import qualified ML.BORL.Proxy         as P
import           ML.BORL.Proxy.Ops     (LookupType (..), getMinMaxVal, lookupNeuralNetwork,
                                        mkNNList)
import           ML.BORL.Proxy.Proxies
import           ML.BORL.Proxy.Type
import           ML.BORL.SaveRestore
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Arrow         (first, second, (&&&), (***))
import           Control.Lens
import           Control.Monad         (when)
import           Data.Function         (on)
import           Data.List             (find, foldl', sort, sortBy)
import qualified Data.Map.Strict       as M
import           Data.Maybe            (fromMaybe)
import qualified Data.Set              as S
import qualified Data.Text             as T
import           Grenade
import           Prelude               hiding ((<>))
import           System.IO.Unsafe      (unsafePerformIO)
import           Text.PrettyPrint      as P
import           Text.Printf

import           Debug.Trace

commas :: Int
commas = 3

nestCols :: Int
nestCols = 55

wideStyle :: Style
wideStyle = Style { lineLength = 200, ribbonsPerLine = 100, mode = PageMode }

printFloat :: Double -> Doc
printFloat = text . showFloat

showFloat :: Double -> String
showFloat = printf ("%." ++ show commas ++ "f")

printFloatWith :: Int -> Double -> Doc
printFloatWith commas x = text $ printf ("%." ++ show commas ++ "f") x


prettyTable :: (MonadBorl' m, Show k, Eq k, Ord k) => BORL k -> (NetInputWoAction -> Maybe (Maybe k, String)) -> (ActionIndex -> Doc) -> P.Proxy -> m Doc
prettyTable borl prettyKey prettyIdx p = vcat <$> prettyTableRows borl prettyKey prettyIdx (\_ v -> return v) p

prettyTableRows ::
     (MonadBorl' m, Show k, Ord k, Eq k)
  => BORL k
  -> (NetInputWoAction -> Maybe (Maybe k, String))
  -> (ActionIndex -> Doc)
  -> (([Double], ActionIndex) -> Double -> m Double)
  -> P.Proxy
  -> m [Doc]
prettyTableRows borl prettyState prettyActionIdx modifier p =
  case p of
    P.Table m _ ->
      let mkAct idx = actionName $ snd $ (borl ^. actionList) !! (idx `mod` length (borl ^. actionList))
          mkInput k = text (filter (/= '"') $ show $ map (\x -> if x < 0 then printFloat x else "+" <> printFloat x) k)
      in mapM (\((k,idx),val) -> modifier (k,idx) val >>= \v -> return (mkInput k <> text (T.unpack $ mkAct idx) <> colon <+> printFloat v)) $
      sortBy (compare `on`  fst.fst) $ M.toList m
    pr -> do
      mtrue <- mkListFromNeuralNetwork borl prettyState prettyActionIdx True pr
      let printFun (kDoc, (valT, valW)) | isEmpty kDoc = []
                                        | otherwise = [kDoc <> colon <+> printFloat valT <+> text "  " <+> printFloat valW]
          unfoldActs = concatMap (\(f,(ts,ws)) -> zipWith (\(nr,t) (_,w) -> (f nr, (t, w))) ts ws)
      return $ concatMap printFun (unfoldActs mtrue)


mkListFromNeuralNetwork ::
     (MonadBorl' m, Show k, Ord k, Eq k)
  => BORL k
  -> (NetInputWoAction -> Maybe (Maybe k, String))
  -> (ActionIndex -> Doc)
  -> Bool
  -> P.Proxy
  -> m [(ActionIndex -> Doc, ([(ActionIndex, Double)], [(ActionIndex, Double)]))]
mkListFromNeuralNetwork borl prettyState prettyActionIdx scaled pr = do
  let subPr
        | isCombinedProxy pr = pr ^?! proxySub
        | otherwise = pr
  if borl ^. t <= pr ^?! proxyNNConfig . replayMemoryMaxSize
    then do
      let inp = map fst tbl
      let tableVals = tbl
      workerVals <- mapM (\x -> unscaleValue (getMinMaxVal pr) <$> lookupNeuralNetwork Worker x (set proxyType (NoScaling $ subPr ^?! proxyType) subPr)) inp
      return $ finalize $ zipWith (\((feat, actIdx), tblV) workerV -> (feat, ([(actIdx, tblV)], [(actIdx, workerV)]))) tbl workerVals
    else finalize <$> mkNNList borl scaled pr
  where
    finalize = map (first $ prettyStateActionEntry borl prettyState prettyActionIdx)
    tbl =
      case pr of
        P.Table t _                   -> M.toList t
        P.Grenade _ _ p _ _ _         -> M.toList p
        P.TensorflowProxy _ _ p _ _ _ -> M.toList p
        P.CombinedProxy p _ _         -> M.toList (p ^?! proxyNNStartup)
        _                             -> error "should not happen"

prettyStateActionEntry :: BORL k -> (NetInputWoAction -> Maybe (Maybe k, String)) -> (ActionIndex -> Doc) -> NetInputWoAction -> ActionIndex -> Doc
prettyStateActionEntry borl pState pActIdx stInp actIdx = case pState stInp of
  Nothing               -> mempty
  Just (Just st, stRep) | actIdx < length bools && bools !! actIdx -> text stRep <> colon <+> pActIdx actIdx
                        | otherwise -> mempty
    where bools = take (length $ borl ^. actionList) $ (borl ^. actionFilter) st
  Just (Nothing, stRep) -> text stRep <> colon <+> pActIdx actIdx


prettyTablesState :: (MonadBorl' m, Show s, Ord s) => BORL s -> (NetInputWoAction -> Maybe (Maybe s, String)) -> (ActionIndex -> Doc) -> P.Proxy -> (NetInputWoAction -> Maybe (Maybe s, String)) -> P.Proxy -> m Doc
prettyTablesState borl p1 pIdx m1 p2 m2 = do
  rows1 <- prettyTableRows borl p1 pIdx (\_ v -> return v) (if fromTable then tbl m1 else m1)
  rows2 <- prettyTableRows borl p2 pIdx (\_ v -> return v) (if fromTable then tbl m2 else m2)
  return $ vcat $ zipWith (\x y -> x $$ nest nestCols y) rows1 rows2
  where fromTable = period < fromIntegral memSize
        period = borl ^. t
        memSize = case m1 of
          P.Table{}                       -> -1
          P.Grenade _ _ _ _ cfg _         -> cfg ^?! replayMemoryMaxSize
          P.TensorflowProxy _ _ _ _ cfg _ -> cfg ^?! replayMemoryMaxSize
          P.CombinedProxy{}               -> m1 ^?! proxyNNConfig.replayMemoryMaxSize
        tbl px = case px of
          p@P.Table{}                   -> p
          P.Grenade _ _ p _ _ _         -> P.Table p 0
          P.TensorflowProxy _ _ p _ _ _ -> P.Table p 0
          P.CombinedProxy p nr _        -> P.Table (M.filterWithKey (\(_,aIdx) _ -> aIdx >= minKey && aIdx <= maxKey) (p ^?! proxyNNStartup)) 0
            where nrActs = p ^?! proxyNrActions
                  minKey = nr * nrActs
                  maxKey = minKey + nrActs - 1

prettyAlgorithm ::  BORL s -> (NetInputWoAction -> String) -> (ActionIndex -> Doc) -> Algorithm s -> Doc
prettyAlgorithm borl prettyState prettyActionIdx (AlgBORL ga0 ga1 avgRewType vPlusPsiV mRefState) =
  text "BORL with gammas " <+>
  text (show (ga0, ga1)) <> text ";" <+>
  prettyAvgRewardType avgRewType <+>
  text "for rho" <> text ";" <+>
  text "Deciding on" <+>
  text
    (if vPlusPsiV
       then "V + PsiV"
       else "V") <+>
  prettyRefState borl prettyState prettyActionIdx mRefState
prettyAlgorithm _ _ _ (AlgDQN ga1)      = text "DQN with gamma" <+> text (show ga1)
prettyAlgorithm borl _ _ (AlgDQNAvgRewardFree ga0 ga1 avgRewType)      = text "Average reward freed DQN with gammas" <+> text (show (ga0, ga1)) <+> ". Rho by" <+> prettyAvgRewardType avgRewType
  where decay = 0.5 ** (fromIntegral (borl ^. t) / 100000)
        ga1Diff = 1 - ga1
        ga1' = ga1 + ga1Diff - ga1Diff * decay

prettyAlgorithm borl prettyState prettyAction (AlgBORLVOnly avgRewType mRefState)      = text "BORL with V ONLY" <> text ";" <+> prettyAvgRewardType avgRewType <> prettyRefState borl prettyState prettyAction mRefState

prettyRefState :: (Show a) => BORL s -> ([Double] -> a) -> (t -> Doc) -> Maybe (s, t) -> Doc
prettyRefState _ _ _ Nothing = mempty
prettyRefState borl prettyState prettyAction (Just (st,aNr)) = ";" <+>  "Ref state: " <> text (show $ prettyState $ (borl ^. featureExtractor) st) <> " - " <> prettyAction aNr

prettyAvgRewardType :: AvgReward -> Doc
prettyAvgRewardType (ByMovAvg nr)          = "moving average" <> parens (int nr)
prettyAvgRewardType ByReward               = "reward"
prettyAvgRewardType ByStateValues          = "state values"
prettyAvgRewardType (ByStateValuesAndReward ratio) = printFloat ratio <> "*state values + " <> printFloat (1-ratio) <> "*reward"
prettyAvgRewardType (Fixed x)              = "fixed value of " <> double x


prettyBORLTables :: (MonadBorl' m, Ord s, Show s) => Maybe (NetInputWoAction -> Maybe s) -> Bool -> Bool -> Bool -> BORL s -> m Doc
prettyBORLTables mStInverse t1 t2 t3 borl = do
  let algDoc doc
        | isAlgBorl (borl ^. algorithm) = doc
        | otherwise = empty
      algDocRho doc =
        case borl ^. algorithm of
          AlgDQN {} -> mempty
          _         -> doc
  let prBoolTblsStateAction True h m1 m2 = (h $+$) <$> prettyTablesState borl prettyState prettyActionIdx m1 prettyState m2
      prBoolTblsStateAction False _ _ _ = return empty
  prettyRhoVal <-
    case borl ^. proxies . rho of
      Scalar val -> return $ text "Rho" <> colon $$ nest nestCols (printFloatWith 8 val)
      m -> do
        prAct <- prettyTable borl prettyState prettyActionIdx m
        return $ text "Rho" $+$ prAct
  docHead <- prettyBORLHead' False prettyState borl
  case borl ^. algorithm of
    AlgBORL {}
      -- let addPsiV k v =
      --       case borl ^. proxies . psiV of
      --         P.Table m def -> return $ M.findWithDefault def k m
      --         px ->
      --           let config = px ^?! proxyNNConfig
      --            in if borl ^. t <= config ^. replayMemoryMaxSize && (config ^. trainBatchSize) /= 1
      --                 then return $ M.findWithDefault 0 k (px ^. proxyNNStartup)
      --                 else do
      --                   vPsi <- P.lookupNeuralNetworkUnscaled P.Worker k (borl ^. proxies . psiV)
      --                   return (v + vPsi)
      -- vPlusPsiV <- prettyTableRows borl prettyState prettyActionIdx addPsiV (borl ^. proxies . v)
     -> do
      prVs <- prBoolTblsStateAction t1 (text "V" $$ nest nestCols (text "PsiV")) (borl ^. proxies . v) (borl ^. proxies . psiV)
      prWs <- prBoolTblsStateAction t1 (text "W" $$ nest nestCols (text "PsiW")) (borl ^. proxies . w) (borl ^. proxies . psiW)
      prW2s <- prBoolTblsStateAction t1 (text "W2" $$ nest nestCols (text "PsiW2")) (borl ^. proxies . w2) (borl ^. proxies . psiW2)
      prR0R1 <- prBoolTblsStateAction t2 (text "R0" $$ nest nestCols (text "R1")) (borl ^. proxies . r0) (borl ^. proxies . r1)
      return $ docHead $$ algDocRho prettyRhoVal $$ -- prVW $+$ prR0R1 $+$ psis $+$ prWW2
        prVs $+$
        prWs $+$
        prW2s $+$
        prR0R1
    AlgBORLVOnly {} -> do
      prV <- prettyTableRows borl prettyState prettyActionIdx (\_ x -> return x) (borl ^. proxies . v)
      return $ docHead $$ algDocRho prettyRhoVal $$ text "V" $+$ vcat prV
    AlgDQN {} -> do
      prR1 <- prettyTableRows borl prettyState prettyActionIdx (\_ x -> return x) (borl ^. proxies . r1)
      return $ docHead $$ algDocRho prettyRhoVal $$ text "Q" $+$ vcat prR1
    AlgDQNAvgRewardFree {}
      -- prR1 <- prettyTableRows borl prettyAction prettyActionIdx (\_ x -> return x) (borl ^. proxies . r1)
     -> do
      prR0R1 <- prBoolTblsStateAction t2 (text "V+e with gamma0" $$ nest nestCols (text "V+e with gamma1")) (borl ^. proxies . r0) (borl ^. proxies . r1)
      return $ docHead $$ algDocRho prettyRhoVal $$ prR0R1
  where
    subtr (k, v1) (_, v2) = (k, v1 - v2)
    prettyState = mkPrettyState mStInverse
    prettyActionIdx aIdx = text (T.unpack $ maybe "unkown" (actionName . snd) (find ((== aIdx `mod` length (borl ^. actionList)) . fst) (borl ^. actionList)))

mkPrettyState :: Show st => Maybe (NetInputWoAction -> Maybe st) -> [Double] -> Maybe (Maybe st, String)
mkPrettyState mStInverse netinp =
  case mStInverse of
    Nothing  -> Just (Nothing, show $ map showFloat netinp)
    Just inv -> (Just &&& show) <$> inv netinp


prettyBORLHead ::  (MonadBorl' m, Show s) => Bool -> Maybe (NetInputWoAction -> Maybe s) -> BORL s -> m Doc
prettyBORLHead printRho mInverseSt = prettyBORLHead' printRho (mkPrettyState mInverseSt)


prettyBORLHead' :: (MonadBorl' m, Show s) => Bool -> ([Double] -> Maybe (Maybe s, String)) -> BORL s -> m Doc
prettyBORLHead' printRho prettyStateFun borl = do
  let prettyState st = fromMaybe ("unkown state: " ++ show st) (snd <$> prettyStateFun st)
      prettyActionIdx aIdx = text (T.unpack $ maybe "unkown" (actionName . snd) (find ((== aIdx `mod` length (borl ^. actionList)) . fst) (borl ^. actionList)))
  let algDoc doc
        | isAlgBorl (borl ^. algorithm) = doc
        | otherwise = empty
  let prettyRhoVal =
        case borl ^. proxies . rho of
          Scalar val -> text "Rho" <> colon $$ nest nestCols (printFloatWith 8 val)
          _          -> empty
  return $ text "\n" $+$ text "Current state" <> colon $$ nest nestCols (text (show $ borl ^. s)) $+$ text "Period" <> colon $$ nest nestCols (int $ borl ^. t) $+$ text "Alpha" <> colon $$
    nest nestCols (printFloatWith 8 $ params' ^. alpha) $+$
    algDoc (text "Beta" <> colon $$ nest nestCols (printFloatWith 8 $ params' ^. beta)) $+$
    algDoc (text "Delta" <> colon $$ nest nestCols (printFloatWith 8 $ params' ^. delta)) $+$
    text "Gamma" <>
    colon $$
    nest nestCols (printFloatWith 8 $ params' ^. gamma) $+$
    text "Epsilon" <>
    colon $$
    nest nestCols (printFloatWith 8 $ params' ^. epsilon) $+$
    text "Exploration" <>
    colon $$
    nest nestCols (printFloatWith 8 $ params' ^. exploration) $+$
    text "Learn From Random Actions until Expl. hits" <>
    colon $$
    nest nestCols (printFloatWith 8 $ params' ^. learnRandomAbove) $+$
    nnTargetUpdate $+$
    nnBatchSize $+$
    nnReplMemSize $+$
    nnLearningParams $+$
    text "Algorithm" <>
    colon $$
    nest nestCols (prettyAlgorithm borl prettyState prettyActionIdx (borl ^. algorithm)) $+$
    algDoc (text "Zeta (for forcing V instead of W)" <> colon $$ nest nestCols (printFloatWith 8 $ params' ^. zeta)) $+$
    algDoc (text "Xi (ratio of W error forcing to V)" <> colon $$ nest nestCols (printFloatWith 8 $ params' ^. xi)) $+$
    (case borl ^. algorithm of
       AlgBORL {} -> text "Scaling (V,W,R0,R1) by V config" <> colon $$ nest nestCols scalingText
       AlgBORLVOnly {} -> text "Scaling BorlVOnly by V config" <> colon $$ nest nestCols scalingTextBorlVOnly
       AlgDQN {} -> text "Scaling (R1) by R1 Config" <> colon $$ nest nestCols scalingTextDqn
       AlgDQNAvgRewardFree {} -> text "Scaling (R0,R1) by R1 Config" <> colon $$ nest nestCols scalingTextAvgRewardFreeDqn) $+$
    algDoc
      (text "Psi Rho/Psi V/Psi W/Psi W2" <> colon $$
       nest nestCols (text (show (printFloatWith 8 $ borl ^. psis . _1, printFloatWith 8 $ borl ^. psis . _2, printFloatWith 8 $ borl ^. psis . _3, printFloatWith 8 $ borl ^. psis . _4)))) $+$
    (if printRho
       then prettyRhoVal
       else empty)
  where
    params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
    scalingText =
      case borl ^. proxies . v of
        P.Table {} -> text "Tabular representation (no scaling needed)"
        px         -> textNNConf (px ^?! proxyNNConfig)
      where
        textNNConf conf =
          text
            (show
               ( (printFloatWith 8 $ conf ^. scaleParameters . scaleMinVValue, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxVValue)
               , (printFloatWith 8 $ conf ^. scaleParameters . scaleMinWValue, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxWValue)
               , (printFloatWith 8 $ conf ^. scaleParameters . scaleMinR0Value, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxR0Value)
               , (printFloatWith 8 $ conf ^. scaleParameters . scaleMinR1Value, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxR1Value)))
    scalingTextDqn =
      case borl ^. proxies . r1 of
        P.Table {} -> text "Tabular representation (no scaling needed)"
        px         -> textNNConf (px ^?! proxyNNConfig)
      where
        textNNConf conf = text (show (printFloatWith 8 $ conf ^. scaleParameters . scaleMinR1Value, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxR1Value))
    scalingTextAvgRewardFreeDqn =
      case borl ^. proxies . r1 of
        P.Table {} -> text "Tabular representation (no scaling needed)"
        px         -> textNNConf (px ^?! proxyNNConfig)
      where
        textNNConf conf = text (show ((printFloatWith 8 $ conf ^. scaleParameters . scaleMinR0Value, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxR0Value),
                                      (printFloatWith 8 $ conf ^. scaleParameters . scaleMinR1Value, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxR1Value)))
    scalingTextBorlVOnly =
      case borl ^. proxies . v of
        P.Table {} -> text "Tabular representation (no scaling needed)"
        px         -> textNNConf (px ^?! proxyNNConfig)
      where
        textNNConf conf = text (show (printFloatWith 8 $ conf ^. scaleParameters . scaleMinVValue, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxVValue))
    nnTargetUpdate =
      case borl ^. proxies . v of
        P.Table {} -> empty
        px         -> textTargetUpdate (px ^?! proxyNNConfig)
      where
        textTargetUpdate conf = text "NN Target Replacment Interval" <> colon $$ nest nestCols (int $ conf ^. updateTargetInterval)
    nnBatchSize =
      case borl ^. proxies . v of
        P.Table {} -> empty
        px         -> textNNConf (px ^?! proxyNNConfig)
      where
        textNNConf conf = text "NN Batchsize" <> colon $$ nest nestCols (int $ conf ^. trainBatchSize)
    nnReplMemSize =
      case borl ^. proxies . v of
        P.Table {} -> empty
        px         -> textNNConf (px ^?! proxyNNConfig)
      where
        textNNConf conf = text "NN Replay Memory size" <> colon $$ nest nestCols (int $ conf ^. replayMemoryMaxSize)
    nnLearningParams =
      case borl ^. proxies . v of
        P.Table {} -> empty
        P.Grenade _ _ _ _ conf _ -> textGrenadeConf conf
        P.TensorflowProxy {} -> text "NN Learning Rate/Momentum/L2" <> colon $$ nest nestCols (text "Specified in tensorflow model")
        P.CombinedProxy P.TensorflowProxy {} _ _ -> text "NN Learning Rate/Momentum/L2" <> colon $$ nest nestCols (text "Specified in tensorflow model")
        P.CombinedProxy (P.Grenade _ _ _ _ conf _) _ _ -> textGrenadeConf conf
        _ -> error "nnLearningParams in Pretty.hs"
      where
        textGrenadeConf conf =
          let LearningParameters l0 m0 l20 = conf ^. grenadeLearningParams
              dec = decaySetup (conf ^. grenadeLearningParamsDecay) (borl ^. t)
              LearningParameters l m l2 = LearningParameters (dec l0) (dec m0) (dec l20)
           in text "NN Learning Rate/Momentum/L2" <> colon $$ nest nestCols (text (show (printFloatWith 8 l, printFloatWith 8 m, printFloatWith 8 l2)))

-- setPrettyPrintElems :: [NetInput] -> BORL s -> BORL s
-- setPrettyPrintElems xs borl = foldl' (\b p -> set (proxies . p . proxyNNConfig . prettyPrintElems) xs b) borl [rhoMinimum, rho, psiV, v, psiW, w, r0, r1]


prettyBORL :: (Ord s, Show s) => BORL s -> IO Doc
prettyBORL = prettyBORLWithStInverse Nothing

prettyBORLM :: (MonadBorl' m, Ord s, Show s) => BORL s -> m Doc
prettyBORLM = prettyBORLTables Nothing True True True

prettyBORLMWithStateInverse :: (MonadBorl' m, Ord s, Show s) => Maybe (NetInputWoAction -> Maybe s) -> BORL s -> m Doc
prettyBORLMWithStateInverse mStInverse = prettyBORLTables mStInverse True True True


prettyBORLWithStInverse :: (Ord s, Show s) => Maybe (NetInputWoAction -> Maybe s) -> BORL s -> IO Doc
prettyBORLWithStInverse mStInverse borl =
  case find isTensorflowProxy (allProxies $ borl ^. proxies) of
    Nothing -> prettyBORLTables mStInverse True True True borl
    Just _ ->
      runMonadBorlTF $ do
        restoreTensorflowModels True borl
        prettyBORLTables mStInverse True True True borl
  where
    isTensorflowProxy P.TensorflowProxy {} = True
    isTensorflowProxy _                    = False

instance (Ord s, Show s) => Show (BORL s) where
  show borl = renderStyle wideStyle $ unsafePerformIO $ prettyBORL borl
