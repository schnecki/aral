{-# LANGUAGE CPP               #-}
{-# LANGUAGE OverloadedStrings #-}
module ML.BORL.Pretty
    ( prettyTable
    , prettyBORL
    , prettyBORLHead
    , prettyBORLTables
    , wideStyle
    , setPrettyPrintElems
    ) where


import           ML.BORL.Action
import           ML.BORL.Algorithm
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import qualified ML.BORL.Proxy         as P
import           ML.BORL.Proxy.Ops     (lookupNeuralNetwork, mkNNList)
import           ML.BORL.Proxy.Type
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Arrow         (first, second, (&&&))
import           Control.Lens
import           Control.Monad         (when)
import           Data.Function         (on)
import           Data.List             (find, foldl', sort, sortBy)
import qualified Data.Map.Strict       as M
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

wideStyle :: Style
wideStyle = Style { lineLength = 200, ribbonsPerLine = 1.5, mode = PageMode }

printFloat :: Double -> Doc
printFloat x = text $ printf ("%." ++ show commas ++ "f") x

printFloatWith :: Int -> Double -> Doc
printFloatWith commas x = text $ printf ("%." ++ show commas ++ "f") x


prettyTable :: (MonadBorl' m, Show k, Eq k, Ord k, Ord k', Show k') => BORL k -> (NetInputWoAction -> k') -> (ActionIndex -> Doc) -> P.Proxy -> m Doc
prettyTable borl prettyKey prettyIdx p = vcat <$> prettyTableRows borl prettyKey prettyIdx (\_ v -> return v) p

prettyTableRows ::
     (MonadBorl' m, Show k, Ord k, Eq k, Ord k', Show k')
  => BORL k
  -> (NetInputWoAction -> k')
  -> (ActionIndex -> Doc)
  -> (([Double], ActionIndex) -> Double -> m Double)
  -> P.Proxy
  -> m [Doc]
prettyTableRows borl prettyAction prettyActionIdx modifier p =
  case p of
    P.Table m _ ->
      let mkAct idx = actionName $ snd $ (borl ^. actionList) !! idx
          mkInput k = text (filter (/= '"') $ show $ map (\x -> if x < 0 then printFloat x else "+" <> printFloat x) k)
      in mapM (\((k,idx),val) -> modifier (k,idx) val >>= \v -> return ( mkInput k <> text (T.unpack $ mkAct idx) <> colon <+> printFloat v)) $
      sortBy (compare `on`  fst.fst) $ M.toList m
    pr -> do
      mtrue <- mkListFromNeuralNetwork borl prettyAction prettyActionIdx True pr
      let printFun (kDoc, (valT, valW)) = kDoc <> colon <+> printFloat valT <+> text "  " <+> printFloat valW
          unfoldActs = concatMap (\(f,(ts,ws)) -> zipWith (\(nr,t) (_,w) -> (f nr, (t, w))) ts ws)
      return $ map printFun (unfoldActs mtrue)


prettyActionEntry :: (Show k') => (k -> k') -> (ActionIndex -> Doc) -> k -> ActionIndex -> Doc
prettyActionEntry pAct pActIdx act actIdx = text (show $ pAct act) <> colon <+> pActIdx actIdx


mkListFromNeuralNetwork ::
     (MonadBorl' m, Show k, Ord k, Eq k, Show k')
  => BORL k
  -> (NetInputWoAction -> k')
  -> (ActionIndex -> Doc)
  -> Bool
  -> P.Proxy
  -> m [(ActionIndex -> Doc, ([(ActionIndex, Double)], [(ActionIndex, Double)]))]
mkListFromNeuralNetwork borl prettyAction prettyActionIdx scaled pr = do
  nnList <- mkNNList borl scaled pr
  if borl ^. t <= pr ^?! proxyNNConfig . replayMemoryMaxSize
    then do
    let inp = map fst tbl
    let tableVals = tbl
    workerVals <- mapM (\x -> lookupNeuralNetwork Worker x pr) inp
    return $ finalize $ zipWith (\((feat,actIdx), tblV) workerV -> (feat, ([(actIdx, tblV)], [(actIdx, workerV)]))  ) tbl workerVals

    else finalize <$> mkNNList borl scaled pr
  where
    finalize = map (first $ prettyActionEntry prettyAction prettyActionIdx)
    tbl =
      case pr of
        P.Table t _                     -> M.toList t
        P.Grenade _ _ p _ cfg _         -> M.toList p
        P.TensorflowProxy _ _ p _ cfg _ -> M.toList p

prettyTablesState :: (MonadBorl' m, Show k, Ord k, Ord k', Show k') => BORL k -> (NetInputWoAction -> k') -> (ActionIndex -> Doc) -> P.Proxy -> (NetInputWoAction -> k') -> P.Proxy -> m Doc
prettyTablesState borl p1 pIdx m1 p2 m2 = do
  rows1 <- prettyTableRows borl p1 pIdx (\_ v -> return v) (if fromTable then tbl m1 else m1)
  rows2 <- prettyTableRows borl p2 pIdx (\_ v -> return v) (if fromTable then tbl m2 else m2)
  return $ vcat $ zipWith (\x y -> x $$ nest 40 y) rows1 rows2
  where fromTable = period < fromIntegral memSize
        period = borl ^. t
        memSize = case m1 of
          P.Table{}                       -> -1
          P.Grenade _ _ _ _ cfg _         -> cfg ^?! replayMemoryMaxSize
          P.TensorflowProxy _ _ _ _ cfg _ -> cfg ^?! replayMemoryMaxSize
        tbl px = case px of
          p@P.Table{}                     -> p
          P.Grenade _ _ p _ cfg _         -> P.Table p 0
          P.TensorflowProxy _ _ p _ cfg _ -> P.Table p 0

prettyAlgorithm ::  (Show k') => BORL s -> (NetInputWoAction -> k') -> (ActionIndex -> Doc) -> Algorithm s -> Doc
prettyAlgorithm borl prettyState prettyAction (AlgBORL ga0 ga1 avgRewType stValHand vPlusPsiV mRefState) =
  text "BORL with gammas " <+>
  text (show (ga0, ga1)) <> text ";" <+>
  prettyAvgRewardType avgRewType <+>
  text "for rho" <> text ";" <+>
  prettyStateValueHandling stValHand <+>
  text "Deciding on" <+>
  text
    (if vPlusPsiV
       then "V + PsiV"
       else "V") <+>
  prettyRefState borl prettyState prettyAction mRefState
prettyAlgorithm _ _ _ (AlgDQN ga1)      = text "DQN with gamma" <+> text (show ga1)
prettyAlgorithm borl _ _ (AlgDQNAvgRewardFree ga0 ga1 avgRewType)      = text "Average reward freed DQN with gammas" <+> text (show (ga0, ga1)) <+> ". Rho by" <+> prettyAvgRewardType avgRewType
  where decay = 0.5 ** (fromIntegral (borl ^. t) / 100000)
        ga1Diff = 1 - ga1
        ga1' = ga1 + ga1Diff - ga1Diff * decay

prettyAlgorithm borl prettyState prettyAction (AlgBORLVOnly avgRewType mRefState)      = text "BORL with V ONLY" <> text ";" <+> prettyAvgRewardType avgRewType <> prettyRefState borl prettyState prettyAction mRefState

prettyRefState :: (Show a) => BORL s -> ([Double] -> a) -> (t -> Doc) -> Maybe (s, t) -> Doc
prettyRefState _ _ _ Nothing = mempty
prettyRefState borl prettyState prettyAction (Just (st,aNr)) = ";" <+>  "Ref state: " <> text (show $ prettyState $ (borl ^. featureExtractor) st) <> " - " <> prettyAction aNr

prettyStateValueHandling :: StateValueHandling -> Doc
prettyStateValueHandling Normal = empty
prettyStateValueHandling (DivideValuesAfterGrowth nr max) = text "Divide values after growth " <> parens (int nr <> text "," <+> int max) <> text ";"

prettyAvgRewardType :: AvgReward -> Doc
prettyAvgRewardType (ByMovAvg nr)          = "moving average" <> parens (int nr)
prettyAvgRewardType ByReward               = "reward"
prettyAvgRewardType ByStateValues          = "state values"
prettyAvgRewardType (ByStateValuesAndReward ratio) = printFloat ratio <> "*state values + " <> printFloat (1-ratio) <> "*reward"
prettyAvgRewardType (Fixed x)              = "fixed value of " <> double x


prettyBORLTables :: (MonadBorl' m, Ord s, Show s) => Bool -> Bool -> Bool -> BORL s -> m Doc
prettyBORLTables t1 t2 t3 borl = do
  let algDoc doc
        | isAlgBorl (borl ^. algorithm) = doc
        | otherwise = empty
      algDocRho doc =
        case borl ^. algorithm of
          AlgDQN {} -> mempty
          _         -> doc
  let prBoolTblsStateAction True h m1 m2 = (h $+$) <$> prettyTablesState borl prettyAction prettyActionIdx m1 prettyAction m2
      prBoolTblsStateAction False _ _ _ = return empty
  let addPsiV k v =
        case borl ^. proxies . psiV of
          P.Table m def -> return $ M.findWithDefault def k m
          px ->
            let config = px ^?! proxyNNConfig
             in if borl ^. t <= config ^. replayMemoryMaxSize && (config ^. trainBatchSize) /= 1
                  then return $ M.findWithDefault 0 k (px ^. proxyNNStartup)
                  else do
                    vPsi <- P.lookupNeuralNetworkUnscaled P.Worker k (borl ^. proxies . psiV)
                    return (v + vPsi)
  vPlusPsiV <- prettyTableRows borl prettyAction prettyActionIdx addPsiV (borl ^. proxies . v)
  prettyRhoVal <-
    case borl ^. proxies . rho of
      Scalar val -> return $ text "Rho" <> colon $$ nest 45 (printFloatWith 8 val)
      m -> do
        prAct <- prettyTable borl prettyAction prettyActionIdx m
        return $ text "Rho" $+$ prAct
  docHead <- prettyBORLHead False borl
  case borl ^. algorithm of
    AlgBORL {} -> do
      prVs <- prBoolTblsStateAction t1 (text "V" $$ nest 40 (text "PsiV")) (borl ^. proxies . v) (borl ^. proxies . psiV)
      prWs <- prBoolTblsStateAction t1 (text "W" $$ nest 40 (text "PsiW")) (borl ^. proxies . w) (borl ^. proxies . psiW)
      prW2s <- prBoolTblsStateAction t1 (text "W2" $$ nest 40 (text "PsiW2")) (borl ^. proxies . w2) (borl ^. proxies . psiW2)
      prR0R1 <- prBoolTblsStateAction t2 (text "R0" $$ nest 40 (text "R1")) (borl ^. proxies . r0) (borl ^. proxies . r1)
      return $ docHead $$ algDocRho prettyRhoVal $$ -- prVW $+$ prR0R1 $+$ psis $+$ prWW2
        prVs $+$
        prWs $+$
        prW2s $+$
        prR0R1
    AlgBORLVOnly {} -> do
      prV <- prettyTableRows borl prettyAction prettyActionIdx (\_ x -> return x) (borl ^. proxies . v)
      return $ docHead $$ algDocRho prettyRhoVal $$ text "V" $+$ vcat prV
    AlgDQN {} -> do
      prR1 <- prettyTableRows borl prettyAction prettyActionIdx (\_ x -> return x) (borl ^. proxies . r1)
      return $ docHead $$ algDocRho prettyRhoVal $$ text "Q" $+$ vcat prR1
    AlgDQNAvgRewardFree {} -> do
      -- prR1 <- prettyTableRows borl prettyAction prettyActionIdx (\_ x -> return x) (borl ^. proxies . r1)
      prR0R1 <- prBoolTblsStateAction t2 (text "V+e with gamma0" $$ nest 40 (text "V+e with gamma1")) (borl ^. proxies . r0) (borl ^. proxies . r1)
      return $ docHead $$ algDocRho prettyRhoVal $$ prR0R1
  where
    subtr (k, v1) (_, v2) = (k, v1 - v2)
    prettyAction st = st
    prettyActionIdx aIdx = text (T.unpack $ maybe "unkown" (actionName . snd) (find ((== aIdx) . fst) (borl ^. actionList)))


prettyBORLHead :: (MonadBorl' m, Show s) => Bool -> BORL s -> m Doc
prettyBORLHead printRho borl = do
  let prettyAction st = st
      prettyActionIdx aIdx = text (T.unpack $ maybe "unkown" (actionName . snd) (find ((== aIdx) . fst) (borl ^. actionList)))

  let algDoc doc
        | isAlgBorl (borl ^. algorithm) = doc
        | otherwise = empty
  let prettyRhoVal = case borl ^. proxies . rho of
        Scalar val -> text "Rho" <> colon $$ nest 45 (printFloatWith 8 val)
        _          -> empty
  return $ text "\n" $+$ text "Current state" <> colon $$ nest 45 (text (show $ borl ^. s)) $+$ text "Period" <> colon $$ nest 45 (int $ borl ^. t) $+$
    text "Alpha" <> colon $$ nest 45 (printFloatWith 8 $ params' ^. alpha) $+$
    algDoc (text "Beta" <> colon $$ nest 45 (printFloatWith 8 $ params' ^. beta)) $+$
    algDoc (text "Delta" <> colon $$ nest 45 (printFloatWith 8 $ params' ^. delta)) $+$
    text "Gamma" <>
    colon $$
    nest 45 (printFloatWith 8 $ params' ^. gamma) $+$
    text "Epsilon" <>
    colon $$
    nest 45 (printFloatWith 8 $ params' ^. epsilon) $+$
    text "Exploration" <>
    colon $$
    nest 45 (printFloatWith 8 $ params' ^. exploration) $+$
    text "Learn From Random Actions until Expl. hits" <> colon $$ nest 45 (printFloatWith 8 $ params' ^. learnRandomAbove) $+$
    nnBatchSize $+$
    nnReplMemSize $+$
    nnLearningParams $+$
    text "Algorithm" <>
    colon $$
    nest 45 (prettyAlgorithm borl prettyAction prettyActionIdx (borl ^. algorithm)) $+$
    algDoc (text "Zeta (for forcing V instead of W)" <> colon $$ nest 45 (printFloatWith 8 $ params' ^. zeta)) $+$
    algDoc (text "Xi (ratio of W error forcing to V)" <> colon $$ nest 45 (printFloatWith 8 $ params' ^. xi)) $+$
    (case borl ^. algorithm of
       AlgBORL{} -> text "Scaling (V,W,R0,R1) by V config" <> colon $$ nest 45 scalingText
       AlgBORLVOnly{} -> text "Scaling BorlVOnly by V config" <> colon $$ nest 45 scalingTextBorlVOnly
       AlgDQN{} -> text "Scaling R1 by V Config" <> colon $$ nest 45 scalingTextDqn
       AlgDQNAvgRewardFree{} -> text "Scaling R1 by V Config" <> colon $$ nest 45 scalingTextDqn
    ) $+$
    algDoc (text "Psi Rho/Psi V/Psi W" <> colon $$ nest 45 (text (show (printFloatWith 8 $ borl ^. psis . _1, printFloatWith 8 $ borl ^. psis . _2, printFloatWith 8 $ borl ^. psis . _3)))) $+$
    (if printRho then prettyRhoVal else empty)
  where
    params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
    scalingText =
      case borl ^. proxies . v of
        P.Table {} -> text "Tabular representation (no scaling needed)"
        P.Grenade _ _ _ _ conf _ ->
          text
            (show
               ( (printFloatWith 8 $ conf ^. scaleParameters . scaleMinVValue, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxVValue)
               , (printFloatWith 8 $ conf ^. scaleParameters . scaleMinWValue, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxWValue)
               , (printFloatWith 8 $ conf ^. scaleParameters . scaleMinR0Value, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxR0Value)
               , (printFloatWith 8 $ conf ^. scaleParameters . scaleMinR1Value, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxR1Value)))
        P.TensorflowProxy _ _ _ _ conf _ ->
          text
            (show
               ( (printFloatWith 8 $ conf ^. scaleParameters . scaleMinVValue, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxVValue)
               , (printFloatWith 8 $ conf ^. scaleParameters . scaleMinWValue, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxWValue)
               , (printFloatWith 8 $ conf ^. scaleParameters . scaleMinR0Value, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxR0Value)
               , (printFloatWith 8 $ conf ^. scaleParameters . scaleMinR1Value, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxR1Value)))
    scalingTextDqn =
      case borl ^. proxies . v of
        P.Table {} -> text "Tabular representation (no scaling needed)"
        P.Grenade _ _ _ _ conf _ -> text (show (printFloatWith 8 $ conf ^. scaleParameters . scaleMinR1Value, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxR1Value))
        P.TensorflowProxy _ _ _ _ conf _ -> text (show (printFloatWith 8 $ conf ^. scaleParameters . scaleMinR1Value, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxR1Value))
    scalingTextBorlVOnly =
      case borl ^. proxies . v of
        P.Table {} -> text "Tabular representation (no scaling needed)"
        P.Grenade _ _ _ _ conf _ -> text (show (printFloatWith 8 $ conf ^. scaleParameters . scaleMinVValue, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxVValue))
        P.TensorflowProxy _ _ _ _ conf _ -> text (show (printFloatWith 8 $ conf ^. scaleParameters . scaleMinVValue, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxVValue))
    nnBatchSize =
      case borl ^. proxies . v of
        P.Table {} -> empty
        P.Grenade _ _ _ _ conf _ -> text "NN Batchsize" <> colon $$ nest 45 (int $ conf ^. trainBatchSize)
        P.TensorflowProxy _ _ _ _ conf _ -> text "NN Batchsize" <> colon $$ nest 45 (int $ conf ^. trainBatchSize)
    nnReplMemSize =
      case borl ^. proxies . v of
        P.Table {} -> empty
        P.Grenade _ _ _ _ conf _ -> text "NN Replay Memory size" <> colon $$ nest 45 (int $ conf ^. replayMemoryMaxSize)
        P.TensorflowProxy _ _ _ _ conf _ -> text "NN Replay Memory size" <> colon $$ nest 45 (int $ conf ^. replayMemoryMaxSize)
    nnLearningParams =
      case borl ^. proxies . v of
        P.Table {} -> empty
        P.Grenade _ _ _ _ conf _ ->
          let LearningParameters l m l2 = conf ^. grenadeLearningParams
           in text "NN Learning Rate/Momentum/L2" <> colon $$ nest 45 (text (show (printFloatWith 8 l, printFloatWith 8 m, printFloatWith 8 l2)))
        P.TensorflowProxy _ _ _ _ conf _ ->
          let LearningParameters l m l2 = conf ^. grenadeLearningParams
           in text "NN Learning Rate/Momentum/L2" <> colon $$ nest 45 (text "Specified in tensorflow model")

setPrettyPrintElems :: [NetInput] -> BORL s -> BORL s
setPrettyPrintElems xs borl = foldl' (\b p -> set (proxies . p . proxyNNConfig . prettyPrintElems) xs b) borl [rhoMinimum, rho, psiV, v, psiW, w, r0, r1]


prettyBORL :: (Ord s, Show s) => BORL s -> IO Doc
prettyBORL borl =
  case find isTensorflowProxy (allProxies $ borl ^. proxies) of
    Nothing -> runMonadBorlIO $ prettyBORLTables True True True borl
    Just _ ->
      runMonadBorlTF $ do
        buildModels
        reloadNets (borl ^. proxies . v)
        reloadNets (borl ^. proxies . w)
        reloadNets (borl ^. proxies . r0)
        reloadNets (borl ^. proxies . r1)
        prettyBORLTables True True True borl
  where
    reloadNets px =
      case px of
        P.TensorflowProxy netT netW _ _ _ _ -> restoreModelWithLastIO netT >> restoreModelWithLastIO netW
        _ -> return ()
    isTensorflowProxy P.TensorflowProxy {} = True
    isTensorflowProxy _                    = False
    buildModels =
      case find isTensorflowProxy (allProxies $ borl ^. proxies) of
        Just (P.TensorflowProxy netT _ _ _ _ _) -> buildTensorflowModel netT
        _                                       -> return ()

instance (Ord s, Show s) => Show (BORL s) where
  show borl = renderStyle wideStyle $ unsafePerformIO $ prettyBORL borl
