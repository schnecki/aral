{-# LANGUAGE CPP               #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections     #-}
module ML.BORL.Pretty
    ( prettyTable
    , prettyBORL
    , prettyBORLM
    , prettyBORLMWithStInverse
    , prettyBORLWithStInverse
    , prettyBORLHead
    , prettyBORLTables
    , wideStyle
    , showFloat
    , showFloatList
    ) where


import           Control.Arrow         (first, second, (&&&), (***))
import           Control.Lens
import           Control.Monad         (join, when)
import           Data.Function         (on)
import           Data.List             (find, foldl', intercalate, intersperse, sort,
                                        sortBy)
import qualified Data.Map.Strict       as M
import           Data.Maybe            (fromMaybe, isJust)
import qualified Data.Set              as S
import qualified Data.Text             as T
import qualified Data.Vector           as VB
import qualified Data.Vector.Storable  as V
import           Grenade
import           Prelude               hiding ((<>))
import           System.IO.Unsafe      (unsafePerformIO)
import           Text.PrettyPrint      as P
import           Text.Printf


import           ML.BORL.Action
import           ML.BORL.Algorithm
import           ML.BORL.Decay
import           ML.BORL.InftyVector
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import qualified ML.BORL.Proxy         as P
import           ML.BORL.Proxy.Ops     (LookupType (..), getMinMaxVal, lookupNeuralNetwork,
                                        mkNNList)
import           ML.BORL.Proxy.Proxies
import           ML.BORL.Proxy.Type
import           ML.BORL.SaveRestore
import           ML.BORL.Settings
import           ML.BORL.Type
import           ML.BORL.Types
import           ML.BORL.Workers.Type


import           Debug.Trace

commas :: Int
commas = 3

nestCols :: Int
nestCols = 75

wideStyle :: Style
wideStyle = Style { lineLength = 300, ribbonsPerLine = 200, mode = PageMode }

printFloat :: Float -> Doc
printFloat = text . showFloat

showFloat :: (PrintfArg n, Fractional n) => n -> String
showFloat = printf ("%+." ++ show commas ++ "f")

showFloatList :: (PrintfArg n, Fractional n) => [n] -> String
showFloatList xs = "[" ++ intercalate "," (map showFloat xs) ++ "]"

printFloatWith :: Int -> Float -> Doc
printFloatWith commas x = text $ printf ("%." ++ show commas ++ "f") x

type Modifier m = LookupType -> (NetInputWoAction, ActionIndex) -> Float -> m Float

noMod :: (Monad m) => Modifier m
noMod _ _ = return

modifierSubtract :: (MonadBorl' m) => BORL k -> P.Proxy -> Modifier m
modifierSubtract borl px lk k v0 = do
  vS <- P.lookupProxy (borl ^. t) lk k px
  return (v0 - vS)

prettyTable :: (MonadBorl' m, Show k, Ord k) => BORL k -> (NetInputWoAction -> Maybe (Maybe k, String)) -> (ActionIndex -> Doc) -> P.Proxy -> m Doc
prettyTable borl prettyKey prettyIdx p = vcat <$> prettyTableRows borl prettyKey prettyIdx noMod p

prettyTableRows :: (MonadBorl' m, Show k, Ord k) => BORL k -> (NetInputWoAction -> Maybe (Maybe k, String)) -> (ActionIndex -> Doc) -> Modifier m -> P.Proxy -> m [Doc]
prettyTableRows borl prettyState prettyActionIdx modifier p =
  case p of
    P.Table m _ ->
      let mkAct idx = actionName $ snd $ (borl ^. actionList) VB.! (idx `mod` length (borl ^. actionList))
          mkInput k = text (filter (/= '"') $ show $ map printFloat (V.toList k))
      in mapM (\((k,idx),val) -> modifier Target (k,idx) val >>= \v -> return (mkInput k <> text (T.unpack $ mkAct idx) <> colon <+> printFloat v)) $
      sortBy (compare `on`  fst.fst) $ M.toList m
    pr -> do
      mtrue <- mkListFromNeuralNetwork borl prettyState prettyActionIdx True modifier pr
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
  -> Modifier m
  -> P.Proxy
  -> m [(ActionIndex -> Doc, ([(ActionIndex, Float)], [(ActionIndex, Float)]))]
mkListFromNeuralNetwork borl prettyState prettyActionIdx scaled modifier pr = do
  let subPr
        | isCombinedProxy pr = pr ^?! proxySub
        | otherwise = pr
  finalize <$> (mkNNList borl scaled pr >>= mapM mkModification)
  where
    finalize = map (first $ prettyStateActionEntry borl prettyState prettyActionIdx)
    mkModification (inp, (xsT, xsW)) = do
      xsT' <- mapM (\(idx, val) -> (idx,) <$> modifier Target (inp, idx) val) xsT
      xsW' <- mapM (\(idx, val) -> (idx,) <$> modifier Worker (inp, idx) val) xsW
      return (inp, (xsT', xsW'))

prettyStateActionEntry :: BORL k -> (NetInputWoAction -> Maybe (Maybe k, String)) -> (ActionIndex -> Doc) -> NetInputWoAction -> ActionIndex -> Doc
prettyStateActionEntry borl pState pActIdx stInp actIdx = case pState stInp of
  Nothing               -> mempty
  Just (Just st, stRep) | actIdx < length bools && bools !! actIdx -> text stRep <> colon <+> pActIdx actIdx
                        | otherwise -> mempty
    where bools = take (length $ borl ^. actionList) $ V.toList $ (borl ^. actionFilter) st
  Just (Nothing, stRep) -> text stRep <> colon <+> pActIdx actIdx


prettyTablesState ::
     (MonadBorl' m, Show s, Ord s)
  => BORL s
  -> (NetInputWoAction -> Maybe (Maybe s, String))
  -> (ActionIndex -> Doc)
  -> P.Proxy
  -> (NetInputWoAction -> Maybe (Maybe s, String))
  -> Modifier m
  -> P.Proxy
  -> m Doc
prettyTablesState borl p1 pIdx m1 p2 modifier2 m2 = do
  rows1 <- prettyTableRows borl p1 pIdx noMod m1
  rows2 <- prettyTableRows borl p2 pIdx modifier2 m2
  return $ vcat $ zipWith (\x y -> x $$ nest nestCols y) rows1 rows2

prettyAlgorithm ::  BORL s -> (NetInputWoAction -> String) -> (ActionIndex -> Doc) -> Algorithm NetInputWoAction -> Doc
prettyAlgorithm borl prettyState prettyActionIdx (AlgBORL ga0 ga1 avgRewType mRefState) =
  text "BORL with gammas " <+>
  text (show (ga0, ga1)) <> text ";" <+>
  prettyAvgRewardType (borl ^. t) avgRewType <+>
  text "for rho" <> text ";" <+>
  prettyRefState prettyState prettyActionIdx mRefState
prettyAlgorithm _ _ _ (AlgDQN ga1 cmp)      = text "DQN with gamma" <+> text (show ga1) <> colon <+> prettyComparison cmp
prettyAlgorithm borl _ _ (AlgDQNAvgRewAdjusted ga0 ga1 avgRewType) =
  text "Average reward adjusted DQN with gammas" <+> text (show (ga0, ga1)) <> ". Rho by" <+> prettyAvgRewardType (borl ^. t) avgRewType
prettyAlgorithm borl prettyState prettyAction (AlgBORLVOnly avgRewType mRefState) =
  text "BORL with V ONLY" <> text ";" <+> prettyAvgRewardType (borl ^. t) avgRewType <> prettyRefState prettyState prettyAction mRefState

prettyComparison :: Comparison -> Doc
prettyComparison EpsilonSensitive = "optimising by epsilon-sensitive comparison"
prettyComparison Exact            = "optimising by exact comparison"


prettyRefState :: (Show a) => (NetInputWoAction -> a) -> (t -> Doc) -> Maybe (NetInputWoAction, t) -> Doc
prettyRefState _ _ Nothing = mempty
prettyRefState prettyState prettyAction (Just (stFeat,aNr)) = ";" <+>  "Ref state: " <> text (show $ prettyState stFeat) <> " - " <> prettyAction aNr

prettyAvgRewardType :: Period -> AvgReward -> Doc
prettyAvgRewardType _ (ByMovAvg nr)          = "moving average" <> parens (int nr)
prettyAvgRewardType _ ByReward               = "reward"
prettyAvgRewardType _ ByStateValues          = "state values"
prettyAvgRewardType period (ByStateValuesAndReward ratio decay) =
  printFloat ratio' <> "*state values + " <> printFloat (1 - ratio') <> "*reward" <+>
  parens (text "Period 0" <> colon <+> printFloat ratio <> "*state values + " <> printFloat (1 - ratio) <> "*reward")
  where
    ratio' = decaySetup decay period ratio
prettyAvgRewardType _ (Fixed x)              = "fixed value of " <> float x


prettyBORLTables :: (MonadBorl' m, Ord s, Show s) => Maybe (NetInputWoAction -> Maybe (Either String s)) -> Bool -> Bool -> Bool -> BORL s -> m Doc
prettyBORLTables mStInverse t1 t2 t3 borl = do
  let algDoc doc
        | isAlgBorl (borl ^. algorithm) = doc
        | otherwise = empty
      algDocRho doc =
        case borl ^. algorithm of
          AlgDQN {} -> mempty
          _         -> doc
  let prBoolTblsStateAction True h m1 m2 = (h $+$) <$> prettyTablesState borl prettyState prettyActionIdx m1 prettyState noMod m2
      prBoolTblsStateAction False _ _ _ = return empty
  prettyRhoVal <-
    case borl ^. proxies . rho of
      Scalar val -> return $ text "Rho" <> colon $$ nest nestCols (printFloatWith 8 val)
      m -> do
        prAct <- prettyTable borl prettyState prettyActionIdx m
        return $ text "Rho" $+$ prAct
  docHead <- prettyBORLHead' False prettyState borl
  case borl ^. algorithm of
    AlgBORL {} -> do
      prVs <- prBoolTblsStateAction t1 (text "V" $$ nest nestCols (text "PsiV")) (borl ^. proxies . v) (borl ^. proxies . psiV)
      prWs <- prBoolTblsStateAction t2 (text "W" $$ nest nestCols (text "PsiW")) (borl ^. proxies . w) (borl ^. proxies . psiW)
      prR0R1 <- prBoolTblsStateAction t3 (text "R0" $$ nest nestCols (text "R1")) (borl ^. proxies . r0) (borl ^. proxies . r1)
      return $ docHead $$ algDocRho prettyRhoVal $$ prVs $+$ prWs $+$ prR0R1
    AlgBORLVOnly {} -> do
      prV <- prettyTableRows borl prettyState prettyActionIdx noMod (borl ^. proxies . v)
      return $ docHead $$ algDocRho prettyRhoVal $$ text "V" $+$ vcat prV
    AlgDQN {} -> do
      prR1 <- prettyTableRows borl prettyState prettyActionIdx noMod (borl ^. proxies . r1)
      return $ docHead $$ algDocRho prettyRhoVal $$ text "Q" $+$ vcat prR1
    AlgDQNAvgRewAdjusted{} -> do
      prR0R1 <- prBoolTblsStateAction t1 (text "V+e with gamma0" $$ nest nestCols (text "V+e with gamma1")) (borl ^. proxies . r0) (borl ^. proxies . r1)
      return $ docHead $$ algDocRho prettyRhoVal $$ prR0R1
  where
    prettyState = mkPrettyState mStInverse
    prettyActionIdx aIdx = text (T.unpack $ maybe "unkown" (actionName . snd) (find ((== aIdx `mod` length (borl ^. actionList)) . fst) (borl ^. actionList)))

mkPrettyState :: Show st => Maybe (NetInputWoAction -> Maybe (Either String st)) -> NetInputWoAction -> Maybe (Maybe st, String)
mkPrettyState mStInverse netinp =
  case mStInverse of
    Nothing  -> Just (Nothing, showFloatList $ V.toList netinp)
    Just inv -> fromEither <$> inv netinp
  where fromEither (Left str) = (Nothing, str)
        fromEither (Right st) = (Just st, show st)

prettyBORLHead ::  (MonadBorl' m, Show s) => Bool -> Maybe (NetInputWoAction -> Maybe (Either String s)) -> BORL s -> m Doc
prettyBORLHead printRho mInverseSt = prettyBORLHead' printRho (mkPrettyState mInverseSt)


prettyBORLHead' :: (MonadBorl' m, Show s) => Bool -> (NetInputWoAction -> Maybe (Maybe s, String)) -> BORL s -> m Doc
prettyBORLHead' printRho prettyStateFun borl = do
  let prettyState st = maybe ("unkown state: " ++ show st) snd (prettyStateFun st)
      prettyActionIdx aIdx = text (T.unpack $ maybe "unkown" (actionName . snd) (find ((== aIdx `mod` length (borl ^. actionList)) . fst) (borl ^. actionList)))
  let algDoc doc
        | isAlgBorl (borl ^. algorithm) = doc
        | otherwise = empty
  let prettyRhoVal =
        case (borl ^. proxies . rho, borl ^. proxies . rhoMinimum) of
          (Scalar val, Scalar valRhoMin) -> text "Rho/RhoMinimum" <> colon $$ nest nestCols (printFloatWith 8 val <> text "/" <> printFloatWith 8 valRhoMin)
          _ -> empty
  let getExpSmthParam decayed p param
        | isANN = 1
        | decayed = params' ^. param
        | otherwise = borl ^. parameters . param
        where
          isANN = P.isNeuralNetwork px && borl ^. t >= px ^?! proxyNNConfig . replayMemoryMaxSize
          px = borl ^. proxies . p
  return $ text "\n" $+$ text "Current state" <> colon $$ nest nestCols (text (show $ borl ^. s) <+> "Exp. Smth Reward: " <> printFloat (borl ^. expSmoothedReward)) $+$
    vcat
      (map
         (\(WorkerState wId wSt _ _ rew) -> text "Current state Worker " <+> int wId <> colon $$ nest nestCols (text $ show wSt) <+> "Exp. Smth Reward: " <> printFloat rew)
         (borl ^. workers)) $+$
    text "Period" <>
    colon $$
    nest nestCols (int $ borl ^. t) $+$
    text "Alpha/AlphaRhoMin" <>
    colon $$
    nest nestCols (printFloatWith 8 (getExpSmthParam True rho alpha) <> text "/" <> printFloatWith 8 (getExpSmthParam True rhoMinimum alphaRhoMin)) <+>
    parens (text "Period 0" <> colon <+> printFloatWith 8 (getExpSmthParam False rho alpha) <> text "/" <> printFloatWith 8 (getExpSmthParam False rhoMinimum alphaRhoMin)) $+$
    algDoc
      (text "Beta" <> colon $$ nest nestCols (printFloatWith 8 $ getExpSmthParam True v beta) <+>
       parens (text "Period 0" <> colon <+> printFloatWith 8 (getExpSmthParam False v beta))) $+$
    algDoc
      (text "Delta" <> colon $$ nest nestCols (printFloatWith 8 $ getExpSmthParam True w delta) <+>
       parens (text "Period 0" <> colon <+> printFloatWith 8 (getExpSmthParam False w delta))) $+$
    (text "Gamma" <> colon $$ nest nestCols (printFloatWith 8 $ getExpSmthParam True r1 gamma)) <+>
    parens (text "Period 0" <> colon <+> printFloatWith 8 (getExpSmthParam False r1 gamma)) $+$
    text "Epsilon" <>
    colon $$
    nest nestCols (hcat $ intersperse (text ", ") $ toFiniteList $ printFloatWith 8 <$> params' ^. epsilon) <+>
    parens (text "Period 0" <> colon <+> hcat (intersperse (text ", ") $ toFiniteList $ printFloatWith 8 <$> params ^. epsilon)) <+>
    text "Strategy" <>
    colon <+>
    text (show $ borl ^. settings . explorationStrategy) $+$
    text "Exploration" <>
    colon $$
    nest nestCols (printFloatWith 8 $ params' ^. exploration) <+>
    parens (text "Period 0" <> colon <+> printFloatWith 8 (params ^. exploration)) $+$
    nnWorkers $+$
    algDoc
      (text "Learn From Random Actions until Expl. hits" <> colon $$ nest nestCols (printFloatWith 8 $ params' ^. learnRandomAbove) <+>
       parens (text "Period 0" <> colon <+> printFloatWith 8 (params ^. learnRandomAbove))) $+$
    text "Function Approximation (inferred by R1 Config)" <>
    colon $$
    nest nestCols (text $ prettyProxyType $ borl ^. proxies . r1) $+$
    nnTargetUpdate $+$
    nnBatchSize $+$
    nnNStep $+$
    nnReplMemSize $+$
    nnLearningParams $+$
    text "Algorithm" <>
    colon $$
    nest nestCols (prettyAlgorithm borl prettyState prettyActionIdx (borl ^. algorithm)) $+$
    algDoc
      (text "Zeta (for forcing V instead of W)" <> colon $$ nest nestCols (printFloatWith 8 $ params' ^. zeta) <+>
       parens (text "Period 0" <> colon <+> printFloatWith 8 (params ^. zeta))) $+$
    algDoc
      (text "Xi (ratio of W error forcing to V)" <> colon $$ nest nestCols (printFloatWith 8 $ params' ^. xi) <+>
       parens (text "Period 0" <> colon <+> printFloatWith 8 (params ^. xi))) $+$
    (case borl ^. algorithm of
       AlgBORL {} -> text "Scaling (V,W,R0,R1) by V config" <> colon $$ nest nestCols scalingText
       AlgBORLVOnly {} -> text "Scaling BorlVOnly by V config" <> colon $$ nest nestCols scalingTextBorlVOnly
       AlgDQN {} -> text "Scaling (R1) by R1 Config" <> colon $$ nest nestCols scalingTextDqn
       AlgDQNAvgRewAdjusted {} -> text "Scaling (R0,R1) by R1 Config" <> colon $$ nest nestCols scalingTextAvgRewardAdjustedDqn) $+$
    algDoc
      (text "Psi Rho/Psi V/Psi W" <> colon $$
       nest nestCols (text (show (printFloatWith 8 $ borl ^. psis . _1, printFloatWith 8 $ borl ^. psis . _2, printFloatWith 8 $ borl ^. psis . _3)))) $+$
    (if printRho
       then prettyRhoVal
       else empty)
  where
    params = borl ^. parameters
    params' = decayedParameters borl
    nnWorkers =
      case borl ^. proxies . r1 of
        P.Table {} -> mempty
        px ->
          text "Workers Minimum Exploration (Epsilon-Greedy)" <> colon $$ nest nestCols (text (showFloatList (borl ^. settings . workersMinExploration))) <+>
          maybe mempty (\(WorkerState _ _ ms _ _) -> text "Replay memories:" <+> textReplayMemoryType ms) (borl ^? workers . _head)
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
    scalingTextAvgRewardAdjustedDqn =
      case borl ^. proxies . r1 of
        P.Table {} -> text "Tabular representation (no scaling needed)"
        px         -> textNNConf (px ^?! proxyNNConfig)
      where
        textNNConf conf =
          text
            (show
               ( (printFloatWith 8 $ conf ^. scaleParameters . scaleMinR0Value, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxR0Value)
               , (printFloatWith 8 $ conf ^. scaleParameters . scaleMinR1Value, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxR1Value)))
    scalingTextBorlVOnly =
      case borl ^. proxies . v of
        P.Table {} -> text "Tabular representation (no scaling needed)"
        px         -> textNNConf (px ^?! proxyNNConfig)
      where
        textNNConf conf = text (show (printFloatWith 8 $ conf ^. scaleParameters . scaleMinVValue, printFloatWith 8 $ conf ^. scaleParameters . scaleMaxVValue))
    nnTargetUpdate =
      case borl ^. proxies . v of
        P.Table {} -> empty
        P.Grenade _ _ _ cfg _
          | cfg ^. grenadeSmoothTargetUpdate > 0 ->
            text "NN Smooth Target Update Rate" <> colon $$ nest nestCols (printFloatWith 8 $ fromRational $ cfg ^. grenadeSmoothTargetUpdate)
        px -> textTargetUpdate (px ^?! proxyNNConfig)
      where
        textTargetUpdate conf =
          text "NN Target Replacment Interval" <> colon $$ nest nestCols (int upTargetInterval) <+> parens (text "Maximum" <> colon <+> int (conf ^. updateTargetInterval))
          where
            upTargetInterval =
              max 1 $ round $ decaySetup (conf ^. updateTargetIntervalDecay) (borl ^. t - conf ^. replayMemoryMaxSize - 1) (fromIntegral $ conf ^. updateTargetInterval)
    nnBatchSize =
      case borl ^. proxies . v of
        P.Table {} -> empty
        px         -> textNNConf (px ^?! proxyNNConfig)
      where
        textNNConf conf = text "NN Batchsize" <> colon $$ nest nestCols (int $ conf ^. trainBatchSize)
    nnNStep =
      case borl ^. proxies . v of
        P.Table {} -> empty
        px         -> textNNConf (borl ^. settings)
      where
        textNNConf conf = text "NStep" <> colon $$ nest nestCols (int $ conf ^. nStep)
    nnReplMemSize =
      case borl ^. proxies . v of
        P.Table {} -> empty
        px         -> textNNConf (px ^?! proxyNNConfig)
      where
        textNNConf conf =
          text "NN Replay Memory size" <> colon $$ nest nestCols (int $ conf ^. replayMemoryMaxSize) <+>
          maybe mempty (brackets . textReplayMemoryType) (borl ^. proxies . replayMemory)
    textReplayMemoryType ReplayMemoriesUnified {} = text "unified replay memory"
    textReplayMemoryType mem@ReplayMemoriesPerActions {} = text "per actions each of size " <> int (replayMemoriesSubSize mem)
    nnLearningParams =
      case borl ^. proxies . v of
        P.Table {} -> empty
        P.Grenade _ _ _ conf _ -> textGrenadeConf conf (conf ^. grenadeLearningParams)
        P.TensorflowProxy _ _ _ conf _ -> textTensorflow conf (conf ^. grenadeLearningParams)
        P.CombinedProxy (P.TensorflowProxy _ _ _ conf _) _ _ -> textTensorflow conf (conf ^. grenadeLearningParams)
        P.CombinedProxy (P.Grenade _ _ _ conf _) _ _ -> textGrenadeConf conf (conf ^. grenadeLearningParams)
        _ -> error "nnLearningParams in Pretty.hs"
      where
        textGrenadeConf :: NNConfig -> Optimizer opt -> Doc
        textGrenadeConf conf (OptSGD rate momentum l2) =
          let dec = decaySetup (conf ^. learningParamsDecay) (borl ^. t)
              l = realToFrac $ dec $ realToFrac rate
           in text "NN Learning Rate/Momentum/L2" <> colon $$
              nest nestCols (text "SGD Optimizer with" <+> text (show (printFloatWith 8 (realToFrac l), printFloatWith 8 (realToFrac momentum), printFloatWith 8 (realToFrac l2))))
        textGrenadeConf conf (OptAdam alpha beta1 beta2 epsilon lambda) =
          let dec = decaySetup (conf ^. learningParamsDecay) (borl ^. t)
              l = realToFrac $ dec $ realToFrac alpha
           in text "NN Learning Rate/Momentum/L2" <> colon $$
              nest
                nestCols
                (text "Adam Optimizer with" <+>
                 text
                   (show
                      ( printFloatWith 8 (realToFrac l)
                      , printFloatWith 8 (realToFrac beta1)
                      , printFloatWith 8 (realToFrac beta2)
                      , printFloatWith 8 (realToFrac epsilon)
                      , printFloatWith 8 (realToFrac lambda))))
        textTensorflow :: NNConfig -> Optimizer opt -> Doc
        textTensorflow conf (OptSGD rate _ _) =
          let dec = decaySetup (conf ^. learningParamsDecay) (borl ^. t)
              l = dec $ realToFrac rate
           in text "NN Learning Rate" <> colon $$ nest nestCols (text (show (printFloatWith 8 l)))
        textTensorflow conf (OptAdam alpha _ _ _ _) =
          let dec = decaySetup (conf ^. learningParamsDecay) (borl ^. t)
              l = realToFrac $ dec $ realToFrac alpha
           in text "NN Learning Rate" <> colon $$ nest nestCols (text (show (printFloatWith 8 l)))

-- setPrettyPrintElems :: [NetInput] -> BORL s -> BORL s
-- setPrettyPrintElems xs borl = foldl' (\b p -> set (proxies . p . proxyNNConfig . prettyPrintElems) xs b) borl [rhoMinimum, rho, psiV, v, psiW, w, r0, r1]


prettyBORL :: (Ord s, Show s) => BORL s -> IO Doc
prettyBORL = prettyBORLWithStInverse Nothing

prettyBORLM :: (MonadBorl' m, Ord s, Show s) => BORL s -> m Doc
prettyBORLM = prettyBORLTables Nothing True True True

prettyBORLMWithStInverse :: (MonadBorl' m, Ord s, Show s) => Maybe (NetInputWoAction -> Maybe (Either String s)) -> BORL s -> m Doc
prettyBORLMWithStInverse mStInverse = prettyBORLTables mStInverse True True True


prettyBORLWithStInverse :: (Ord s, Show s) => Maybe (NetInputWoAction -> Maybe (Either String s)) -> BORL s -> IO Doc
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
