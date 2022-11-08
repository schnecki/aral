{-# LANGUAGE CPP               #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections     #-}
module ML.ARAL.Pretty
    ( prettyTable
    , prettyARAL
    , prettyARALM
    , prettyARALMWithStInverse
    , prettyARALWithStInverse
    , prettyARALHead
    , prettyARALTables
    , wideStyle
    , showDouble
    , showDoubleList
    ) where


import           Control.Arrow                       (first, second, (&&&), (***))
import           Control.Lens
import           Control.Monad                       (join, when)
import           Control.Monad.IO.Class
import           Data.Function                       (on)
import           Data.List                           (find, foldl', intercalate, intersperse, nub, sort, sortBy)
import qualified Data.Map.Strict                     as M
import           Data.Maybe                          (fromMaybe, isJust, listToMaybe)
import qualified Data.Set                            as S
import qualified Data.Text                           as T
import qualified Data.Vector                         as VB
import qualified Data.Vector.Storable                as V
import           Grenade
import           Prelude                             hiding ((<>))
import           System.IO.Unsafe                    (unsafePerformIO)
import           Text.PrettyPrint                    as P
import           Text.Printf

import           ML.ARAL.Action
import           ML.ARAL.Algorithm
import           ML.ARAL.Decay
import           ML.ARAL.InftyVector
import           ML.ARAL.NeuralNetwork
import           ML.ARAL.NeuralNetwork.Normalisation
import           ML.ARAL.Parameters
import qualified ML.ARAL.Proxy                       as P
import           ML.ARAL.Proxy.Ops                   (LookupType (..), getMinMaxVal, lookupNeuralNetwork, mkNNList)
import           ML.ARAL.Proxy.Proxies
import           ML.ARAL.Proxy.Regression
import           ML.ARAL.Proxy.Type
import           ML.ARAL.Settings
import           ML.ARAL.Type
import           ML.ARAL.Types
import           ML.ARAL.Workers.Type


import           Debug.Trace

semicolon :: Doc
semicolon = char ';'

commas :: Int
commas = 3

nestCols :: Int
nestCols = 75

wideStyle :: Style
wideStyle = Style { lineLength = 300, ribbonsPerLine = 200, mode = PageMode }

printDouble :: Double -> Doc
printDouble = text . showDouble

printValue :: Value -> Doc
printValue = hcat . punctuate ", " . map printDouble . fromValue

printActionIndices :: (ActionIndex -> Doc) -> [ActionIndex] -> Doc
printActionIndices f = hcat . punctuate ", " . map f


showDouble :: (PrintfArg n, Fractional n) => n -> String
showDouble = printf ("%+." ++ show commas ++ "f")

showDoubleList :: (PrintfArg n, Fractional n) => [n] -> String
showDoubleList xs = "[" ++ intercalate "," (map showDouble xs) ++ "]"

printDoubleWith :: Int -> Double -> Doc
printDoubleWith commas x = text $ printf ("%." ++ show commas ++ "f") x

printDoubleListWith :: Int -> [Double] -> Doc
printDoubleListWith commas xs = hcat $ punctuate ", " $ map (text . printf ("%." ++ show commas ++ "f")) xs


type Modifier m = LookupType -> (NetInputWoAction, ActionIndex) -> Value -> m Value

noMod :: (Monad m) => Modifier m
noMod _ _ = return

modifierSubtract :: (MonadIO m) => ARAL s as -> P.Proxy -> Modifier m
modifierSubtract borl px lk k v0 = do
  vS <- P.lookupProxy MainAgent (borl ^. t) lk (second (VB.replicate agents) k) px
  return (v0 - vS)
  where agents = px ^?! proxyNrAgents

prettyTable :: (MonadIO m, Show s, Show as, Ord s) => ARAL s as -> (NetInputWoAction -> Maybe (Maybe s, String)) -> (ActionIndex -> Doc) -> P.Proxy -> m Doc
prettyTable borl prettyKey prettyIdx p = vcat <$> prettyTableRows borl prettyKey prettyIdx noMod p

prettyTableRows :: (MonadIO m, Show s, Show as, Ord s) => ARAL s as -> (NetInputWoAction -> Maybe (Maybe s, String)) -> (ActionIndex -> Doc) -> Modifier m -> P.Proxy -> m [Doc]
prettyTableRows borl prettyState prettyActionIdx modifier p =
  case p of
    P.Table m _ _ ->
      let mkAct idx = show $ (borl ^. actionList) VB.! (idx `mod` length (borl ^. actionList))
          mkInput k = maybe (text (filter (/= '"') $ show $ map printDouble (V.toList k))) (\(ms, st) -> text $ maybe st show ms) (prettyState k)
       in mapM (\((k, idx), val) -> modifier Target (k, idx) val >>= \v -> return (mkInput k <> comma <+> text (mkAct idx) <> colon <+> printValue v)) $
          sortBy (compare `on` fst . fst) $ map (\((st, a), v) -> ((st, a), AgentValue v)) (M.toList m)
    P.RegressionProxy layer@(RegressionLayer (low, high) welInp step regime) aNr ->
      let mkAct idx = show $ (borl ^. actionList) VB.! (idx `mod` length (borl ^. actionList))
          mkInput k = maybe (text (filter (/= '"') $ show $ map printDouble (V.toList k))) (\(ms, st) -> text $ maybe st show ms) (prettyState k)
          mkInputs :: VB.Vector RegressionNode -> [NetInputWoAction]
          mkInputs xs = nub $ concatMap (map (VB.convert . normaliseStateFeatureUnbounded welInp . obsInputValues) . filter ((>= step - 1) . obsPeriod) . concatMap VB.toList . M.elems . regNodeObservations) $ VB.toList xs
          inputs = mkInputs low ++ maybe [] mkInputs high
          -- inputs = nnCfg ^. prettyPrintElems
          inputActionValue = concatMap (\inp -> map (\aId -> ((inp, aId), V.singleton $ applyRegressionLayer (agentIndex MainAgent) layer aId inp)) [0..aNr-1]) inputs
       in
        fmap (++ [text "" $+$ prettyRegressionLayerNoObs layer, text "" $+$ text (show regime)]) $
           mapM (\((k, idx), val) -> modifier Target (k, idx) val >>= \v -> return (mkInput k <> comma <+> text (mkAct idx) <> colon <+> printValue v)) $
             sortBy (compare `on` fst . fst) $ map (\((st, a), v) -> ((st, a), AgentValue v)) inputActionValue
    pr -> do
      mtrue <- mkListFromNeuralNetwork borl prettyState prettyActionIdx True modifier pr
      let printFun (kDoc, (valT, valW))
            | isEmpty kDoc = []
            | otherwise = [kDoc <> colon <+> printValue valT <+> text "  " <+> printValue valW]
          unfoldActs = concatMap (\(f, (ts, ws)) -> zipWith (\(nr, t) (_, w) -> (f nr, (t, w))) ts ws)
      return $ concatMap printFun (unfoldActs mtrue)


mkListFromNeuralNetwork ::
     (MonadIO m)
  => ARAL s as
  -> (NetInputWoAction -> Maybe (Maybe s, String))
  -> (ActionIndex -> Doc)
  -> Bool
  -> Modifier m
  -> P.Proxy
  -> m [(ActionIndex -> Doc, ([(ActionIndex, Value)], [(ActionIndex, Value)]))]
mkListFromNeuralNetwork borl prettyState prettyActionIdx scaled modifier pr = finalize <$> (mkNNList borl scaled pr >>= mapM mkModification)
  where
    finalize = map (first (prettyStateActionEntry borl prettyState prettyActionIdx))
    mkModification (inp, (xsT, xsW)) = do
      xsT' <- mapM (\(idx, val) -> (idx, ) <$> modifier Target (inp, idx) val) xsT
      xsW' <- mapM (\(idx, val) -> (idx, ) <$> modifier Worker (inp, idx) val) xsW
      return (inp, (xsT', xsW'))

prettyStateActionEntry :: ARAL s as -> (NetInputWoAction -> Maybe (Maybe s, String)) -> (ActionIndex -> Doc) -> NetInputWoAction -> ActionIndex -> Doc
prettyStateActionEntry borl pState pActIdx stInp actIdx =
  case pState stInp of
    Nothing -> mempty
    Just (Just st, stRep) ->
      if all (== text "-") txts
        then mempty
        else text stRep <> colon <+> hcat (punctuate ", " txts)
      where txts = map agentText (borl ^. actionFilter $ st)
            agentText boolVec
              | actIdx >= V.length boolVec = error $ "filter function output length length does not match action Index: " ++ show (V.toList boolVec)
              | actIdx < V.length boolVec && boolVec V.! actIdx = pActIdx actIdx
              | otherwise = text "-"
    Just (Nothing, stRep) -> text stRep <> colon <+> pActIdx actIdx


-- prettyStateActionEntry :: ARAL s as -> (NetInputWoAction -> Maybe (Maybe s, String)) -> (ActionIndex -> Doc) -> AgentNumber -> NetInputWoAction -> ActionIndex -> Doc
-- prettyStateActionEntry borl pState pActIdx agentNr stInp actIdx = case pState stInp of
--   Nothing               -> mempty
--   Just (Just st, stRep) | and (zipWith (!!) bools actIdx) -> text stRep <> colon <+> printActionIndices pActIdx actIdx
--                         | otherwise -> mempty
--     where bools = map (take (length $ borl ^. actionList) . V.toList) $ (borl ^. actionFilter) st
--   Just (Nothing, stRep) -> text stRep <> colon <+> printActionIndices pActIdx actIdx


prettyTablesState ::
     (MonadIO m, Show s, Show as, Ord s)
  => ARAL s as
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

prettyAlgorithm ::  ARAL s as -> (NetInputWoAction -> String) -> (ActionIndex -> Doc) -> Algorithm NetInputWoAction -> Doc
prettyAlgorithm borl prettyState prettyActionIdx (AlgNBORL ga0 ga1 avgRewType mRefState) =
  text "ARAL with gammas " <+>
  text (show (ga0, ga1)) <> text ";" <+>
  prettyAvgRewardType (borl ^. t) avgRewType <+>
  text "for rho" <> text ";" <+>
  prettyRefState prettyState prettyActionIdx mRefState
prettyAlgorithm _ _ _ (AlgDQN ga1 cmp)      = text "DQN with gamma" <+> text (show ga1) <> colon <+> prettyComparison cmp
prettyAlgorithm borl _ _ (AlgARAL ga0 ga1 avgRewType) =
  text "Average reward adjusted DQN with gammas" <+> text (show (ga0, ga1)) <> ". Rho by" <+> prettyAvgRewardType (borl ^. t) avgRewType
prettyAlgorithm borl prettyState prettyAction (AlgARALVOnly avgRewType mRefState) =
  text "ARAL with V ONLY" <> text ";" <+> prettyAvgRewardType (borl ^. t) avgRewType <> prettyRefState prettyState prettyAction mRefState
prettyAlgorithm borl prettyState prettyAction AlgRLearning =
  text "R-Learning"

prettyComparison :: Comparison -> Doc
prettyComparison EpsilonSensitive = "optimising by epsilon-sensitive comparison"
prettyComparison Exact            = "optimising by exact comparison"


prettyRefState :: (Show a) => (NetInputWoAction -> a) -> (ActionIndex -> Doc) -> Maybe (NetInputWoAction, [ActionIndex]) -> Doc
prettyRefState _ _ Nothing                                  = mempty
prettyRefState prettyState prettyAction (Just (stFeat,aNr)) = ";" <+>  "Ref state: " <> text (show $ prettyState stFeat) <> " - " <> printActionIndices prettyAction aNr

prettyAvgRewardType :: Period -> AvgReward -> Doc
prettyAvgRewardType _ (ByMovAvg nr)          = "moving average" <> parens (int nr)
prettyAvgRewardType _ ByReward               = "reward"
prettyAvgRewardType _ ByStateValues          = "state values"
prettyAvgRewardType period (ByStateValuesAndReward ratio decay) =
  printDouble ratio' <> "*state values + " <> printDouble (1 - ratio') <> "*reward" <+>
  parens (text "Period 0" <> colon <+> printDouble ratio <> "*state values + " <> printDouble (1 - ratio) <> "*reward")
  where
    ratio' = decaySetup decay period ratio
prettyAvgRewardType _ (Fixed x)              = "fixed value of " <> double x


prettyARALTables :: (MonadIO m, Ord s, Show s, Show as) => Maybe (NetInputWoAction -> Maybe (Either String s)) -> Bool -> Bool -> Bool -> ARAL s as -> m Doc
prettyARALTables mStInverse t1 t2 t3 borl = do
  let algDoc doc
        | isAlgBorl (borl ^. algorithm) = doc
        | otherwise = empty
      algDocRho doc =
        case borl ^. algorithm of
          AlgDQN {} -> mempty
          _         -> doc
  let prBoolTblsStateAction True h m1 m2 = (h $+$) <$> prettyTablesState borl prettyState prettyActionIdx m1 prettyState noMod m2
      prBoolTblsStateAction False _ _ _  = return empty
  prettyRhoVal <-
    case (borl ^. proxies . rho, borl ^. proxies . rhoMinimum) of
      (Scalar val _, Scalar valRhoMin _) ->
        return $ text "Rho/RhoMinimum/Exp.Smth" <> colon $$ nest nestCols (printDoubleListWith 8 (V.toList val) <> text "/" <> printDoubleListWith 8 (V.toList valRhoMin)) <>
                 text "/" <> printDoubleWith 8 (borl ^. expSmoothedReward)
      (m,_) -> do
        prAct <- prettyTable borl prettyState prettyActionIdx m
        return $ text "Rho" $+$ prAct
  docHead <- prettyARALHead' False prettyState borl
  case borl ^. algorithm of
    AlgNBORL {} -> do
      prVs <- prBoolTblsStateAction t1 (text "V" $$ nest nestCols (text "PsiV")) (borl ^. proxies . v) (borl ^. proxies . psiV)
      prWs <- prBoolTblsStateAction t2 (text "W" $$ nest nestCols (text "PsiW")) (borl ^. proxies . w) (borl ^. proxies . psiW)
      prR0R1 <- prBoolTblsStateAction t3 (text "R0" $$ nest nestCols (text "R1")) (borl ^. proxies . r0) (borl ^. proxies . r1)
      return $ docHead $$ algDocRho prettyRhoVal $$ prVs $+$ prWs $+$ prR0R1
    AlgARALVOnly {} -> do
      prV <- prettyTableRows borl prettyState prettyActionIdx noMod (borl ^. proxies . v)
      return $ docHead $$ algDocRho prettyRhoVal $$ text "V" $+$ vcat prV
    AlgRLearning -> do
      prV <- prettyTableRows borl prettyState prettyActionIdx noMod (borl ^. proxies . v)
      return $ docHead $$ algDocRho prettyRhoVal $$ text "R" $+$ vcat prV
    AlgDQN {} -> do
      prR1 <- prettyTableRows borl prettyState prettyActionIdx noMod (borl ^. proxies . r1)
      return $ docHead $$ algDocRho prettyRhoVal $$ text "Q" $+$ vcat prR1
    AlgARAL{} -> do
      prR0R1 <- prBoolTblsStateAction t1 (text "V+e with gamma0" $$ nest nestCols (text "V+e with gamma1")) (borl ^. proxies . r0) (borl ^. proxies . r1)
      return $ docHead $$ algDocRho prettyRhoVal $$ prR0R1
  where
    prettyState = mkPrettyState mStInverse
    prettyActionIdx aIdx = text (show ((borl ^. actionList) VB.! (aIdx `mod` actionNrs)))
    actionNrs = length (borl ^. actionList)

mkPrettyState :: Show st => Maybe (NetInputWoAction -> Maybe (Either String st)) -> NetInputWoAction -> Maybe (Maybe st, String)
mkPrettyState mStInverse netinp =
  case mStInverse of
    Nothing  -> Just (Nothing, showDoubleList $ V.toList netinp)
    Just inv -> fromEither <$> inv netinp
  where fromEither (Left str) = (Nothing, str)
        fromEither (Right st) = (Just st, show st)

prettyARALHead ::  (MonadIO m, Show s, Show as) => Bool -> Maybe (NetInputWoAction -> Maybe (Either String s)) -> ARAL s as -> m Doc
prettyARALHead printRho mInverseSt = prettyARALHead' printRho (mkPrettyState mInverseSt)


prettyARALHead' :: (MonadIO m, Show s, Show as) => Bool -> (NetInputWoAction -> Maybe (Maybe s, String)) -> ARAL s as -> m Doc
prettyARALHead' printRho prettyStateFun borl = do
  let prettyState st = maybe ("unkown state: " ++ show st) snd (prettyStateFun st)
      prettyActionIdx aIdx = text (show ((borl ^. actionList) VB.! (aIdx `mod` actionNrs)))
      actionNrs = length (borl ^. actionList)
      prettyRhoVal =
        case (borl ^. proxies . rho, borl ^. proxies . rhoMinimum) of
          (Scalar val _, Scalar valRhoMin _) -> text "Rho/RhoMinimum/Exp.Smth Rho" <> colon $$ nest nestCols (printDoubleListWith 8 (V.toList val) <> text "/" <> printDoubleListWith 8 (V.toList valRhoMin)) <>
                                                text "/" <> printDoubleWith 8 (borl ^. expSmoothedReward)
          _                                  -> empty
  let algDoc doc
        | isAlgBorl (borl ^. algorithm) = doc
        | otherwise = empty
  let getExpSmthParam decayed p param
        | isANN = 1
        | decayed = params' ^. param
        | otherwise = borl ^. parameters . param
        where
          isANN = P.isNeuralNetwork px && borl ^. t >= px ^?! proxyNNConfig . replayMemoryMaxSize
          px = borl ^. proxies . p
  return $ text "\n" $+$ text "Current state" <> colon $$ nest nestCols (text (show $ borl ^. s) <+> "Exp. Smth Reward: " <> printDouble (borl ^. expSmoothedReward)) $+$
    vcat
      (map
         (\(WorkerState wId wSt _ _ rew) -> text "Current state Worker " <+> int wId <> colon $$ nest nestCols (text $ show wSt) <+> "Exp. Smth Reward: " <> printDouble rew)
         (borl ^. workers)) $+$

    text "Period" <> colon $$ nest nestCols (int $ borl ^. t) $+$
    text "Objective" <> colon $$ nest nestCols (text $ show $ borl ^. objective) $+$
    text "Alpha/AlphaRhoMin" <>
    colon $$
    nest nestCols (printDoubleWith 8 (getExpSmthParam True rho alpha) <> text "/" <> printDoubleWith 8 (getExpSmthParam True rhoMinimum alphaRhoMin)) <+>
    parens (text "Period 0" <> colon <+> printDoubleWith 8 (getExpSmthParam False rho alpha) <> text "/" <> printDoubleWith 8 (getExpSmthParam False rhoMinimum alphaRhoMin)) $+$
    algDoc
      (text "Beta" <> colon $$ nest nestCols (printDoubleWith 8 $ getExpSmthParam True v beta) <+>
       parens (text "Period 0" <> colon <+> printDoubleWith 8 (getExpSmthParam False v beta))) $+$
    algDoc
      (text "Delta" <> colon $$ nest nestCols (printDoubleWith 8 $ getExpSmthParam True w delta) <+>
       parens (text "Period 0" <> colon <+> printDoubleWith 8 (getExpSmthParam False w delta))) $+$
    (text "Gamma" <> colon $$ nest nestCols (printDoubleWith 8 $ getExpSmthParam True r1 gamma)) <+>
    parens (text "Period 0" <> colon <+> printDoubleWith 8 (getExpSmthParam False r1 gamma)) $+$
    text "Epsilon" <>
    colon $$
    nest nestCols (hcat $ intersperse (text ", ") $ toFiniteList $ printDoubleWith 8 <$> params' ^. epsilon) <+>
    parens (text "Period 0" <> colon <+> hcat (intersperse (text ", ") $ toFiniteList $ printDoubleWith 8 <$> params ^. epsilon)) <+>
    text "Strategy" <> colon <+> text (show $ borl ^. settings . explorationStrategy) $+$
    text "Exploration" <> colon $$ nest nestCols (printDoubleWith 8 $ params' ^. exploration) <+> parens (text "Period 0" <> colon <+> printDoubleWith 8 (params ^. exploration)) $+$
    text "Learn From Random Actions until Expl. hits" <> colon $$ nest nestCols (printDoubleWith 8 $ params' ^. learnRandomAbove) <+>
       parens (text "Period 0" <> colon <+> printDoubleWith 8 (params ^. learnRandomAbove)) $+$
    nnWorkers $+$
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
    text "Number of Agents" <> colon $$ nest nestCols (text (show $ borl ^. settings . independentAgents)) $+$
    algDoc
      (text "Zeta (for forcing V instead of W)" <> colon $$ nest nestCols (printDoubleWith 8 $ params' ^. zeta) <+>
       parens (text "Period 0" <> colon <+> printDoubleWith 8 (params ^. zeta))) $+$
    algDoc
      (text "Xi (ratio of W error forcing to V)" <> colon $$ nest nestCols (printDoubleWith 8 $ params' ^. xi) <+>
       parens (text "Period 0" <> colon <+> printDoubleWith 8 (params ^. xi))) $+$
    (case borl ^. algorithm of
       AlgNBORL {}     -> text "Scaling (V,W,R0,R1) by V config" <> colon $$ nest nestCols scalingText <>                  comma <+> text "Auto input scaling: " <> autoInpScale
       AlgARALVOnly {} -> text "Scaling BorlVOnly by V config" <> colon $$ nest nestCols scalingTextBorlVOnly <>           comma <+> text "Auto input scaling: " <> autoInpScale
       AlgRLearning    -> text "Scaling ROnly by V config" <> colon $$ nest nestCols scalingTextBorlVOnly <>               comma <+> text "Auto input scaling: " <> autoInpScale
       AlgDQN {}       -> text "Scaling (R1) by R1 Config" <> colon $$ nest nestCols scalingTextDqn <>                     comma <+> text "Auto input scaling: " <> autoInpScale
       AlgARAL {}      -> text "Scaling (R0,R1) by R1 Config" <> colon $$ nest nestCols scalingTextAvgRewardAdjustedDqn <> comma <+> text "Auto input scaling: " <> autoInpScale) $+$
    algDoc
      (text "Psi Rho/Psi V/Psi W" <> colon $$
       nest nestCols (text (show (printDoubleListWith 8 $ fromValue $ borl ^. psis . _1, printDoubleListWith 8 $ fromValue $ borl ^. psis . _2, printDoubleListWith 8 $ fromValue $ borl ^. psis . _3)))) $+$
    (if printRho
       then prettyRhoVal
       else empty) $+$
    text "Overestimate Rho" <> colon $$ nest nestCols (text $ show $ borl ^. settings. overEstimateRho) $+$
    text "Main Agent Chooses Greedy Action" <> colon $$ nest nestCols (text $ show $ borl ^. settings. mainAgentSelectsGreedyActions)
  where
    params = borl ^. parameters
    params' = decayedParameters borl
    scalingAlg cfg = case cfg ^. scaleOutputAlgorithm of
      ScaleMinMax    -> "Min-Max Normalisation"
      ScaleLog shift -> "Logarithmic Scaling w/ shift: " <> text (showDouble shift)
    nnWorkers =
      case borl ^. proxies . r1 of
        P.Table {} -> mempty
        _ ->
          text "Workers Minimum Exploration (Epsilon-Greedy)" <> semicolon $$ nest nestCols (text (showDoubleList (borl ^. settings . workersMinExploration))) <+>
          maybe mempty (\(WorkerState _ _ ms _ _) -> text "Replay memories:" <+> textReplayMemoryType ms) (borl ^? workers . _head)
    autoInpScale =
      case borl ^. proxies . r1 of
        P.Table {}           -> mempty
        P.RegressionProxy {} -> text "True (always on)"
        px                   -> text $ show (px ^?! proxyNNConfig . autoNormaliseInput)
    scalingText =
      case borl ^. proxies . v of
        P.Table {}           -> text "Tabular representation (no scaling needed)"
        P.RegressionProxy {} -> "Regression representation (no scaling needed/implemented!)"
        px                   -> textNNConf (px ^?! proxyNNConfig) <> semicolon <+> scalingAlg (px ^?! proxyNNConfig)
      where
        textNNConf conf =
          text
            (show
               ( (printDoubleWith 8 $ conf ^. scaleParameters . scaleMinVValue, printDoubleWith 8 $ conf ^. scaleParameters . scaleMaxVValue)
               , (printDoubleWith 8 $ conf ^. scaleParameters . scaleMinWValue, printDoubleWith 8 $ conf ^. scaleParameters . scaleMaxWValue)
               , (printDoubleWith 8 $ conf ^. scaleParameters . scaleMinR0Value, printDoubleWith 8 $ conf ^. scaleParameters . scaleMaxR0Value)
               , (printDoubleWith 8 $ conf ^. scaleParameters . scaleMinR1Value, printDoubleWith 8 $ conf ^. scaleParameters . scaleMaxR1Value)))
    scalingTextDqn =
      case borl ^. proxies . r1 of
        P.Table {}           -> text "Tabular representation (no scaling needed)"
        P.RegressionProxy {} -> text "Regression representation (no scaling needed)"
        px                   -> textNNConf (px ^?! proxyNNConfig) <> semicolon <+> scalingAlg (px ^?! proxyNNConfig)
      where
        textNNConf conf = text (show (printDoubleWith 8 $ conf ^. scaleParameters . scaleMinR1Value, printDoubleWith 8 $ conf ^. scaleParameters . scaleMaxR1Value))
    scalingTextAvgRewardAdjustedDqn =
      case borl ^. proxies . r1 of
        P.Table {}           -> text "Tabular representation (no scaling needed)"
        P.RegressionProxy {} -> text "Regression representation (no scaling needed)"
        px                   -> textNNConf (px ^?! proxyNNConfig) <> semicolon <+> scalingAlg (px ^?! proxyNNConfig)
      where
        textNNConf conf =
          text
            (show
               ( (printDoubleWith 8 $ conf ^. scaleParameters . scaleMinR0Value, printDoubleWith 8 $ conf ^. scaleParameters . scaleMaxR0Value)
               , (printDoubleWith 8 $ conf ^. scaleParameters . scaleMinR1Value, printDoubleWith 8 $ conf ^. scaleParameters . scaleMaxR1Value)))
    scalingTextBorlVOnly =
      case borl ^. proxies . v of
        P.Table {}           -> text "Tabular representation (no scaling needed)"
        P.RegressionProxy {} -> text "Regression representation (no scaling needed)"
        px                   -> textNNConf (px ^?! proxyNNConfig) <> colon <+> scalingAlg (px ^?! proxyNNConfig)
      where
        textNNConf conf = text (show (printDoubleWith 8 $ conf ^. scaleParameters . scaleMinVValue, printDoubleWith 8 $ conf ^. scaleParameters . scaleMaxVValue))
    nnTargetUpdate =
      case borl ^. proxies . v of
        P.Table {}           -> empty
        P.RegressionProxy {} -> empty
        px                   -> textTargetUpdate (px ^?! proxyNNConfig)
      where
        textTargetUpdate conf =
            text "NN Smooth Target Update Rate" <> colon $$ nest nestCols (printDoubleWith 8 $ conf ^. grenadeSmoothTargetUpdate) <+>
            text "every" <+> int (conf ^. grenadeSmoothTargetUpdatePeriod) <+> text "periods"
    nnBatchSize =
      case borl ^. proxies . v of
        P.Table {}           -> empty
        P.RegressionProxy {} -> empty
        px                   -> textNNConf (px ^?! proxyNNConfig)
      where
        textNNConf conf = text "NN Batchsize" <> colon $$ nest nestCols (int $ conf ^. trainBatchSize)
    nnNStep =
      case borl ^. proxies . v of
        P.Table {}           -> empty
        P.RegressionProxy {} -> empty
        px                   -> textNNConf (borl ^. settings)
      where
        textNNConf conf = text "NStep" <> colon $$ nest nestCols (int $ conf ^. nStep)
    nnReplMemSize =
      case borl ^. proxies . v of
        P.Table {}           -> empty
        P.RegressionProxy {} -> empty
        px                   -> textNNConf (px ^?! proxyNNConfig)
      where
        textNNConf conf =
          text "NN Replay Memory size" <> colon $$ nest nestCols (int $ conf ^. replayMemoryMaxSize) <+>
          maybe mempty (brackets . textReplayMemoryType) (borl ^. proxies . replayMemory)
    textReplayMemoryType ReplayMemoriesUnified {}        = text "unified replay memory"
    textReplayMemoryType mem@ReplayMemoriesPerActions {} = text "per actions each of size " <> int (replayMemoriesSubSize mem)
    nnLearningParams =
      case borl ^. proxies . r1 of
        P.Table {}                                       -> empty
        px@P.RegressionProxy{}                           -> textRegressionConf (px ^?! proxyRegressionLayer)
        P.Grenade _ _ _ conf _ _ _                       -> textGrenadeConf conf (conf ^. grenadeLearningParams)
        P.Hasktorch _ _ _ conf _ _ _ _ _                 -> textGrenadeConf conf (conf ^. grenadeLearningParams)
        P.CombinedProxy (P.Grenade _ _ _ conf _ _ _) _ _ -> textGrenadeConf conf (conf ^. grenadeLearningParams)
        _                                                -> error "nnLearningParams in Pretty.hs"
      where
        textRegressionConf :: RegressionLayer -> Doc
        textRegressionConf lay =
          text "Regression Model" <> colon $$ nest nestCols (text $ show $ regConfigModel cfg)
          where cfg = regNodeConfig $ VB.head $ fst (regressionLayerActions lay)
        textGrenadeConf :: NNConfig -> Optimizer opt -> Doc
        textGrenadeConf conf (OptSGD rate momentum l2) =
          let dec = decaySetup (conf ^. learningParamsDecay) (borl ^. t)
              l = realToFrac $ dec $ realToFrac rate
           in text "NN Learning Rate/Momentum/L2" <> colon $$
              nest nestCols (text "SGD Optimizer with" <+> text (show (printDoubleWith 8 (realToFrac l), printDoubleWith 8 (realToFrac momentum), printDoubleWith 8 (realToFrac l2))))
        textGrenadeConf conf (OptAdam alpha beta1 beta2 epsilon lambda) =
          let dec = decaySetup (conf ^. learningParamsDecay) (borl ^. t)
              l = realToFrac $ dec $ realToFrac alpha
           in text "NN Learning Rate/Momentum/L2" <> colon $$
              nest
                nestCols
                (text "Adam Optimizer with LR=" <+> nest nestCols (text (show (printDoubleWith 8 (realToFrac l)))) <+>
                parens (text "Period 0" <> colon <+> printDoubleWith 8 (realToFrac alpha))
                )


prettyARAL :: (Ord s, Show s, Show as) => ARAL s as -> IO Doc
prettyARAL = prettyARALWithStInverse Nothing

prettyARALM :: (MonadIO m, Ord s, Show s, Show as) => ARAL s as -> m Doc
prettyARALM = prettyARALTables Nothing True True True

prettyARALMWithStInverse :: (MonadIO m, Ord s, Show s, Show as) => Maybe (NetInputWoAction -> Maybe (Either String s)) -> ARAL s as -> m Doc
prettyARALMWithStInverse mStInverse = prettyARALTables mStInverse True True True


prettyARALWithStInverse :: (Ord s, Show s, Show as) => Maybe (NetInputWoAction -> Maybe (Either String s)) -> ARAL s as -> IO Doc
prettyARALWithStInverse mStInverse borl =
  prettyARALTables mStInverse True True True borl

instance (Ord s, Show s, Show as) => Show (ARAL s as) where
  show borl = renderStyle wideStyle $ unsafePerformIO $ prettyARAL borl
