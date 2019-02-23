{-# LANGUAGE OverloadedStrings #-}

module ML.BORL.Pretty
    ( prettyTable
    , prettyBORL
    , prettyBORLTables
    ) where

import           ML.BORL.Action
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import qualified ML.BORL.Proxy         as P
import           ML.BORL.Proxy.Ops     (mkNNList)
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Arrow         (first, second, (&&&))
import           Control.Lens
import           Control.Monad         (when)
import           Data.Function         (on)
import           Data.List             (find, sort, sortBy)
import qualified Data.Map.Strict       as M
import qualified Data.Text             as T
import           Grenade
import           Prelude               hiding ((<>))
import           System.IO.Unsafe      (unsafePerformIO)
import           Text.PrettyPrint      as P
import           Text.Printf

commas :: Int
commas = 4

printFloat :: Double -> Doc
printFloat x = text $ printf ("%." ++ show commas ++ "f") x

prettyTable :: (Show k, Eq k, Ord k, Ord k', Show k') => BORL k -> (k -> k') -> (ActionIndex -> Doc) -> P.Proxy k -> MonadBorl Doc
prettyTable borl prettyKey prettyIdx p = vcat <$> prettyTableRows borl prettyKey prettyIdx p

prettyTableRows :: (Show k, Ord k, Eq k, Ord k', Show k') => BORL k -> (k -> k') -> (ActionIndex -> Doc) -> P.Proxy k -> MonadBorl [Doc]
prettyTableRows borl prettyAction prettyActionIdx p =
  case p of
    P.Table m -> return $ map (\((k,idx),val) -> prettyActionEntry id prettyActionIdx k idx <> colon <+> printFloat val) $
                 sortBy (compare `on` fst) $ M.toList (M.mapKeys (first prettyAction) m)
    -- pr | maybe False (config ^. replayMemory . replayMemorySize >=) (fromIntegral <$> mPeriod) -> prettyTableRows mPeriod prettyAction (P.Table tab)
    --   where (tab, config) = case pr of
    --           P.Grenade _ _ tab' _ config' -> (tab', config')
    --           P.Tensorflow _ _ tab' _ config' -> (tab', config')
    --           _ -> error "missing implementation in mkListFromNeuralNetwork"
    pr -> do
      mfalse <- mkListFromNeuralNetwork borl prettyAction prettyActionIdx False pr
      mtrue <- mkListFromNeuralNetwork borl prettyAction prettyActionIdx True pr
      let printFun (kDoc, (valT, valW)) = kDoc <> colon <+> printFloat valT <+> text "  " <+> printFloat valW
          unfoldActs = concatMap (\(f,(ts,ws)) -> zipWith (\(nr,t) (_,w) -> (f nr, (t, w))) ts ws)
      return $ map printFun (unfoldActs mfalse) ++ [text "---"] ++ map printFun (unfoldActs mtrue)


prettyActionEntry :: (Show k') => (k -> k') -> (ActionIndex -> Doc) -> k -> ActionIndex -> Doc
prettyActionEntry pAct pActIdx act actIdx = text (show $ pAct act) <> colon <+> pActIdx actIdx


mkListFromNeuralNetwork ::
     (Show k, Ord k, Eq k, Show k')
  => BORL k
  -> (k -> k')
  -> (ActionIndex -> Doc)
  -> Bool
  -> P.Proxy k
  -> MonadBorl [(ActionIndex -> Doc, ([(ActionIndex, Double)], [(ActionIndex, Double)]))]
mkListFromNeuralNetwork borl prettyAction prettyActionIdx scaled pr = map (first $ prettyActionEntry prettyAction prettyActionIdx) <$> mkNNList borl scaled pr

prettyTablesState :: (Show k, Ord k, Eq k, Ord k', Show k') => BORL k -> Period -> (k -> k') -> (ActionIndex -> Doc) -> P.Proxy k -> (k -> k') -> P.Proxy k -> MonadBorl Doc
prettyTablesState borl period p1 pIdx m1 p2 m2 = do
  rows1 <- prettyTableRows borl p1 pIdx (if fromTable then tbl m1 else m1)
  rows2 <- prettyTableRows borl p2 pIdx (if fromTable then tbl m2 else m2)
  return $ vcat $ zipWith (\x y -> x $$ nest 40 y) rows1 rows2
  where fromTable = period < fromIntegral memSize
        memSize = case m1 of
          P.Table{}                       -> -1
          P.Grenade _ _ _ _ cfg _         -> cfg ^?! replayMemory.replayMemorySize
          P.TensorflowProxy _ _ _ _ cfg _ -> cfg ^?! replayMemory.replayMemorySize
        tbl px = case px of
          p@P.Table{}                   -> p
          P.Grenade _ _ p _ _ _         -> P.Table p
          P.TensorflowProxy _ _ p _ _ _ -> P.Table p


prettyBORLTables :: (Ord s, Show s) => Bool -> Bool -> Bool -> BORL s -> MonadBorl Doc
prettyBORLTables t1 t2 t3 borl = do

  let prBoolTblsStateAction True h m1 m2 = (h $+$) <$> prettyTablesState borl (borl ^. t) prettyAction prettyActionIdx m1 prettyAction m2
      prBoolTblsStateAction False _ _ _ = return empty
  let mkErr scale = case (borl ^. r1, borl ^. r0) of
           (P.Table rm1, P.Table rm0) -> return $ P.Table $ M.fromList $ zipWith subtr (M.toList rm1) (M.toList rm0)
           (prNN1, prNN0) -> do
             n1 <- mkNNList borl scale prNN1
             n0 <- mkNNList borl scale prNN0
             return $ P.Table $ M.fromList $ concat $ zipWith (\(k,ts) (_,ws) -> zipWith (\(nr, t) (_,w) -> ((k, nr), t-w)) ts ws) (map (second fst) n1) (map (second fst) n0)
  errUnscaled <- mkErr False
  errScaled <- mkErr True
  prettyErr <- if t3
               then prBoolTblsStateAction t3 "Error W / Error R1-R0" (borl ^. psiWTbl) errScaled
               else return empty
  prettyRhoVal <- case borl ^. rho of
       Left val -> return $ text "Rho" <> colon $$ nest 45 (printFloat val)
       Right m  -> do
         prAct <- prettyTable borl prettyAction prettyActionIdx m
         return $ text "Rho" $+$ prAct
  prettyVisits <- prettyTable borl id (const empty) (P.Table vis)
  prVW <- prBoolTblsStateAction t1 (text "V" $$ nest 40 (text "W")) (borl ^. v) (borl ^. w)
  prR0R1 <- prBoolTblsStateAction t2 (text "R0" $$ nest 40 (text "R1")) (borl ^. r0) (borl ^. r1)
  return $
    text "\n" $+$ text "Current state" <> colon $$ nest 45 (text (show $ borl ^. s)) $+$ text "Period" <> colon $$ nest 45 (integer $ borl ^. t) $+$ text "Alpha" <> colon $$
    nest 45 (printFloat $ borl ^. parameters . alpha) $+$
    text "Beta" <>
    colon $$
    nest 45 (printFloat $ borl ^. parameters . beta) $+$
    text "Delta" <>
    colon $$
    nest 45 (printFloat $ borl ^. parameters . delta) $+$
    text "Epsilon" <>
    colon $$
    nest 45 (printFloat $ borl ^. parameters . epsilon) $+$
    text "Exploration" <>
    colon $$
    nest 45 (printFloat $ borl ^. parameters . exploration) $+$
    text "Learn From Random Actions until Expl. hits" <>
    colon $$
    nest 45 (printFloat $ borl ^. parameters . learnRandomAbove) $+$
    nnBatchSize $+$
    nnReplMemSize $+$
    nnLearningParams $+$
    text "Gammas" <>
    colon $$
    nest 45 (text (show (printFloat $ borl ^. gammas . _1, printFloat $ borl ^. gammas . _2))) $+$
    text "Xi (ratio of W error forcing to V)" <>
    colon $$
    nest 45 (printFloat $ borl ^. parameters . xi) $+$
    text "Scaling (V,W,R0,R1) by V Config" <>
    colon $$
    nest 45 scalingText $+$
    text "Psi Rho/Psi V/Psi W" <>
    colon $$
    nest 45 (text (show (printFloat $ borl ^. psis . _1, printFloat $ borl ^. psis . _2, printFloat $ borl ^. psis . _3))) $+$
    prettyRhoVal $$
    prVW $+$
    prR0R1 $+$
    prettyErr $+$
    text "Visits [%]" $+$
    prettyVisits
  where
    vis = M.mapKeys (\x -> (x,0)) $ M.map (\x -> 100 * fromIntegral x / fromIntegral (borl ^. t)) (borl ^. visits)
    subtr (k, v1) (_, v2) = (k, v1 - v2)
    -- prettyAction (st, aIdx) = (st, maybe "unkown" (actionName . snd) (find ((== aIdx) . fst) (borl ^. actionList)))
    prettyAction st = st
    prettyActionIdx aIdx = text (T.unpack $ maybe "unkown" (actionName . snd) (find ((== aIdx) . fst) (borl ^. actionList)))
    scalingText =
      case borl ^. v of
        P.Table {} -> text "Tabular representation (no scaling needed)"
        P.Grenade _ _ _ _ conf _ ->
          text
            (show
               ( (printFloat $ conf ^. scaleParameters . scaleMinVValue, printFloat $ conf ^. scaleParameters . scaleMaxVValue)
               , (printFloat $ conf ^. scaleParameters . scaleMinWValue, printFloat $ conf ^. scaleParameters . scaleMaxWValue)
               , (printFloat $ conf ^. scaleParameters . scaleMinR0Value, printFloat $ conf ^. scaleParameters . scaleMaxR0Value)
               , (printFloat $ conf ^. scaleParameters . scaleMinR1Value, printFloat $ conf ^. scaleParameters . scaleMaxR1Value)))
        P.TensorflowProxy _ _ _ _ conf _ ->
          text
            (show
               ( (printFloat $ conf ^. scaleParameters . scaleMinVValue, printFloat $ conf ^. scaleParameters . scaleMaxVValue)
               , (printFloat $ conf ^. scaleParameters . scaleMinWValue, printFloat $ conf ^. scaleParameters . scaleMaxWValue)
               , (printFloat $ conf ^. scaleParameters . scaleMinR0Value, printFloat $ conf ^. scaleParameters . scaleMaxR0Value)
               , (printFloat $ conf ^. scaleParameters . scaleMinR1Value, printFloat $ conf ^. scaleParameters . scaleMaxR1Value)))

    nnBatchSize = case borl ^. v of
      P.Table {} -> empty
      P.Grenade _ _ _ _ conf _ -> text "NN Batchsize" <> colon $$ nest 45 (int $ conf ^. trainBatchSize)
      P.TensorflowProxy _ _ _ _ conf _ -> text "NN Batchsize" <> colon $$ nest 45 (int $ conf ^. trainBatchSize)
    nnReplMemSize = case borl ^. v of
      P.Table {} -> empty
      P.Grenade _ _ _ _ conf _ -> text "NN Replay Memory size" <> colon $$ nest 45 (int $ conf ^. replayMemory.replayMemorySize)
      P.TensorflowProxy _ _ _ _ conf _ -> text "NN Replay Memory size" <> colon $$ nest 45 (int $ conf ^. replayMemory.replayMemorySize)
    nnLearningParams = case borl ^. v of
      P.Table {} -> empty
      P.Grenade _ _ _ _ conf _ -> let LearningParameters l m l2 = conf ^. learningParams
                         in text "NN Learning Rate/Momentum/L2" <> colon $$ nest 45 (text (show (printFloat l, printFloat m, printFloat l2)))
      P.TensorflowProxy _ _ _ _ conf _ -> let LearningParameters l m l2 = conf ^. learningParams
                         in text "NN Learning Rate/Momentum/L2" <> colon $$ nest 45 (text "Specified in tensorflow model")


prettyBORL :: (Ord s, Show s) => BORL s -> IO Doc
prettyBORL borl = runMonadBorl $ do
  buildModels
  reloadNets (borl ^. v)
  reloadNets (borl ^. w)
  reloadNets (borl ^. r0)
  reloadNets (borl ^. r1)
  prettyBORLTables True True True borl
    where reloadNets px = case px of
            P.TensorflowProxy netT netW _ _ _ _ -> restoreModelWithLastIO netT >> restoreModelWithLastIO netW
            _ -> return ()
          isTensorflowProxy P.TensorflowProxy{} = True
          isTensorflowProxy _                   = False
          buildModels = case find isTensorflowProxy [borl^.v, borl^.w, borl^.r0, borl^.r1] of
            Just (P.TensorflowProxy netT _ _ _ _ _) -> buildTensorflowModel netT
            _                                       -> return ()

instance (Ord s, Show s) => Show (BORL s) where
  show borl = show $ unsafePerformIO $ prettyBORL borl


