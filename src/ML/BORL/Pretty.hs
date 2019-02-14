{-# LANGUAGE OverloadedStrings #-}

module ML.BORL.Pretty
    ( prettyTable
    , prettyBORL
    , prettyBORLTables
    ) where

import           ML.BORL.Action
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import           ML.BORL.Proxy         (mkNNList)
import qualified ML.BORL.Proxy         as P
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Arrow         (first, second)
import           Control.Lens
import           Data.Function         (on)
import           Data.List             (find, sortBy)
import qualified Data.Map.Strict       as M
import           Grenade
import           Prelude               hiding ((<>))
import           System.IO.Unsafe      (unsafePerformIO)
import           Text.PrettyPrint      as P
import           Text.Printf

commas :: Int
commas = 4

printFloat :: Double -> Doc
printFloat x = text $ printf ("%." ++ show commas ++ "f") x

prettyTable :: (Ord k', Show k') => Period -> (k -> k') -> P.Proxy k -> MonadBorl Doc
prettyTable period prettyKey p = vcat <$> prettyTableRows (Just period) prettyKey p

prettyTableRows :: (Ord k', Show k') => Maybe Period -> (k -> k') -> P.Proxy k -> MonadBorl [Doc]
prettyTableRows mPeriod prettyAction p =
  case p of
    P.Table m -> return $ map (\(k, val) -> text (show k) <> colon <+> printFloat val) (sortBy (compare `on` fst) $ M.toList (M.mapKeys prettyAction m))
    -- pr | maybe False (config ^. replayMemory . replayMemorySize >=) (fromIntegral <$> mPeriod) -> prettyTableRows mPeriod prettyAction (P.Table tab)
    --   where (tab, config) = case pr of
    --           P.Grenade _ _ tab' _ config' -> (tab', config')
    --           P.Tensorflow _ _ tab' _ config' -> (tab', config')
    --           _ -> error "missing implementation in mkListFromNeuralNetwork"
    pr -> do
      mfalse <- mkListFromNeuralNetwork mPeriod prettyAction False pr
      mtrue <- mkListFromNeuralNetwork mPeriod prettyAction True pr
      return $ map (\(k, (valT, valW)) -> text (show k) <> colon <+> printFloat valT <+> text "  " <+> printFloat valW) (sortBy (compare `on` fst) mfalse) ++ [text "---"] ++
        map (\(k, (valT, valW)) -> text (show k) <> colon <+> printFloat valT <+> text "  " <+> printFloat valW) (sortBy (compare `on` fst) mtrue)


mkListFromNeuralNetwork :: (Integral a) => Maybe a -> (k -> c) -> Bool -> P.Proxy k -> MonadBorl [(c, (Double, Double))]
mkListFromNeuralNetwork mPeriod prettyAction scaled pr
  | maybe False (config ^. replayMemory . replayMemorySize >=) (fromIntegral <$> mPeriod) = do
      list <- mkNNList scaled pr
      return $ map (first prettyAction) $ zip (map fst $ M.toList tab) (zip (map snd $ M.toList tab) (map (snd . snd) list))
  | otherwise = map (first prettyAction) <$> mkNNList scaled pr
  where (tab,config) = case pr of
          P.Grenade _ _ tab' _ config' _ -> (tab', config')
          P.TensorflowProxy _ _ tab' _ config' _ -> (tab', config')
          _ -> error "missing implementation in mkListFromNeuralNetwork"

prettyTablesState :: (Ord k', Ord k1', Show k', Show k1') => Period -> (k -> k') -> P.Proxy k -> (k1 -> k1') -> P.Proxy k1 -> MonadBorl Doc
prettyTablesState period p1 m1 p2 m2 = do
  rows1 <- prettyTableRows (Just period) p1 m1
  rows2 <- prettyTableRows (Just period) p2 m2
  return $ vcat $ zipWith (\x y -> x $$ nest 40 y) rows1 rows2


prettyBORLTables :: (Ord s, Show s) => Bool -> Bool -> Bool -> BORL s -> MonadBorl Doc
prettyBORLTables t1 t2 t3 borl = do

  let prBoolTblsStateAction True h m1 m2 = (h $+$) <$> prettyTablesState (borl ^. t) prettyAction m1 prettyAction m2
      prBoolTblsStateAction False _ _ _ = return empty
  let mkErr scale = case (borl ^. r1, borl ^. r0) of
           (P.Table rm1, P.Table rm0) -> return $ P.Table $ M.fromList $ zipWith subtr (M.toList rm1) (M.toList rm0)
           (prNN1, prNN0) -> do
             n1 <- mkNNList scale prNN1
             n0 <- mkNNList scale prNN0
             return $ P.Table $ M.fromList $ zipWith subtr (map (second fst) n1) (map (second fst) n0)
  errUnscaled <- mkErr False
  errScaled <- mkErr True
  prettyErr <- if t3
               then prBoolTblsStateAction t3 "E" errUnscaled errScaled
               else return empty
  prettyRhoVal <- case borl ^. rho of
       Left val -> return $ text "Rho" <> colon $$ nest 45 (printFloat val)
       Right m  -> do
         prAct <- prettyTable (borl ^. t) prettyAction m
         return $ text "Rho" $+$ prAct
  prettyVisits <- prettyTable (borl ^. t) id (P.Table vis)
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
    text "Zeta (enables W error forcing)" <>
    colon $$
    nest 45 (printFloat $ borl ^. parameters . zeta) $+$
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
    vis = M.map (\x -> 100 * fromIntegral x / fromIntegral (borl ^. t)) (borl ^. visits)
    subtr (k, v1) (_, v2) = (k, v1 - v2)
    prettyAction (st, aIdx) = (st, maybe "unkown" (actionName . snd) (find ((== aIdx) . fst) (borl ^. actionList)))
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


