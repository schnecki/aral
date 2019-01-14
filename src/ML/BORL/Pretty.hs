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
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Arrow         (first, second)
import           Control.Lens
import           Data.Function         (on)
import           Data.List             (find, sortBy)
import qualified Data.Map.Strict       as M
import           Grenade
import           Prelude               hiding ((<>))
import           Text.PrettyPrint      as P
import           Text.Printf

commas :: Int
commas = 4

printFloat :: Double -> Doc
printFloat x = text $ printf ("%." ++ show commas ++ "f") x

-- prettyE :: (Ord s, Show s) => BORL s -> Doc
-- prettyE borl = case (borl^.r1,borl^.r0) of
--   (P.Table rm1, P.Table rm0) -> prettyTable id (P.Table $ M.fromList $ zipWith subtr (M.toList rm1) (M.toList rm0))
--   where subtr (k,v1) (_,v2) = (k,v1-v2)

prettyProxy :: (Ord k', Show k') => Period -> (k -> k') -> P.Proxy k -> Doc
prettyProxy = prettyTable

prettyTable :: (Ord k', Show k') => Period -> (k -> k') -> P.Proxy k -> Doc
prettyTable period prettyKey p = vcat $ prettyTableRows (Just period) prettyKey p

prettyTableRows :: (Ord k', Show k') => Maybe Period -> (k -> k') -> P.Proxy k -> [Doc]
prettyTableRows mPeriod prettyAction p =
  case p of
    P.Table m -> map (\(k, val) -> text (show k) <> colon <+> printFloat val) (sortBy (compare `on` fst) $ M.toList (M.mapKeys prettyAction m))
    pr@(P.Grenade _ _ tab _ config) ->
      map (\(k, (valT, valW)) -> text (show k) <> colon <+> printFloat valT <+> text "  " <+> printFloat valW) (sortBy (compare `on` fst) (mList False)) ++ [text "---"] ++
      map (\(k, (valT, valW)) -> text (show k) <> colon <+> printFloat valT <+> text "  " <+> printFloat valW) (sortBy (compare `on` fst) (mList True))
      where mList scaled
              | maybe False (config ^. replayMemory . replayMemorySize >=) (fromIntegral <$> mPeriod) =
                map (first prettyAction) $ zip (map fst $ M.toList tab) (zip (map snd $ M.toList tab) (map (snd . snd) (mkNNList scaled pr)))
              | otherwise = map (first prettyAction) (mkNNList scaled pr)
    pr@(P.Tensorflow _ _ tab _ config) -> undefined

prettyTablesState :: (Ord k', Ord k1', Show k', Show k1') => Period -> (k -> k') -> P.Proxy k -> (k1 -> k1') -> P.Proxy k1 -> Doc
prettyTablesState period p1 m1 p2 m2 = vcat $ zipWith (\x y -> x $$ nest 40 y) (prettyTableRows (Just period) p1 m1) (prettyTableRows (Just period) p2 m2)


prettyBORLTables :: (Ord s, Show s) => Bool -> Bool -> Bool -> BORL s -> Doc
prettyBORLTables t1 t2 t3 borl =
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
  text "Zeta (enables W error forcing)" <>
  colon $$
  nest 45 (printFloat $ borl ^. parameters . zeta) $+$
  text "Scaling (V,W,R0,R1) by V Config" <>
  colon $$
  nest 45 scalingText $+$
  text "Psi Rho/Psi V/Psi W" <>
  colon $$
  nest 45 (text (show (printFloat $ borl ^. psis . _1, printFloat $ borl ^. psis . _2, printFloat $ borl ^. psis . _3))) $+$
  (case borl ^. rho of
     Left val -> text "Rho" <> colon $$ nest 45 (printFloat val)
     Right m  -> text "Rho" $+$ prettyProxy (borl ^. t) prettyAction m) $$
  prBoolTblsStateAction t1 (text "V" $$ nest 40 (text "W")) (borl ^. v) (borl ^. w) $+$
  prBoolTblsStateAction t2 (text "R0" $$ nest 40 (text "R1")) (borl ^. r0) (borl ^. r1) $+$
  (if t3
     then text "E" $+$ prettyTable (borl ^. t) prettyAction e
     else empty) $+$
  text "Visits [%]" $+$
  prettyTable (borl ^. t) id (P.Table vis)
  where
    e =
      case (borl ^. r1, borl ^. r0) of
        (P.Table rm1, P.Table rm0) -> P.Table $ M.fromList $ zipWith subtr (M.toList rm1) (M.toList rm0)
        (prNN1@P.Grenade {}, prNN0@P.Grenade {}) -> P.Table $ M.fromList $ zipWith subtr (map (second fst) $ mkNNList False prNN1) (map (second fst) $ mkNNList False prNN0)
        _ -> error "Pretty printing of mixed data structures is not allowed!"
    vis = M.map (\x -> 100 * fromIntegral x / fromIntegral (borl ^. t)) (borl ^. visits)
    subtr (k, v1) (_, v2) = (k, v1 - v2)
    prBoolTblsStateAction True h m1 m2 = h $+$ prettyTablesState (borl ^. t) prettyAction m1 prettyAction m2
    prBoolTblsStateAction False _ _ _ = empty
    prettyAction (st, aIdx) = (st, maybe "unkown" (actionName . snd) (find ((== aIdx) . fst) (borl ^. actionList)))
    scalingText =
      case borl ^. v of
        P.Table {} -> text "Tabular representation (no scaling needed)"
        P.Grenade _ _ _ _ conf ->
          text
            (show
               ( (printFloat $ conf ^. scaleParameters . scaleMinVValue, printFloat $ conf ^. scaleParameters . scaleMaxVValue)
               , (printFloat $ conf ^. scaleParameters . scaleMinWValue, printFloat $ conf ^. scaleParameters . scaleMaxWValue)
               , (printFloat $ conf ^. scaleParameters . scaleMinR0Value, printFloat $ conf ^. scaleParameters . scaleMaxR0Value)
               , (printFloat $ conf ^. scaleParameters . scaleMinR1Value, printFloat $ conf ^. scaleParameters . scaleMaxR1Value)))

    nnBatchSize = case borl ^. v of
      P.Table {} -> empty
      P.Grenade _ _ _ _ conf -> text "NN Batchsize" <> colon $$ nest 45 (int $ conf ^. trainBatchSize)
    nnReplMemSize = case borl ^. v of
      P.Table {} -> empty
      P.Grenade _ _ _ _ conf -> text "NN Replay Memory size" <> colon $$ nest 45 (int $ conf ^. replayMemory.replayMemorySize)
    nnLearningParams = case borl ^. v of
      P.Table {} -> empty
      P.Grenade _ _ _ _ conf -> let LearningParameters l m l2 = conf ^. learningParams
                         in text "NN Learning Rate/Momentum/L2" <> colon $$ nest 45 (text (show (printFloat l, printFloat m, printFloat l2)))

mkNNList :: Bool -> P.Proxy k -> [(k, (Double, Double))]
mkNNList unscaled pr@(P.Grenade _ _ _ _ conf) = map (\inp -> (inp, ( if unscaled then P.lookupNeuralNetwork P.Target inp pr else P.lookupNeuralNetworkUnscaled P.Target inp pr,
                                                               if unscaled then P.lookupNeuralNetwork P.Worker inp pr else P.lookupNeuralNetworkUnscaled P.Worker inp pr))) (P._prettyPrintElems conf)
mkNNList _ _ = error "mkNNList called on non-neural network"

prettyBORL :: (Ord s, Show s) => BORL s -> Doc
prettyBORL = prettyBORLTables True True True


instance (Ord s, Show s) => Show (BORL s) where
  show borl = show (prettyBORL borl)
