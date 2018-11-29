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

prettyProxy :: (Ord k', Show k') => (k -> k') -> P.Proxy k -> Doc
prettyProxy = prettyTable

prettyTable :: (Ord k', Show k') => (k -> k') -> P.Proxy k -> Doc
prettyTable prettyKey p = vcat $ prettyTableRows prettyKey p

prettyTableRows :: (Ord k', Show k') => (k -> k') -> P.Proxy k -> [Doc]
prettyTableRows prettyAction p = case p of
  P.Table m -> map (\(k,val) -> text (show k) <> colon <+> printFloat val) (sortBy (compare `on` fst) $ M.toList (M.mapKeys prettyAction m))
  pr@P.NN{} -> map (\(k,(valT,valW)) -> text (show k) <> colon <+> printFloat valT <+> text "  " <+> printFloat valW) (sortBy (compare `on` fst) mList)
    where mList = map (first prettyAction) (mkNNList pr)

-- prettyTablesStateAction :: (Ord k, Ord k1, Show k, Show k1) => P.Proxy k -> P.Proxy k1 -> Doc
-- prettyTablesStateAction m1 m2 = vcat $ zipWith (\x y -> x $$ nest 40 y) (prettyTableRows m1) (prettyTableRows m2)

prettyTablesState :: (Ord k', Ord k1', Show k', Show k1') => (k -> k') -> P.Proxy k -> (k1 -> k1') -> P.Proxy k1 -> Doc
prettyTablesState p1 m1 p2 m2 = vcat $ zipWith (\x y -> x $$ nest 40 y) (prettyTableRows p1 m1) (prettyTableRows p2 m2)


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
  text "Zeta" <>
  colon $$
  nest 45 (printFloat $ borl ^. parameters . zeta) $+$
  text "Scaling (V,W,R0,R1) by V" <>
  colon $$
  nest 45 scalingText $+$
  text "Psi Rho/Psi V/Psi W" <>
  colon $$
  nest 45 (text (show (printFloat $ borl ^. psis . _1, printFloat $ borl ^. psis . _2, printFloat $ borl ^. psis . _3))) $+$
  (case borl ^. rho of
     Left val -> text "Rho" <> colon $$ nest 45 (printFloat val)
     Right m  -> text "Rho" $+$ prettyProxy prettyAction m) $$
  prBoolTblsStateAction t1 (text "V" $$ nest 40 (text "W")) (borl ^. v) (borl ^. w) $+$
  prBoolTblsStateAction t2 (text "R0" $$ nest 40 (text "R1")) (borl ^. r0) (borl ^. r1) $+$
  (if t3
     then text "E" $+$ prettyTable prettyAction e
     else empty) $+$
  text "Visits [%]" $+$
  prettyTable id (P.Table vis)
  where
    e =
      case (borl ^. r1, borl ^. r0) of
        (P.Table rm1, P.Table rm0) -> P.Table $ M.fromList $ zipWith subtr (M.toList rm1) (M.toList rm0)
        (prNN1@P.NN {}, prNN0@P.NN {}) -> P.Table $ M.fromList $ zipWith subtr (map (second fst) $ mkNNList prNN1) (map (second fst) $ mkNNList prNN0)
        _ -> error "Pretty printing of mixed data structures is not allowed!"
    vis = M.map (\x -> 100 * fromIntegral x / fromIntegral (borl ^. t)) (borl ^. visits)
    subtr (k, v1) (_, v2) = (k, v1 - v2)
    prBoolTblsStateAction True h m1 m2 = h $+$ prettyTablesState prettyAction m1 prettyAction m2
    prBoolTblsStateAction False _ _ _ = empty
    prettyAction (st, aIdx) = (st, maybe "unkown" (actionName . snd) (find ((== aIdx) . fst) (borl ^. actionList)))
    scalingText =
      case borl ^. v of
        P.Table {} -> text "Tabular representation (no scaling needed)"
        P.NN _ _ _ conf ->
          text
            (show
               ( (printFloat $ conf ^. scaleParameters . scaleMinVValue, printFloat $ conf ^. scaleParameters . scaleMaxVValue)
               , (printFloat $ conf ^. scaleParameters . scaleMinWValue, printFloat $ conf ^. scaleParameters . scaleMaxWValue)
               , (printFloat $ conf ^. scaleParameters . scaleMinR0Value, printFloat $ conf ^. scaleParameters . scaleMaxR0Value)
               , (printFloat $ conf ^. scaleParameters . scaleMinR1Value, printFloat $ conf ^. scaleParameters . scaleMaxR1Value)))

    nnBatchSize = case borl ^. v of
      P.Table {} -> empty
      P.NN _ _ _ conf -> text "NN Batchsize" <> colon $$ nest 45 (int $ conf ^. trainBatchSize)
    nnReplMemSize = case borl ^. v of
      P.Table {} -> empty
      P.NN _ _ _ conf -> text "NN Replay Memory size" <> colon $$ nest 45 (int $ conf ^. replayMemory.replayMemorySize)
    nnLearningParams = case borl ^. v of
      P.Table {} -> empty
      P.NN _ _ _ conf -> let LearningParameters l m l2 = conf ^. learningParams
                         in text "NN Learning Rate/Momentum/L2" <> colon $$ nest 45 (text (show (printFloat l, printFloat m, printFloat l2)))

mkNNList :: P.Proxy k -> [(k, (Double, Double))]
mkNNList pr@(P.NN _ _ _ conf) = map (\inp -> (inp, (P.lookupNeuralNetwork P.Target inp pr, P.lookupNeuralNetwork P.Worker inp pr))) (P._prettyPrintElems conf)
mkNNList _ = error "mkNNList called on non-neural network"

prettyBORL :: (Ord s, Show s) => BORL s -> Doc
prettyBORL = prettyBORLTables True True True


instance (Ord s, Show s) => Show (BORL s) where
  show borl = show (prettyBORL borl)
