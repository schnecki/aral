{-# LANGUAGE OverloadedStrings #-}

module ML.BORL.Pretty
    ( prettyV
    , prettyW
    , prettyR0
    , prettyR1
    , prettyTable
    , prettyBORL
    , prettyBORLTables
    ) where

import           ML.BORL.Parameters
import           ML.BORL.Type

import           Control.Lens
import           Data.Function      (on)
import           Data.List          (find, sortBy)
import qualified Data.Map.Strict    as M
import           Prelude            hiding ((<>))
import           Text.PrettyPrint   as P
import           Text.Printf

commas :: Int
commas = 4

printFloat :: Double -> Doc
printFloat x = text $ printf ("%." ++ show commas ++ "f") x


prettyV :: (Ord s, Show s) => BORL s -> Doc
prettyV borl = prettyTable (borl ^. v)

prettyW :: (Ord s, Show s) => BORL s -> Doc
prettyW borl = prettyTable (borl ^. w)

prettyR0 :: (Ord s, Show s) => BORL s -> Doc
prettyR0 borl = prettyTable (borl ^. r0)

prettyR1 :: (Ord s, Show s) => BORL s -> Doc
prettyR1 borl = prettyTable (borl ^. r1)

prettyE :: (Ord s, Show s) => BORL s -> Doc
prettyE borl = prettyTable (M.fromList $ zipWith subtr (M.toList $ borl^.r1) (M.toList $ borl^.r0))
  where subtr (k,v1) (_,v2) = (k,v1-v2)

prettyVRhoR1 :: (Ord s, Show s) => BORL s -> Doc
prettyVRhoR1 borl = prettyTable (M.fromList $ zipWith subtr (map (\(k,v) -> (k,v+ avgRew k)) (M.toList $ borl^.v)) (M.toList $ borl^.r0))
  where subtr (k,v1) (_,v2) = (k,v1-v2)
        avgRew s = case borl ^. rho of
          Left v  -> v
          Right m -> M.findWithDefault 0 s m

prettyTable :: (Ord s, Show s) => M.Map s Double -> Doc
prettyTable = vcat . prettyTableRows

prettyTableRows :: (Ord s, Show s) => M.Map s Double -> [Doc]
prettyTableRows m = map (\(k,v) -> text (show k) <> colon <+> printFloat v) (sortBy (compare `on` fst) $ M.toList m)

prettyTablesStateAction :: (Ord s, Ord s1, Show s, Show s1) => M.Map s Double -> M.Map s1 Double -> Doc
prettyTablesStateAction m1 m2 = vcat $ zipWith (\x y -> x $$ nest 40 y) (prettyTableRows m1) (prettyTableRows m2)

prettyTablesState :: (Ord s, Ord s1, Show s, Show s1) => M.Map s Double -> M.Map s1 Double -> Doc
prettyTablesState m1 m2 = vcat $ zipWith (\x y -> x $$ nest 40 y) (prettyTableRows m1) (prettyTableRows m2)


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
  text "Learning Random Actions" <>
  colon $$
  nest 45 (printFloat $ borl ^. parameters . learnRandomAbove) $+$
  text "Gammas" <>
  colon $$
  nest 45 (text (show (printFloat $ borl ^. gammas . _1, printFloat $ borl ^. gammas . _2))) $+$
  text "Xi (ratio of W error forcing to V)" <>
  colon $$
  nest 45 (printFloat $ borl ^. parameters . xi) $+$
  text "Psi Rho/Psi V/Psi W" <>
  colon $$
  nest 45 (text (show (printFloat $ borl ^. psis . _1, printFloat $ borl ^. psis . _2, printFloat $ borl ^. psis . _3))) $+$
  (case borl ^. rho of
     Left v  -> text "Rho" <> colon $$ nest 45 (printFloat v)
     Right m -> text "Rho" $+$ prettyTable (M.mapKeys prettyAction m)) $$
  prBoolTblsStateAction t1 (text "V" $$ nest 40 (text "W")) (borl ^. v) (borl ^. w) $+$
  prBoolTblsState t2 (text "Psi V" $$ nest 40 (text "Psi W")) (borl ^. psiStates._2) (borl ^. psiStates._3) $$
  -- prBoolTbls t2 (text "V+Psi V" $$ nest 40 (text "W + Psi W")) (M.fromList $ zipWith add (M.toList $ borl ^. v) (M.toList $ borl ^. psiStates._2))
  -- (M.fromList $ zipWith add (M.toList $ borl ^. w) (M.toList $ borl ^. psiStates._3)) $+$
  prBoolTblsStateAction t2 (text "R0" $$ nest 40 (text "R1")) (borl ^. r0) (borl ^. r1) $+$
  (if t3 then text "E" $+$ prettyTable (M.mapKeys prettyAction e) else empty) $+$
  text "Visits [%]" $+$ prettyTable vis
  where
    e = M.fromList $ zipWith subtr (M.toList $ borl ^. r1) (M.toList $ borl ^. r0)
    vis = M.map (\x -> 100 * fromIntegral x / fromIntegral (borl ^. t)) (borl ^. visits)
    subtr (k, v1) (_, v2) = (k, v1 - v2)
    add (k, v1) (_, v2) = (k, v1 + v2)
    prBoolTblsState True h m1 m2 = h $+$ prettyTablesState m1 m2
    prBoolTblsState False _ _ _  = empty
    prBoolTblsStateAction True h m1 m2 = h $+$ prettyTablesStateAction (M.mapKeys prettyAction m1) (M.mapKeys prettyAction m2)
    prBoolTblsStateAction False _ _ _  = empty
    prettyAction (s,aIdx) = (s, maybe "unkown" (actionName . snd) (find ((== aIdx) . fst) (borl ^. actionList)))


prettyBORL :: (Ord s, Show s) => BORL s -> Doc
prettyBORL = prettyBORLTables True True True


instance (Ord s, Show s) => Show (BORL s) where
  show borl = show (prettyBORL borl)
