{-# LANGUAGE CPP               #-}
{-# LANGUAGE OverloadedStrings #-}
module ML.BORL.Pretty
    ( prettyTable
    , prettyBORL
    , prettyBORLTables
    , wideStyle
    ) where


import           ML.BORL.Action
import           ML.BORL.Algorithm
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import qualified ML.BORL.Proxy         as P
import           ML.BORL.Proxy.Ops     (mkNNList)
import           ML.BORL.Proxy.Type
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Arrow         (first, second, (&&&))
import           Control.Lens
import           Control.Monad         (when)
import           Data.Function         (on)
import           Data.List             (find, sort, sortBy)
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
commas = 2

wideStyle :: Style
wideStyle = Style { lineLength = 200, ribbonsPerLine = 1.5, mode = PageMode }

printFloat :: Double -> Doc
printFloat x = text $ printf ("%." ++ show commas ++ "f") x

prettyTable :: (Show k, Eq k, Ord k, Ord k', Show k') => BORL k -> (NetInputWoAction -> k') -> (ActionIndex -> Doc) -> P.Proxy k -> MonadBorl Doc
prettyTable borl prettyKey prettyIdx p = vcat <$> prettyTableRows borl prettyKey prettyIdx (\_ v -> return v) p

prettyTableRows ::
     (Show k, Ord k, Eq k, Ord k', Show k')
  => BORL k
  -> (NetInputWoAction -> k')
  -> (ActionIndex -> Doc)
  -> (([Double], ActionIndex) -> Double -> MonadBorl Double)
  -> P.Proxy k
  -> MonadBorl [Doc]
prettyTableRows borl prettyAction prettyActionIdx modifier p =
  case p of
    P.Table m _ gen ->
      let mkAct idx = actionName $ snd $ (borl ^. actionList) !! idx
          mkInput k = text (filter (/= '"') $ show $ map (\x -> if x < 0 then printFloat x else "+" <> printFloat x) k)
      in mapM (\((k,idx),val) -> modifier (k,idx) val >>= \v -> return ( mkInput k <> text (T.unpack $ mkAct idx) <> colon <+> printFloat v)) $
      sortBy (compare `on`  fst.fst) $ M.toList m
    pr -> do
      mfalse <- mkListFromNeuralNetwork borl prettyAction prettyActionIdx False pr
      mtrue <- mkListFromNeuralNetwork borl prettyAction prettyActionIdx True pr
      let printFun (kDoc, (valT, valW)) = kDoc <> colon <+> printFloat valT <+> text "  " <+> printFloat valW
          unfoldActs = concatMap (\(f,(ts,ws)) -> zipWith (\(nr,t) (_,w) -> (f nr, (t, w))) ts ws)
      return $ map printFun (unfoldActs mtrue)


prettyActionEntry :: (Show k') => (k -> k') -> (ActionIndex -> Doc) -> k -> ActionIndex -> Doc
prettyActionEntry pAct pActIdx act actIdx = text (show $ pAct act) <> colon <+> pActIdx actIdx


mkListFromNeuralNetwork ::
     (Show k, Ord k, Eq k, Show k')
  => BORL k
  -> (NetInputWoAction -> k')
  -> (ActionIndex -> Doc)
  -> Bool
  -> P.Proxy k
  -> MonadBorl [(ActionIndex -> Doc, ([(ActionIndex, Double)], [(ActionIndex, Double)]))]
mkListFromNeuralNetwork borl prettyAction prettyActionIdx scaled pr = map (first $ prettyActionEntry prettyAction prettyActionIdx) <$> mkNNList borl scaled pr

prettyTablesState :: (Show k, Ord k, Eq k, Ord k', Show k') => BORL k -> Period -> (NetInputWoAction -> k') -> (ActionIndex -> Doc) -> P.Proxy k -> (NetInputWoAction -> k') -> P.Proxy k -> MonadBorl Doc
prettyTablesState borl period p1 pIdx m1 p2 m2 = do
  rows1 <- prettyTableRows borl p1 pIdx (\_ v -> return v) (if fromTable then tbl m1 else m1)
  rows2 <- prettyTableRows borl p2 pIdx (\_ v -> return v) (if fromTable then tbl m2 else m2)
  return $ vcat $ zipWith (\x y -> x $$ nest 40 y) rows1 rows2
  where fromTable = period < fromIntegral memSize
        memSize = case m1 of
          P.Table{}                       -> -1
          P.Grenade _ _ _ _ cfg _         -> cfg ^?! replayMemoryMaxSize
          P.TensorflowProxy _ _ _ _ cfg _ -> cfg ^?! replayMemoryMaxSize
        tbl px = case px of
          p@P.Table{}                     -> p
          P.Grenade _ _ p _ cfg _         -> P.Table p 0 (const [])
          P.TensorflowProxy _ _ p _ cfg _ -> P.Table p 0 (const [])

prettyAlgorithm :: Algorithm -> Doc
prettyAlgorithm (AlgBORL ga0 ga1 avgRewType stValHand vPlusPsiV) = text "BORL with gammas " <+> text (show (ga0, ga1)) <> text ";" <+> prettyAvgRewardType avgRewType <+> text "for rho" <> text ";" <+> prettyStateValueHandling stValHand <+> text "Deciding on" <+> text (if vPlusPsiV then "V + PsiV" else "V")
prettyAlgorithm (AlgDQN ga1)      = text "DQN with gamma" <+> text (show ga1)
prettyAlgorithm (AlgDQNAvgRew ga1 avgRewType)      = text "DQN SUBTRACT AvgReward with gamma" <+> text (show ga1) <> text ";" <+> prettyAvgRewardType avgRewType

prettyStateValueHandling :: StateValueHandling -> Doc
prettyStateValueHandling Normal = empty
prettyStateValueHandling (DivideValuesAfterGrowth nr max) = text "Divide values after growth " <> parens (int nr <> text "," <+> integer max) <> text ";"

prettyAvgRewardType :: AvgReward -> Doc
prettyAvgRewardType (ByMovAvg nr) = "moving average" <> parens (int nr)
prettyAvgRewardType ByReward      = "reward"
prettyAvgRewardType ByStateValues = "state values"
prettyAvgRewardType (Fixed x)     = "fixed value of " <> double x


prettyBORLTables :: (Ord s, Show s) => Bool -> Bool -> Bool -> BORL s -> MonadBorl Doc
prettyBORLTables t1 t2 t3 borl = do

  let algDoc doc | isAlgBorl (borl ^. algorithm) = doc
                 | otherwise = empty
      algDocRho doc = case borl ^. algorithm of
        AlgDQN{} -> empty
        _        -> doc

  let prBoolTblsStateAction True h m1 m2 = (h $+$) <$> prettyTablesState borl (borl ^. t) prettyAction prettyActionIdx m1 prettyAction m2
      prBoolTblsStateAction False _ _ _ = return empty
  let mkErr scale = case (borl ^. proxies.r1, borl ^. proxies.r0) of
           (P.Table rm1 _ _, P.Table rm0 _ _) -> return $ (\t -> P.Table t 0 (const [])) (M.fromList $ zipWith subtr (M.toList rm1) (M.toList rm0))
           (prNN1, prNN0) -> do
             n1 <- mkNNList borl scale prNN1
             n0 <- mkNNList borl scale prNN0
             return $ (\t -> P.Table t 0 (const []))
                    (M.fromList $ concat $ zipWith (\(k,ts) (_,ws) -> zipWith (\(nr, t) (_,w) -> ((k, nr), t-w)) ts ws) (map (second fst) n1) (map (second fst) n0))
  errScaled <- mkErr True
  prettyErr <- if t3
               then prettyTableRows borl prettyAction prettyActionIdx (\_ x -> return x) errScaled >>= \x -> return (text "Error Scaled (R1-R0)" $+$ vcat x)
               else return empty
  let addPsiV k v = case borl ^. proxies.psiV of
        P.Table m def _ -> return $ M.findWithDefault def k m
        px -> let config = px ^?! proxyNNConfig

          in if borl ^. t <= fromIntegral (config^.replayMemoryMaxSize) && (config ^. trainBatchSize) /= 1
          then return $ M.findWithDefault 0 k (px ^. proxyNNStartup)
          else do vPsi <- P.lookupNeuralNetworkUnscaledGen P.Worker k (borl ^. proxies.psiV)
                  return (v + vPsi)

  vPlusPsiV <- prettyTableRows borl prettyAction prettyActionIdx addPsiV (borl ^. proxies.v)

  prettyRhoVal <- case borl ^. proxies.rho of
       Scalar val -> return $ text "Rho" <> colon $$ nest 45 (printFloat val)
       m  -> do
         prAct <- prettyTable borl prettyAction prettyActionIdx m
         return $ text "Rho" $+$ prAct
#ifdef DEBUG
  prettyVisits <- prettyTable borl id (const empty) (P.Table vis 0 id)
#endif
  prVW <- prBoolTblsStateAction t1 (text "V" $$ nest 40 (text "W")) (borl ^. proxies.v) (borl ^. proxies.w)
  prR0R1 <- prBoolTblsStateAction t2 (text "R0" $$ nest 40 (text "R1")) (borl ^. proxies.r0) (borl ^. proxies.r1)
  prR1 <- prettyTableRows borl prettyAction prettyActionIdx (\_ x -> return x) (borl ^. proxies.r1)
  return $
    text "\n" $+$ text "Current state" <> colon $$ nest 45 (text (show $ borl ^. s)) $+$ text "Period" <> colon $$ nest 45 (integer $ borl ^. t) $+$
    algDoc (text "Alpha" <> colon $$ nest 45 (printFloat $ borl ^. parameters . alpha)) $+$
    algDoc (text "Beta" <> colon $$ nest 45 (printFloat $ borl ^. parameters . beta)) $+$
    algDoc (text "Delta" <> colon $$ nest 45 (printFloat $ borl ^. parameters . delta)) $+$
    text "Gamma" <> colon $$ nest 45 (printFloat $ borl ^. parameters . gamma) $+$
    text "Epsilon" <> colon $$ nest 45 (printFloat $ borl ^. parameters . epsilon) $+$
    text "Exploration" <> colon $$ nest 45 (printFloat $ borl ^. parameters . exploration) $+$
    -- text "Learn From Random Actions until Expl. hits" <> colon $$ nest 45 (printFloat $ borl ^. parameters . learnRandomAbove) $+$
    nnBatchSize $+$
    nnReplMemSize $+$
    nnLearningParams $+$
    text "Algorithm" <> colon $$ nest 45 (prettyAlgorithm (borl ^. algorithm)) $+$
    algDoc (text "Xi (ratio of W error forcing to V)" <> colon $$ nest 45 (printFloat $ borl ^. parameters . xi)) $+$
    (if isAlgBorl (borl ^. algorithm) then text "Scaling (V,W,R0,R1) by V Config" <> colon $$ nest 45 scalingText else text "Scaling R1 by V Config" <> colon $$ nest 45 scalingTextDqn) $+$
    algDoc (text "Psi Rho/Psi V/Psi W" <> colon $$ nest 45 (text (show (printFloat $ borl ^. psis . _1, printFloat $ borl ^. psis . _2, printFloat $ borl ^. psis . _3)))) $+$
    algDocRho prettyRhoVal $$
    algDoc prVW $+$
    -- prettyVWPsi $+$
    (if isAlgBorl (borl ^. algorithm) then prR0R1 else vcat prR1) $+$
    -- prettyErr $+$
    algDoc (text "V+PsiV" $+$ vcat vPlusPsiV)
#ifdef DEBUG
    $+$ text "Visits [%]" $+$ prettyVisits
#endif
  where
#ifdef DEBUG
    vis = M.mapKeys (\x -> (x,0)) $ M.map (\x -> 100 * fromIntegral x / fromIntegral (borl ^. t)) (borl ^. visits)
#endif


    subtr (k, v1) (_, v2) = (k, v1 - v2)
    -- prettyAction (st, aIdx) = (st, maybe "unkown" (actionName . snd) (find ((== aIdx) . fst) (borl ^. actionList)))
    prettyAction st = st
    prettyActionIdx aIdx = text (T.unpack $ maybe "unkown" (actionName . snd) (find ((== aIdx) . fst) (borl ^. actionList)))
    scalingText =
      case borl ^. proxies.v of
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
    scalingTextDqn =
      case borl ^. proxies.v of
        P.Table {} -> text "Tabular representation (no scaling needed)"
        P.Grenade _ _ _ _ conf _ -> text (show ( (printFloat $ conf ^. scaleParameters . scaleMinR1Value, printFloat $ conf ^. scaleParameters . scaleMaxR1Value)))
        P.TensorflowProxy _ _ _ _ conf _ -> text (show ( (printFloat $ conf ^. scaleParameters . scaleMinR1Value, printFloat $ conf ^. scaleParameters . scaleMaxR1Value)))

    nnBatchSize = case borl ^. proxies.v of
      P.Table {} -> empty
      P.Grenade _ _ _ _ conf _ -> text "NN Batchsize" <> colon $$ nest 45 (int $ conf ^. trainBatchSize)
      P.TensorflowProxy _ _ _ _ conf _ -> text "NN Batchsize" <> colon $$ nest 45 (int $ conf ^. trainBatchSize)
    nnReplMemSize = case borl ^. proxies.v of
      P.Table {} -> empty
      P.Grenade _ _ _ _ conf _ -> text "NN Replay Memory size" <> colon $$ nest 45 (int $ conf ^. replayMemoryMaxSize)
      P.TensorflowProxy _ _ _ _ conf _ -> text "NN Replay Memory size" <> colon $$ nest 45 (int $ conf ^. replayMemoryMaxSize)
    nnLearningParams = case borl ^. proxies.v of
      P.Table {} -> empty
      P.Grenade _ _ _ _ conf _ -> let LearningParameters l m l2 = conf ^. grenadeLearningParams
                         in text "NN Learning Rate/Momentum/L2" <> colon $$ nest 45 (text (show (printFloat l, printFloat m, printFloat l2)))
      P.TensorflowProxy _ _ _ _ conf _ -> let LearningParameters l m l2 = conf ^. grenadeLearningParams
                         in text "NN Learning Rate/Momentum/L2" <> colon $$ nest 45 (text "Specified in tensorflow model")


prettyBORL :: (Ord s, Show s) => BORL s -> IO Doc
prettyBORL borl = runMonadBorl $ do
  buildModels
  reloadNets (borl ^. proxies.v)
  reloadNets (borl ^. proxies.w)
  reloadNets (borl ^. proxies.r0)
  reloadNets (borl ^. proxies.r1)
  prettyBORLTables True True True borl
    where reloadNets px = case px of
            P.TensorflowProxy netT netW _ _ _ _ -> restoreModelWithLastIO netT >> restoreModelWithLastIO netW
            _ -> return ()
          isTensorflowProxy P.TensorflowProxy{} = True
          isTensorflowProxy _                   = False
          buildModels = case find isTensorflowProxy (allProxies $ borl ^. proxies) of
            Just (P.TensorflowProxy netT _ _ _ _ _) -> buildTensorflowModel netT
            _                                       -> return ()

instance (Ord s, Show s) => Show (BORL s) where
  show borl = renderStyle wideStyle $ unsafePerformIO $ prettyBORL borl
