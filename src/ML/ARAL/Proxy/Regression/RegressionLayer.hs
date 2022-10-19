{-# LANGUAGE DeriveAnyClass    #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE OverloadedStrings #-}
module ML.ARAL.Proxy.Regression.RegressionLayer
    ( Observation (..)
    , RegressionLayer (..)
    , randRegressionLayer
    , addGroundTruthValueLayer
    , trainRegressionLayer
    , applyRegressionLayer
    , prettyRegressionLayer
    , prettyRegressionLayerNoObs
    ) where

import           Control.DeepSeq
import           Control.Monad
import           Data.Default
import           Data.List                                        (foldl', foldr)
import           Data.Maybe                                       (fromMaybe)
import           Data.Serialize
import qualified Data.Vector                                      as VB
import qualified Data.Vector.Storable                             as VS
import           GHC.Generics
import           Prelude                                          hiding ((<>))
import           Statistics.Sample.WelfordOnlineMeanVariance
import           System.IO.Unsafe                                 (unsafePerformIO)
import           Text.PrettyPrint

import           ML.ARAL.NeuralNetwork.Normalisation
import           ML.ARAL.Proxy.Regression.Observation
import           ML.ARAL.Proxy.Regression.RegressionConfig
import           ML.ARAL.Proxy.Regression.RegressionNode
import           ML.ARAL.Proxy.Regression.VolatilityRegimeExpSmth
import           ML.ARAL.Types

-- | Starting periods in which the there is no regime seperation. To prevent divergence the same regime is used until the specified number of data points are collected per regime.
periodsSharedRegime :: Int
periodsSharedRegime = 10000

-- | Earliest possible training period when training starts.
periodsTrainStart :: Int
periodsTrainStart = 1000


-- | A RegressionLayer holds one node for each action.
data RegressionLayer =
  RegressionLayer
    { regressionLayerActions :: !(VB.Vector RegressionNode, Maybe (VB.Vector RegressionNode)) -- ^ One set of actions for each regime.
    , regressionInpWelford   :: !(WelfordExistingAggregate (VS.Vector Double))
    , regressionStep         :: !Int
    , regressionRegime       :: !(VB.Vector RegimeDetection)    -- Low or High variance regime
    }
  deriving (Show, Generic, Serialize, NFData)


prettyRegressionLayerWithObs :: Bool -> RegressionLayer -> Doc
prettyRegressionLayerWithObs wObs (RegressionLayer (nodesLow, mNodesHigh) welInp _ _) =
  vcat (text "Low Regime" : zipWith (\idx n -> text "Layer Node" <+> int idx $+$ prettyRegressionNode wObs (Just welInp) n) [0 ..] (VB.toList nodesLow)) $+$ mempty $+$
  maybe mempty (\nodesHigh -> vcat (text "High Regime" : zipWith (\idx n -> text "Layer Node" <+> int idx $+$ prettyRegressionNode wObs (Just welInp) n) [0 ..] (VB.toList nodesHigh))) mNodesHigh

prettyRegressionLayerNoObs :: RegressionLayer -> Doc
prettyRegressionLayerNoObs = prettyRegressionLayerWithObs False

prettyRegressionLayer :: RegressionLayer -> Doc
prettyRegressionLayer = prettyRegressionLayerWithObs True


-- | Create a new empty regression layer by providing the config, the number of nodes for the layer and the number of inputs.
randRegressionLayer :: Maybe RegressionConfig -> Int -> Int -> IO RegressionLayer
randRegressionLayer mCfg nrInput nrOutput = do
  let cfg = fromMaybe def mCfg
  nodes <- mapM (randRegressionNode cfg nrInput) [0 .. nrOutput - 1]  -- different init values for each node
  -- node <- randRegressionNode cfg nrInput 0
  -- let nodes = zipWith (\i x -> x { regNodeIndex = i}) [0..] $ replicate nrOutput node -- use same initial values for all nodes
  let mRegHigh
        | regConfigUseVolatilityRegimes cfg = Just $ VB.fromList nodes
        | otherwise = Nothing
  return $ RegressionLayer (VB.fromList nodes, mRegHigh) WelfordExistingAggregateEmpty 0 (VB.singleton def)


-- | Add ground truth values from different workers to the layer.
addGroundTruthValueLayer :: [(Observation, ActionIndex)] -> RegressionLayer -> RegressionLayer
addGroundTruthValueLayer [] lay = lay
addGroundTruthValueLayer obs (RegressionLayer nodes welInp step regime)
  | step > 0 && length obs /= VB.length regime =
    error $ "Regime length does ot fit number of observations: " ++ show (VB.length regime, length obs) ++ ". Number of parallel observations must be constant!"
  | otherwise =
    let regExp = currentRegimeExp (VB.head regime')
     in writeRegimeFile regExp `seq`
        RegressionLayer (foldr (flip updateNodes) nodes (zip [0 ..] obs)) welInp' (step + 1) regime' -- foldr as Main Agent is more important!
  where
    updateNodes nds (regId, (ob, aId)) = overRegime step (currentRegimeExp (regime' VB.! regId)) (\ns -> replaceIndex aId (addGroundTruthValueNode ob (ns VB.! aId)) ns) nds
    -- reward = obsVarianceRegimeValue $ fst $ head obs
    regime0
      | VB.length regime == length obs = regime
      | step == 0 = regime VB.++ VB.replicate (length obs - VB.length regime) (VB.head regime)
      | otherwise = error "addGroundTruthValueNode: Should not happen!"
    regime' = VB.zipWith (\reg -> addValueToRegime reg . obsVarianceRegimeValue . fst) regime0 (VB.fromList obs)
    welInp'
      | step < 30000 = foldl' addValue welInp (map (obsInputValues . fst) obs)
      | otherwise = welInp
    replaceIndex idx x xs = xs VB.// [(idx, x)]
    getBorder
      | step < 10 = const 0
      | otherwise = (\(mean, _, x) -> mean + sqrt x) . finalize
    writeRegimeFile reg =
      unsafePerformIO $ do
        let txt = show step ++ "\t" ++ show (obsVarianceRegimeValue $ fst $ head obs) ++ "\t" ++ show (fromEnum reg) ++ "\t" ++ show (regimeExpSmthFast $ VB.head regime') ++
              "\t" ++ show (regimeExpSmthSlow $ VB.head regime') ++ "\t" ++ show (getBorder $ regimeWelfordAll $ VB.head regime') ++ "\n"
        when (step == 0) $ do writeFile "regime" $ "period\treward\tregime\tExpFast\tExpSlow\tBorder\n"
        appendFile "regime" txt


-- TODO
-- data RegFunction = RegLinear
--   deriving (Eq, Ord, Show, Generic, Serialize, NFData)
-- costFunction :: RegFunction -> VB.Vector Double -> VS.Vector Double -> Double
-- costFunction RegLinear coefs inps =
--   VS.sum (VS.zipWith (*) (VB.convert coefs) inps) + VB.last coefs


-- | Train regression layger (= all nodes).
trainRegressionLayer :: RegressionLayer -> RegressionLayer
trainRegressionLayer (RegressionLayer nodes welInp step regime)
  | step < periodsTrainStart = RegressionLayer nodes welInp step regime -- only used for learning the normalization
  | otherwise = RegressionLayer (overBothRegimes (\ns -> VB.map (trainRegressionNode welInp (VB.length ns, nrWorkers) step) ns) nodes) welInp step regime
  where nrWorkers = VB.length regime


-- | Apply regression layer to given inputs
applyRegressionLayer :: Int -> RegressionLayer -> ActionIndex -> VS.Vector Double -> Double
applyRegressionLayer regId (RegressionLayer nodes welInp step regime) actIdx stateFeat =
  preventDivergence $ withRegime step (currentRegimeExp (regime VB.! regId')) (\ns -> applyRegressionNode (ns VB.! actIdx) (normaliseStateFeatureUnbounded welInp stateFeat)) nodes
  where
    regId'
      | regId < VB.length regime = regId -- might no be available on first iteration(s)
      | otherwise = 0
    preventDivergence
      | step < periodsSharedRegime = min 10 . max (-10)
      | otherwise = id


-- Regime helpers

-- | Apply a function to a specific regime (Low, High).
overRegime :: Int -> Regime -> (VB.Vector RegressionNode -> VB.Vector RegressionNode) -> (VB.Vector RegressionNode, Maybe (VB.Vector RegressionNode)) -> (VB.Vector RegressionNode, Maybe (VB.Vector RegressionNode))
overRegime step _ f (nodesLow, nodesHigh)
  | step < periodsSharedRegime = (f nodesLow, f <$> nodesHigh)
overRegime _ _ f (nodesLow, Nothing)  = (f nodesLow, Nothing)
overRegime _ Low f (nodesLow,  mNodesHigh)  = (f nodesLow, mNodesHigh)
overRegime _ High f (nodesLow, Just nodesHigh) = (nodesLow, Just $ f nodesHigh)

-- | Apply a function to the right regime.
withRegime :: Int -> Regime -> (VB.Vector RegressionNode -> a) -> (VB.Vector RegressionNode, Maybe (VB.Vector RegressionNode)) -> a
withRegime step _ f (nodesLow, _)
  | step < periodsSharedRegime = f nodesLow
withRegime _ _ f (nodesLow, Nothing)   = f nodesLow
withRegime _ Low f (nodesLow, _)   = f nodesLow
withRegime _ High f (_, Just nodesHigh) = f nodesHigh


-- | Apply a function over both regimes and update regimes.
overBothRegimes ::(VB.Vector RegressionNode -> VB.Vector RegressionNode) -> (VB.Vector RegressionNode, Maybe (VB.Vector RegressionNode)) -> (VB.Vector RegressionNode, Maybe (VB.Vector RegressionNode))
overBothRegimes f (nodesLow, nodesHigh) = (f nodesLow, f <$> nodesHigh)
