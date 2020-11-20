{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Strict              #-}
{-# LANGUAGE TypeFamilies        #-}

module ML.BORL.NeuralNetwork.Grenade
    ( trainGrenade
    , runGrenade
    ) where


import           Control.DeepSeq
import           Control.Lens                     (set, (^.))
import           Control.Parallel.Strategies
import           Data.List                        (foldl', foldl1, genericLength)
import qualified Data.Map.Strict                  as M
import           Data.Proxy
import           Data.Singletons
import           Data.Singletons.Prelude.List
import qualified Data.Vector.Storable             as V
import           GHC.TypeLits
import           Grenade

import           ML.BORL.NeuralNetwork.Conversion
import           ML.BORL.NeuralNetwork.NNConfig
import           ML.BORL.Types


import           Debug.Trace

trainGrenade ::
     forall layers shapes nrH opt.
     ( GNum (Gradients layers)
     , NFData (Network layers shapes)
     , NFData (Tapes layers shapes)
     , KnownNat nrH
     , 'D1 nrH ~ Head shapes
     , SingI (Last shapes)
     , NFData (Gradients layers)
     , FoldableGradient (Gradients layers)
     )
  => Optimizer opt
  -> NNConfig
  -> Maybe (MinValue Double, MaxValue Double)
  -> Network layers shapes
  -> [[((StateFeatures, ActionIndex), Double)]]
  -> Network layers shapes
trainGrenade opt nnConfig mMinMaxVal net chs =
  let trainIter = nnConfig ^. trainingIterations
      cropFun = maybe id (\x -> max (-x) . min x) (nnConfig ^. cropTrainMaxValScaled)
      batchGradients = parMap (rparWith rdeepseq) (makeGradients cropFun net) chs
      clippingRatio =
        case mMinMaxVal of
          Nothing               -> 0.01
          Just (minVal, maxVal) -> 0.01 / (maxVal - minVal)
      clipGrads | nnConfig ^. clipGradients = clipByGlobalNorm clippingRatio
                | otherwise = id
      -- res = foldl' (applyUpdate opt) net batchGradients
      res =
        -- force $
        applyUpdate opt net $
        clipGrads $
        -- 1/sum (map genericLength batchGradients) |* -- better to use avg?!!? also for RL?: https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent
        -- foldl1 (zipVectorsWithInPlaceReplSnd (+)) batchGradients
        sumG batchGradients -- foldl1 (|+) batchGradients
   in if trainIter <= 1
        then res
        else trainGrenade opt (set trainingIterations (trainIter - 1) nnConfig) mMinMaxVal res chs

-- | Accumulate gradients for a list of n-step updates. The arrising gradients are summed up.
makeGradients ::
     forall layers shapes nrH. (GNum (Gradients layers), NFData (Tapes layers shapes), NFData (Gradients layers), KnownNat nrH, 'D1 nrH ~ Head shapes, SingI (Last shapes))
  => (Double -> Double)
  -> Network layers shapes
  -> [((StateFeatures, ActionIndex), Double)]
  -> Gradients layers
makeGradients _ _ [] = error "Empty list of n-step updates in NeuralNetwork.Grenade"
makeGradients cropFun net chs
  | length chs == 1 =
    head $ parMap (rparWith rdeepseq) (\(tape, output, label) -> fst $ runGradient net tape (mkLoss (toLastShapes net output) (toLastShapes net label))) (zip3 tapes outputs labels)
  | otherwise =
    -- foldl1 (zipVectorsWithInPlaceReplSnd (+)) $
    -- foldl1 (|+) $
    sumG $
    parMap (rparWith rdeepseq) (\(tape, output, label) -> fst $ runGradient net tape (mkLoss (toLastShapes net output) (toLastShapes net label))) (zip3 tapes outputs labels)
  where
    -- valueMap = foldl' (\m ((inp, acts), AgentValue outs) -> M.insertWith (++) inp (zipWith (\act out -> (act, cropFun out)) acts outs) m) mempty chs
    valueMap = foldl' (\m ((inp, act), out) -> M.insertWith (++) inp [(act, cropFun out)] m) mempty chs
    inputs = M.keys valueMap
    (tapes, outputs) = unzip $ parMap (rparWith rdeepseq) (fromLastShapesVector net . runNetwork net . toHeadShapes net) inputs
    labels = zipWith (V.//) outputs (M.elems valueMap)

runGrenade :: (KnownNat nr, Head shapes ~ 'D1 nr) => Network layers shapes -> NrAgents -> StateFeatures -> [Values]
runGrenade net nrAgents st = snd $ fromLastShapes net nrAgents $ runNetwork net (toHeadShapes net st)


mkLoss :: (Show a, Fractional a) => a -> a -> a
mkLoss o t =

  let l = o-t in
     -- trace ("o: " ++ show o ++ "\n" ++
     -- "t: " ++ show t ++ "\n" ++
     -- "l: " ++ show l ++ "\n" ++
     -- "r: " ++ show (0.5 * signum l * l^(2::Int))) undefined $


    0.5 * signum l * l^(2::Int)
