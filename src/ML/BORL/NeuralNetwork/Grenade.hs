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
  -> Maybe (MinValue Float, MaxValue Float)
  -> Network layers shapes
  -> [[((StateFeatures, [ActionIndex]), Value)]]
  -> Network layers shapes
trainGrenade opt nnConfig mMinMaxVal net chs =
  let trainIter = nnConfig ^. trainingIterations
      cropFun = maybe id (\x -> max (-x) . min x) (nnConfig ^. cropTrainMaxValScaled)
      batchGradients = parMap rdeepseq (makeGradients cropFun net) chs
      clippingRatio =
        realToFrac $
        case mMinMaxVal of
          Nothing               -> 0.01
          Just (minVal, maxVal) -> 0.01 / (maxVal - minVal)
      res =
        force $
        applyUpdate opt net $
        clipByGlobalNorm clippingRatio $
        -- 1/sum (map genericLength batchGradients) |* -- better to use avg?!!? also for RL?: https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent
        foldl1 (|+) batchGradients
   in if trainIter <= 1
        then res
        else trainGrenade opt (set trainingIterations (trainIter - 1) nnConfig) mMinMaxVal res chs

-- | Accumulate gradients for a list of n-step updates. The arrising gradients are summed up.
makeGradients ::
     forall layers shapes nrH. (GNum (Gradients layers), NFData (Tapes layers shapes), NFData (Gradients layers), KnownNat nrH, 'D1 nrH ~ Head shapes, SingI (Last shapes))
  => (Float -> Float)
  -> Network layers shapes
  -> [((StateFeatures, [ActionIndex]), Value)]
  -> Gradients layers
makeGradients _ _ [] = error "Empty list of n-step updates in NeuralNetwork.Grenade"
makeGradients cropFun net chs
  | length chs == 1 =
    head $ parMap rdeepseq (\(tape, output, label) -> fst $ runGradient net tape (mkLoss (toLastShapes net output) (toLastShapes net label))) (zip3 tapes outputs labels)
  | otherwise =
    foldl1 (|+) $ parMap rdeepseq (\(tape, output, label) -> fst $ runGradient net tape (mkLoss (toLastShapes net output) (toLastShapes net label))) (zip3 tapes outputs labels)
  where
    valueMap = foldl' (\m ((inp, acts), AgentValue outs) -> M.insertWith (++) inp (zipWith (\act out -> (act, cropFun out)) acts outs) m) mempty chs
    inputs = M.keys valueMap
    (tapes, outputs) = unzip $ parMap rdeepseq (fromLastShapesVector net . runNetwork net . toHeadShapes net) inputs
    labels = zipWith (V.//) outputs (M.elems valueMap)

runGrenade :: (KnownNat nr, Head shapes ~ 'D1 nr) => Network layers shapes -> NrAgents -> StateFeatures -> [Values]
runGrenade net nrAgents st = snd $ fromLastShapes net nrAgents $ runNetwork net (toHeadShapes net st)


mkLoss :: (Fractional a) => a -> a -> a
mkLoss o t = let l = o-t in 0.5 * signum l * l^(2::Int)
