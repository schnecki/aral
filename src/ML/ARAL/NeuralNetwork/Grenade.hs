{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}

module ML.ARAL.NeuralNetwork.Grenade
    ( trainGrenade
    , runGrenade
    ) where

import           Control.DeepSeq
import           Control.Lens                     (set, (^.))
import           Control.Parallel.Strategies
import           Data.List                        (genericLength)
import           Data.Singletons
import           Data.Singletons.Prelude.List
import qualified Data.Vector.Storable             as V
import           GHC.TypeLits
import           Grenade

import           ML.ARAL.NeuralNetwork.Conversion
import           ML.ARAL.NeuralNetwork.NNConfig
import           ML.ARAL.Types


-- | Train grenade network.
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
  => Period
  -> Optimizer opt
  -> NNConfig
  -> Network layers shapes
  -> [[((StateFeatures, ActionIndex), Double)]]
  -> IO (Network layers shapes)
trainGrenade period opt nnConfig net chs = do
  let trainIter = nnConfig ^. trainingIterations
      cropFun = maybe id (\x -> max (-x) . min x) (nnConfig ^. cropTrainMaxValScaled)
      ~batchGradients = parMap (rparWith rdeepseq) (makeGradients cropFun net) chs
      ~clipGrads =
        case nnConfig ^. clipGradients of
          NoClipping         -> id
          ClipByGlobalNorm v -> clipByGlobalNorm v
          ClipByValue v      -> clipByValue v
      ~res =
        clipGrads $
        (1 / sum (map genericLength chs)) |* -- avg prevents divergance on huge networks
        sumG batchGradients
  -- let !net' = force $ applyUpdate opt net res
  let !net' = applyUpdate opt net res
  if trainIter <= 1
    then return net'
    else trainGrenade period opt (set trainingIterations (trainIter - 1) nnConfig) net' chs

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
    sumG $ parMap (rparWith rdeepseq) (\(tape, output, label) -> fst $ runGradient net tape (mkLoss (toLastShapes net output) (toLastShapes net label))) (zip3 tapes outputs labels)
  where
    inputs = map (fst . fst) chs
    (tapes, outputs) = unzip $ parMap (rparWith rdeepseq) (fromLastShapesVector net . runNetwork net . toHeadShapes net) inputs
    labels = zipWith (V.//) outputs (map (\((_, act), out) -> [(act, cropFun out)]) chs)

runGrenade :: (KnownNat nr, Head shapes ~ 'D1 nr) => Network layers shapes -> NrActions -> NrAgents -> StateFeatures -> [Values]
runGrenade net nrAs nrAgents st = snd $ fromLastShapes net nrAs nrAgents $ runNetwork net (toHeadShapes net st)

mkLoss :: (Fractional a) => a -> a -> a
mkLoss o t =
  let l = o - t
   in signum l * l ^ (2 :: Int)
