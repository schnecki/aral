{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Strict              #-}
{-# LANGUAGE TypeFamilies        #-}

module ML.BORL.NeuralNetwork.Grenade
    ( trainGrenade
    ) where


import           Control.DeepSeq
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
import           ML.BORL.Types


import           Debug.Trace

trainMaxVal :: Float
trainMaxVal = 0.98

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
  -> Int
  -> Maybe (MinValue Float, MaxValue Float)
  -> Network layers shapes
  -> [[((StateFeatures, ActionIndex), Float)]]
  -> Network layers shapes
trainGrenade opt trainIter mMinMaxVal net chs =
  let batchGradients = parMap rdeepseq (makeGradients net) chs
      clippingRatio =
        realToFrac $
        case mMinMaxVal of
          Nothing               -> 0.01
          Just (minVal, maxVal) -> 0.01 / (maxVal - minVal)
      -- applyAndMkOut grads = map (snd . fromLastShapes net . runNetwork (foldl' (applyUpdate opt) net grads)) (map (toHeadShapes net) inputs)
     -- trace ("applyUpdate: " ++ show (applyAndMkOut gradients == applyAndMkOut [foldl1 (|+) gradients]))
      -- trace ("same? " ++ show (applyAndMkOut [(1/genericLength gradients |* foldl1 (|+) gradients)] == applyAndMkOut [(1/genericLength gradients |* foldl1 (|+) (take 2 gradients))]))
    -- foldl' (applyUpdate opt) net gradients   -- slow
      res = force $ applyUpdate opt net $ clipByGlobalNorm clippingRatio $
        -- 1/sum (map genericLength chs) |* -- better to use avg: https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent
        foldl1 (|+) batchGradients
   in if trainIter <= 1
        then res
        else trainGrenade opt (trainIter - 1) mMinMaxVal res chs
    -- force $ applyUpdate opt net $ 1/genericLength batchGradients |* foldl1 (|+) batchGradients
    -- foldl' (applyUpdate opt) net $ replicate 8 $ 1/genericLength gradients |* foldl1 (|+) gradients
    -- applyUpdate opt net $ foldl1 (|+) gradients  -- better to use avg: https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent

-- | Accumulate gradients for a list of n-step updates. The arrising gradients are summed up.
makeGradients ::
     forall layers shapes nrH. (GNum (Gradients layers), NFData (Tapes layers shapes), NFData (Gradients layers), KnownNat nrH, 'D1 nrH ~ Head shapes, SingI (Last shapes))
  => Network layers shapes
  -> [((StateFeatures, ActionIndex), Float)]
  -> Gradients layers
makeGradients _ [] = error "Empty list of n-step updates in NeuralNetwork.Grenade"
makeGradients net chs
  | length chs == 1 =
    head $ parMap rdeepseq (\(tape, output, label) -> fst $ runGradient net tape (mkLoss (toLastShapes net output) (toLastShapes net label))) (zip3 tapes outputs labels)
  | otherwise
    -- (1 / genericLength chs |*) $
   = foldl1 (|+) $ parMap rdeepseq (\(tape, output, label) -> fst $ runGradient net tape (mkLoss (toLastShapes net output) (toLastShapes net label))) (zip3 tapes outputs labels)
  where
    valueMap = foldl' (\m ((inp, act), out) -> M.insertWith (++) inp [(act, max (-trainMaxVal) $ min trainMaxVal out)] m) mempty chs
    inputs = M.keys valueMap
    (tapes, outputs) = unzip $ parMap rdeepseq (fromLastShapes net . runNetwork net . toHeadShapes net) inputs
    labels = zipWith (V.//) outputs (M.elems valueMap)


mkLoss :: (Fractional a) => a -> a -> a
mkLoss o t = let l = o-t in 0.5 * signum l * l^(2::Int)

