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

trainGrenade :: forall layers shapes nrH opt .
     (GNum (Gradients layers), NFData (Network layers shapes), NFData (Tapes layers shapes), KnownNat nrH, 'D1 nrH ~ Head shapes, SingI (Last shapes))
  => Optimizer opt
  -> Network layers shapes
  -> [((StateFeatures, ActionIndex), Float)]
  -> Network layers shapes
trainGrenade lp net chs =
  let valueMap = foldl' (\m ((inp, act), out) -> M.insertWith (++) inp [(act, out)] m) mempty chs
      inputs = M.keys valueMap
      (tapes, outputs) = unzip $ parMap rdeepseq (fromLastShapes net . runNetwork net . toHeadShapes net) inputs
      labels = zipWith (V.//) outputs (M.elems valueMap)
      gradients = zipWith3 (\tape output label -> fst $ runGradient net tape (mkLoss (toLastShapes net output) (toLastShapes net label))) tapes outputs labels
      -- applyAndMkOut grads = map (snd . fromLastShapes net . runNetwork (foldl' (applyUpdate lp) net grads)) (map (toHeadShapes net) inputs)
  in -- trace ("applyUpdate: " ++ show (applyAndMkOut gradients == applyAndMkOut [foldl1 (|+) gradients]))
    -- trace ("same? " ++ show (applyAndMkOut [(1/genericLength gradients |* foldl1 (|+) gradients)] == applyAndMkOut [(1/genericLength gradients |* foldl1 (|+) (take 2 gradients))]))
    -- foldl' (applyUpdate lp) net gradients   -- slow

    -- force $ applyUpdate lp net $ foldl1 (|+) gradients
    force $ applyUpdate lp net $ 1/genericLength gradients |* foldl1 (|+) gradients

    -- foldl' (applyUpdate lp) net $ replicate 8 $ 1/genericLength gradients |* foldl1 (|+) gradients
    -- applyUpdate lp net $ foldl1 (|+) gradients  -- better to use avg: https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent

mkLoss :: (Fractional a) => a -> a -> a
mkLoss o t = let l = o-t in signum l * l^(2::Int)

