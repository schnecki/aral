{-# LANGUAGE DataKinds    #-}
{-# LANGUAGE GADTs        #-}
{-# LANGUAGE TypeFamilies #-}

module ML.BORL.NeuralNetwork.Training
    ( trainNetwork
    ) where

import           ML.BORL.NeuralNetwork.Conversion


import           Data.List                        (foldl')
import           Data.Singletons.Prelude.List
import           GHC.TypeLits
import           Grenade

import           Debug.Trace

trainNetwork :: (KnownNat nrH, KnownNat nrL, 'D1 nrH ~ Head shapes, 'D1 nrL ~ Last shapes) => LearningParameters -> Network layers shapes -> [([Double], Double)] -> Network layers shapes
trainNetwork lp net chs = foldl' (applyUpdate lp) net $ zipWith mkGradients chs $ tapesAndActual chs
  where
    tapesAndActual = map runForward
    runForward (inp, _) = fromLastShapes net $ runNetwork net (toHeadShapes net inp)
    mkGradients (inp, target) (tape, output) =
      trace ("inp/target/output: " ++ show (inp, target, output)) $
      fst $ runGradient net tape (mkLoss (toLastShapes net output) (toLastShapes net [max (-0.9) $ min 0.9 target]))


mkLoss :: (Fractional a) => a -> a -> a
mkLoss o t = let l = o-t in signum l * l^(2::Int)

