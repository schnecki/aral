{-# LANGUAGE DataKinds    #-}
{-# LANGUAGE GADTs        #-}
{-# LANGUAGE TypeFamilies #-}
module ML.BORL.NeuralNetwork.Conversion
    ( toHeadShapes
    , toLastShapes
    , fromLastShapes
    ) where

import           Data.Singletons.Prelude.List
import qualified Data.Vector.Storable         as DV
import           GHC.TypeLits
import           Grenade
import           Numeric.LinearAlgebra.Static
-- import           Unsafe.Coerce

import           Debug.Trace

-------------------- Conversion --------------------

toHeadShapes :: (KnownNat nr, 'D1 nr ~ Head shapes) => Network layers shapes -> [Double] -> S (Head shapes)
toHeadShapes _ inp = S1D $ vector inp

toLastShapes :: (KnownNat nr, 'D1 nr ~ Last shapes) => Network layers shapes -> [Double] -> S (Last shapes)
toLastShapes _ inp = S1D $ vector inp


fromLastShapes :: Network layers shapes -> (Tapes layers shapes, S (Last shapes)) -> (Tapes layers shapes, [Double])
fromLastShapes _ (tapes, S1D out) = (tapes, DV.toList $ extract out)
fromLastShapes _ _                = error "NN output currently not supported."


-- -- | Create Vec from a list.
-- reifySVec :: (KnownNat nr) => [a] -> SV.Vec nr a
-- reifySVec xs = head <$> SV.iterateI tail xs

-- -- | Create SNat from Integer >= 0, otherwise SNat 0.
-- reifySNat :: Integer -> SV.SNat a
-- reifySNat n | n < 0 = reifyNat 0 (unsafeCoerce . SV.snatProxy)
--             | otherwise = reifyNat n (unsafeCoerce . SV.snatProxy)
