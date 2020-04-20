{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
module ML.BORL.NeuralNetwork.Conversion
    ( toHeadShapes
    , toLastShapes
    , fromLastShapes
    ) where

import           Data.Singletons
import           Data.Singletons.Prelude.List
import qualified Data.Vector.Storable         as V
import           GHC.TypeLits
import           Grenade
import           Numeric.LinearAlgebra.Static

import           ML.BORL.Types

import           Debug.Trace

-------------------- Conversion --------------------

toHeadShapes :: (KnownNat nr, 'D1 nr ~ Head shapes) => Network layers shapes -> StateFeatures -> S (Head shapes)
toHeadShapes _ inp = S1D $ vector $ map realToFrac $ V.toList inp

toLastShapes :: forall layers shapes . (SingI (Last shapes)) => Network layers shapes -> StateFeatures -> S (Last shapes)
toLastShapes _ inp =
  case (sing :: Sing (Last shapes)) of
    D1Sing SNat           -> S1D $ vector $ map realToFrac $ V.toList inp
    D2Sing SNat SNat      -> S2D $ tr $ matrix $ map realToFrac $ V.toList inp -- transpose as it is read in row major, but we got column major
    D3Sing SNat SNat SNat -> S3D $ tr $ matrix $ map realToFrac $ V.toList inp -- transpose as it is read in row major, but we got column major


fromLastShapes :: Network layers shapes -> (Tapes layers shapes, S (Last shapes)) -> (Tapes layers shapes, V.Vector Float)
fromLastShapes _ (tapes, S1D out)    = (tapes, V.map realToFrac $ extract out)
fromLastShapes _ (tapes, S2D mat) = (tapes, V.concat $ map (V.map realToFrac . extract) $ toColumns mat)
fromLastShapes _ _ = error "3D output not supported"

-- -- | Create Vec from a list.
-- reifySVec :: (KnownNat nr) => [a] -> SV.Vec nr a
-- reifySVec xs = head <$> SV.iterateI tail xs
-- -- | Create SNat from Integer >= 0, otherwise SNat 0.
-- reifySNat :: Integer -> SV.SNat a
-- reifySNat n | n < 0 = reifyNat 0 (unsafeCoerce . SV.snatProxy)
--             | otherwise = reifyNat n (unsafeCoerce . SV.snatProxy)
