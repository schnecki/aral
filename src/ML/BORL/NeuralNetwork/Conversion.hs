{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
module ML.BORL.NeuralNetwork.Conversion
    ( toHeadShapes
    , toLastShapes
    , fromLastShapes
    , fromLastShapesVector
    , NrAgents
    ) where

import           Data.Singletons
import           Data.Singletons.Prelude.List
import           Data.Singletons.TypeLits
import qualified Data.Vector                  as VB
import qualified Data.Vector.Storable         as V
import           GHC.TypeLits
import           Grenade
import           Numeric.LinearAlgebra.Static

import           ML.BORL.Types

import           Debug.Trace

-------------------- Conversion --------------------

toHeadShapes :: (KnownNat nr, 'D1 nr ~ Head shapes) => Network layers shapes -> StateFeatures -> S (Head shapes)
toHeadShapes _ inp = S1DV inp

toLastShapes :: forall layers shapes . (SingI (Last shapes)) => Network layers shapes -> StateFeatures -> S (Last shapes)
toLastShapes _ inp =
  case (sing :: Sing (Last shapes)) of
    D1Sing SNat           -> fromRowMajorVectorToSD1V inp
    D2Sing SNat SNat      -> fromRowMajorVectorToSD2V inp
    D3Sing SNat SNat SNat -> S3D $ tr $ matrix $ V.toList inp -- transpose as it is read in row major, but we got column major

type NrAgents = Int

-- | Get the ANN output as a vector.
fromLastShapesVector :: Network layers shapes -> (Tapes layers shapes, S (Last shapes)) -> (Tapes layers shapes, V.Vector Double)
fromLastShapesVector _ (tapes, S1DV out)   = (tapes, out)
fromLastShapesVector _ (tapes, S1D out)    = (tapes, extract out)
fromLastShapesVector _ (tapes, val@S2DV{}) = (tapes, V.concat $ toColumnsS2D val)
fromLastShapesVector _ (tapes, val@S2D{})  = (tapes, V.concat $ toColumnsS2D val)
fromLastShapesVector _ _                   = error "3D output not supported"

-- fromLastShapesVector :: Network layers shapes -> (Tapes layers shapes, S (Last shapes)) -> (Tapes layers shapes, V.Vector Float)
-- fromLastShapesVector _ (tapes, S1D out)    = (tapes, V.map realToFrac $ extract out)
-- fromLastShapesVector _ (tapes, S2D mat) = (tapes, V.concat $ map (V.map realToFrac . extract) (toColumns mat))


fromLastShapes :: Network layers shapes -> NrAgents -> (Tapes layers shapes, S (Last shapes)) -> (Tapes layers shapes, [Values])
fromLastShapes _ nrAgents (tapes, S1DV val)    = (tapes, [toAgents nrAgents $ val])
fromLastShapes _ nrAgents (tapes, S1D out)    = (tapes, [toAgents nrAgents $ extract out])
fromLastShapes _ nrAgents (tapes, val@S2DV{}) = (tapes, map (toAgents nrAgents) (toColumnsS2D val))
fromLastShapes _ nrAgents (tapes, val@S2D{}) = (tapes, map (toAgents nrAgents) (toColumnsS2D val))
fromLastShapes _ _ _ = error "3D output not supported"

-- | Split the data into the agent vectors
toAgents :: Int -> V.Vector Double -> Values
toAgents 1 vec = AgentValues (VB.singleton vec)
toAgents nr vec
  | V.length vec `mod` nr /= 0 = error $ "undivisable length in toAgents in Conversion.hs: " ++ show (V.length vec, nr)
  | otherwise =
#ifdef DEBUG
    checkAgents $
#endif
    AgentValues $ toAgents' vec
  where
    len = V.length vec `div` nr
    toAgents' v
      | V.null v = VB.empty
      | otherwise =
        let (this, that) = V.splitAt len v
        in this `VB.cons` toAgents' that
    checkAgents :: Values -> Values
    checkAgents vs@(AgentValues xs)
      | length xs == nr = vs
      | otherwise = error $ "Number of agents does not fit number length of data in toAgents: " ++ show (nr, length xs, xs, vec)

-- -- | Create Vec from a list.
-- reifySVec :: (KnownNat nr) => [a] -> SV.Vec nr a
-- reifySVec xs = head <$> SV.iterateI tail xs
-- -- | Create SNat from Integer >= 0, otherwise SNat 0.
-- reifySNat :: Integer -> SV.SNat a
-- reifySNat n | n < 0 = reifyNat 0 (unsafeCoerce . SV.snatProxy)
--             | otherwise = reifyNat n (unsafeCoerce . SV.snatProxy)
