{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
module ML.ARAL.NeuralNetwork.Conversion
    ( toAgents
    , NrActions
    , NrAgents
    ) where

import qualified Data.Vector           as VB
import qualified Data.Vector.Storable  as V
import           GHC.TypeLits
import           GHC.TypeLits.KnownNat

import           ML.ARAL.Types

import           Debug.Trace


-- | Split the data into the agent vectors
toAgents :: NrActions -> NrAgents -> V.Vector Double -> Values
toAgents nrAs 1 vec
  | V.length vec /= nrAs =
    error $ "Length of network output does not fit with length of actions: " ++ show (nrAs, vec) ++ ". Check the number of output nodes! Do you use a combined network?"
  | otherwise = AgentValues (VB.singleton vec)
toAgents nrAs nr vec
  | V.length vec `mod` nr /= 0 = error $ "Undivisable length in toAgents in Conversion.hs: " ++ show (V.length vec, nr)
  | V.length vec /= nrAs * nr = error $ "Unexpected output length in toAgents. Output nodes:" ++ show len ++ " expected: " ++ show (nrAs * nr) ++ ". Check the number of output nodes! Do you use a combined network?"
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
      | length xs == nr && all ((== len) . V.length) xs = vs
      | otherwise = error $ "Number of agents does not fit number length of data in toAgents: " ++ show (nr, length xs, xs, vec)

-- -- | Create Vec from a list.
-- reifySVec :: (KnownNat nr) => [a] -> SV.Vec nr a
-- reifySVec xs = head <$> SV.iterateI tail xs
-- -- | Create SNat from Integer >= 0, otherwise SNat 0.
-- reifySNat :: Integer -> SV.SNat a
-- reifySNat n | n < 0 = reifyNat 0 (unsafeCoerce . SV.snatProxy)
--             | otherwise = reifyNat n (unsafeCoerce . SV.snatProxy)
