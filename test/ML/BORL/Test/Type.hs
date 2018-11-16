{-# LANGUAGE FlexibleInstances #-}

module ML.BORL.Test.Type where

import           Test.QuickCheck

import           ML.BORL.Test.Parameters

import           ML.BORL.Type

instance Arbitrary (BORL Int) where
  arbitrary = sized $ \n -> do
    let maxS = max 1 (min (10*n) 1000)
    init <- choose (0, maxS)
    let states = [0..maxS]
    refState <- choose (0, maxS)
    params <- arbitrary
    decayFunction <- arbitrary
    let action p r s' s = return $ scaleProb $ map (const (p, (r, max 0 (min maxS s')))) [1..n+1]
          where scaleProb xs = map (scale (sum $ map fst xs)) xs
                scale v (p,x) = (p/v,x)
    let mkAction = do
          p <- choose (0.001,1)
          r <- arbitrary
          s' <- choose (0, maxS)
          return $ action p r s'

    nActs <- choose (1,maxS)
    as <- mapM (const mkAction) [1..nActs]
    return $ mkBORL init as refState params decayFunction

instance CoArbitrary (BORL Int) where
  coarbitrary (BORL _ s r t par dec rho v w r0 r1) = variant 0 . coarbitrary ((s, r, t), (par, dec, rho, v, w), r0, r1)
