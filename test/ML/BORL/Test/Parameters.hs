
module ML.BORL.Test.Parameters where


import           Test.QuickCheck

import           ML.BORL.Parameters

instance Arbitrary Parameters where
  arbitrary = Parameters <$> ar <*> ar <*> ar <*> ar <*> ar <*> ar <*> ar <*> ar
    where
      ar = choose (0, 1)
      -- tpl = do
      --   x <- ar
      --   y <- ar
      --   if x < y
      --     then return (x, y)
      --     else if x == y
      --            then return (max 0 (x - 0.05), min 1 (x + 0.05))
      --            else return (y, x)

instance CoArbitrary Parameters where
  coarbitrary (Parameters a b c d e l x z) = variant 0 . coarbitrary ((a, b, c, d), (e, l, z, x))


