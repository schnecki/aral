{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE ExplicitForAll            #-}
{-# LANGUAGE GADTs                     #-}

module ML.BORL.Proxy
    ( Proxy (..)
    , insert
    , findWithDefault
    ) where


import           ML.BORL.NeuralNetwork

import qualified Data.Map.Strict              as M
import           Data.Singletons.Prelude.List
import           GHC.TypeLits
import           Grenade

type Cache k = [(k,Double)]

data Proxy k = Table !(M.Map k Double)
             | forall layers shapes nr . (KnownNat nr, Head shapes ~ 'D1 nr) => NN (Network layers shapes) (NNConfig k)

data NNConfig k = NNConfig
  { toNetInp :: k -> [Double]
  , cache    :: Cache k
  }

-- | Insert (or update) a value.
insert :: (Ord k) => k -> Double -> Proxy k -> Proxy k
insert k v (Table m)                     = Table (M.insert k v m)
insert k v (NN net (NNConfig toInp chs)) = NN net (NNConfig toInp ((k, v) : chs))


-- | Retrieve a value.
findWithDefault :: (Ord k) => Double -> k -> Proxy k -> Double
findWithDefault def k (Table m) = M.findWithDefault def k m
findWithDefault _ k (NN net conf) = head $ snd $ fromLastShapes net $ runNetwork net (toHeadShapes net $ toNetInp conf k)


