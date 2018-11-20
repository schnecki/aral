{-# LANGUAGE BangPatterns #-}

module ML.BORL.Proxy
    ( Proxy (..)
    , insert
    , findWithDefault
    ) where


import qualified Data.Map.Strict as M

-- import           Grenade

data Proxy k = Table !(M.Map k Double)
             -- | NN (Network layers shapes)

-- | Insert (or update) a value.
insert :: (Ord k) => k -> Double -> Proxy k -> Proxy k
insert k v (Table m) = Table (M.insert k v m)


-- | Retrieve a value.
findWithDefault :: (Ord k) => Double -> k -> Proxy k -> Double
findWithDefault def k (Table m) = M.findWithDefault def k m
