{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
{-# LANGUAGE TypeFamilies   #-}
module ML.ARAL.InftyVector
    ( InftyVector (..)
    , getNthElement
    , toFiniteList
    ) where

import           Control.DeepSeq
import           Data.Serialize
import           GHC.Exts        (IsList (..))
import           GHC.Generics


-- | Infinite vector for which the last element is repeated endlessly.
data InftyVector a = Cons !a !(InftyVector a) | Last !a
  deriving (Show, Eq, Ord, NFData, Generic, Serialize)


instance IsList (InftyVector a) where
  type Item (InftyVector a) = a
  fromList []     = error "empty list in InftyVector IsList instance. InftyVector must be a non-empty list!"
  fromList [a]    = Last a
  fromList (a:as) = Cons a (fromList as)

  toList (Cons x xs) = x : toList xs
  toList (Last x)    = repeat x

toFiniteList :: InftyVector a -> [a]
toFiniteList (Last x)    = [x]
toFiniteList (Cons x xs) = x : toFiniteList xs

instance Foldable InftyVector where
  foldMap f (Cons x xs) = f x <> foldMap f xs
  foldMap f (Last x)    = f x

  length Last{}      = 1
  length (Cons _ xs) = 1 + length xs


instance (Num a) => Num (InftyVector a) where
  (+) xs ys = fromList $ take (max (length xs) (length ys)) $ zipWith (+) (toList xs) (toList ys)
  (-) xs ys = fromList $ take (max (length xs) (length ys)) $ zipWith (-) (toList xs) (toList ys)
  (*) xs ys = fromList $ take (max (length xs) (length ys)) $ zipWith (*) (toList xs) (toList ys)
  abs xs = fromList $ fmap abs (toList xs)
  signum xs = fromList $ fmap signum (toList xs)
  fromInteger x = Last (fromInteger x)

instance Fractional a => Fractional (InftyVector a) where
  (/) xs ys = fromList $ take (max (length xs) (length ys)) $ zipWith (/) (toList xs) (toList ys)
  fromRational x = Last (fromRational x)

instance Functor InftyVector where
  fmap f (Last x)    = Last (f x)
  fmap f (Cons x xs) = Cons (f x) (fmap f xs)

instance Applicative InftyVector where
  pure = Last
  Last f <*> Last x       = Last (f x)
  Last f <*> Cons x xs    = Cons (f x) (Last f <*> xs)
  Cons f _ <*> Last x     = Last (f x)
  Cons f fs <*> Cons x xs = Cons (f x) (fs <*> xs)


getNthElement :: (Show a) => Int -> InftyVector a -> a
getNthElement n _ | n < 0 = error $ "negative index in getNthElement: " ++ show n
getNthElement _ (Last x) = x
getNthElement 0 (Cons x _) = x
getNthElement n (Cons _ xs) = getNthElement (n - 1) xs
