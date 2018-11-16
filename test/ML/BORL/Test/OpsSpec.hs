module ML.BORL.Test.OpsSpec
  ( spec
  ) where

import           ML.BORL

import           ML.BORL.Test.Type

import           Control.Lens
import           Data.Function     (on)
import           Data.List         (groupBy, sortBy)
import qualified Data.Map.Strict   as M
import           Test.Hspec
import           Test.QuickCheck


spec :: Spec
spec = do
  describe "choosing the action" $ do
    it "if random action is chosen" $ property prop_randomAction
    it "if non-random action is chosen" $ property prop_noRandomAction
    it "tests the sorting of the actions" $ property prop_orderActions

prop_randomAction :: BORL Int -> Property
prop_randomAction borlIn = ioProperty $ fst <$> chooseAndExectueAction borl
  where borl = parameters.epsilon .~ 1.00001 $ borlIn

prop_noRandomAction :: BORL Int -> Property
prop_noRandomAction borlIn = ioProperty $ not . fst <$> chooseAndExectueAction borl
  where borl = parameters.epsilon .~ 0 $ borlIn


prop_orderActions :: BORL Int -> Property
prop_orderActions borlIn =
  ioProperty $ do
    as <- mapM (\f -> f (borl ^. s)) (borl ^. acts)
    let xs = concatMap (sortBy (compare `on` sortFunE)) $ groupBy (\x y -> epsEq (sortFunV x) (sortFunV y)) $ sortBy (compare `on` sortFunV) as
    return $ sortBy (orderActions borl) as === xs
  where
    borl = parameters . epsilon .~ 0 $ borlIn
    mV = borlIn ^. v
    mr0 = borlIn ^. r0
    mr1 = borlIn ^. r1
    (ga0,ga1) = borlIn ^. parameters.gammas
    eps = max 0.01 (borl ^. parameters.epsilon)
    sortFunV xs = sum (map (\(p, (r, s')) -> p * (r + M.findWithDefault 0 s' mV)) xs)
    sortFunE xs = sum (map (\(p, (r, s')) -> p * (r + ga1 * M.findWithDefault 0 s' mr1)) xs) - sum (map (\(p, (r, s')) -> p * (r + ga0 * M.findWithDefault 0 s' mr0)) xs)
    epsEq x y | abs (x - y) <= eps = True
              | otherwise = False

