-- file test/Main.hs
module Main where

import           Prelude
import qualified Spec
import           Test.Hspec.Formatters
import           Test.Hspec.Runner


main :: IO ()
main = hspecWith defaultConfig Spec.spec
