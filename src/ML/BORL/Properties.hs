module ML.BORL.Properties
    ( isUnichain
    , isMultichain
    ) where

import           ML.BORL.Type

import           Control.Lens ((^.))

-------------------- Properties --------------------


isMultichain :: BORL s -> Bool
isMultichain borl =
  case borl ^. rho of
    Left {}  -> False
    Right {} -> True


isUnichain :: BORL s -> Bool
isUnichain = not . isMultichain


