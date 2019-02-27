module ML.BORL.Properties
    ( isUnichain
    , isMultichain
    ) where

import           ML.BORL.Proxy.Type
import           ML.BORL.Type

import           Control.Lens       ((^.))

-------------------- Properties --------------------


isMultichain :: BORL s -> Bool
isMultichain borl =
  case borl ^. proxies.rho of
    Scalar {} -> False
    _         -> True


isUnichain :: BORL s -> Bool
isUnichain = not . isMultichain


