module ML.ARAL.Properties
    ( isUnichain
    , isMultichain
    , isAnn
    ) where

import           ML.ARAL.Proxy.Proxies
import           ML.ARAL.Proxy.Type
import           ML.ARAL.Type

import           Control.Lens          ((^.))

-------------------- Properties --------------------


isMultichain :: BORL s as -> Bool
isMultichain borl =
  case borl ^. proxies.rho of
    Scalar {} -> False
    _         -> True


isUnichain :: BORL s as -> Bool
isUnichain = not . isMultichain


isAnn :: BORL s as -> Bool
isAnn borl = any isNeuralNetwork (allProxies $ borl ^. proxies)
