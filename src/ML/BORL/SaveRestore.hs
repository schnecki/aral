{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Unsafe              #-}


module ML.BORL.SaveRestore where

import           ML.BORL.NeuralNetwork
import           ML.BORL.Proxy.Type
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Lens
import           Control.Monad
import           Data.List             (find)


saveTensorflowModels :: (MonadBorl' m) => BORL s -> m (BORL s)
saveTensorflowModels borl = do
  mapM_ saveProxy (allProxies $ borl ^. proxies)
  return borl
  where
    saveProxy px =
      case px of
        TensorflowProxy netT netW _ _ _ _ -> saveModelWithLastIO netT >> saveModelWithLastIO netW >> return ()
        _ -> return ()

type BuildModels = Bool

restoreTensorflowModels :: (MonadBorl' m) => BuildModels -> BORL s -> m ()
restoreTensorflowModels build borl = do
  when build buildModels
  mapM_ restoreProxy (allProxies $ borl ^. proxies)
  where
    restoreProxy px =
      case px of
        TensorflowProxy netT netW _ _ _ _ -> restoreModelWithLastIO netT >> restoreModelWithLastIO netW >> return ()
        _ -> return ()
    buildModels =
      case find isTensorflow (allProxies $ borl ^. proxies) of
        Just (TensorflowProxy netT _ _ _ _ _) -> buildTensorflowModel netT
        _                                     -> return ()
