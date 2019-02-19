{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE ExplicitForAll            #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TemplateHaskell           #-}
{-# LANGUAGE UndecidableInstances      #-}

module ML.BORL.Proxy.Ops
    ( mkNNList
    ) where


import           ML.BORL.NeuralNetwork
import           ML.BORL.Proxy.Type
import           ML.BORL.Type
import           ML.BORL.Types         as T

import           Control.Lens


-- | This function loads the model from the checkpoint file and finds then retrieves the data.
mkNNList :: BORL k -> Bool -> Proxy k -> T.MonadBorl [(k, ([Double], [Double]))]
mkNNList borl scaled pr =
  mapM
    (\st -> do
       let fil = actFilt st
           filterActions xs = map snd $ filter fst $ zip fil xs
       t <-
         if scaled
           then lookupActionsNeuralNetwork Target st pr
           else lookupActionsNeuralNetworkUnscaled Target st pr
       w <-
         if scaled
           then lookupActionsNeuralNetwork Worker st pr
           else lookupActionsNeuralNetworkUnscaled Worker st pr

       return (st, (filterActions t, filterActions w)))
    (conf ^. prettyPrintElems)
  where
    conf =
      case pr of
        Grenade _ _ _ _ conf _         -> conf
        TensorflowProxy _ _ _ _ conf _ -> conf
        _                              -> error "mkNNList called on non-neural network"
    actIdxs = [0 .. _proxyNrActions pr]
    actFilt = borl ^. actionFilter

