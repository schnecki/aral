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
import qualified Data.Map.Strict       as M

-- | This function loads the model from the checkpoint file and finds then retrieves the data.
mkNNList :: (Ord k, Eq k) => BORL k -> Bool -> Proxy k -> T.MonadBorl [(k, ([(ActionIndex, Double)], [(ActionIndex, Double)]))]
mkNNList borl scaled pr =
  mapM
    (\st -> do
       let fil = actFilt st
           filterActions xs = map (\(_, a, b) -> (a, b)) $ filter (\(f, _, _) -> f) $ zip3 fil [(0 :: Int) ..] xs
       t <-
         if useTable
           then return $ lookupTable scaled st
           else if scaled
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
    useTable = False && borl ^. t == fromIntegral (_proxyNNConfig pr ^?! replayMemory . replayMemorySize)
    lookupTable scale st
      | scale = val -- values are being unscaled, thus let table value be unscaled
      | otherwise = map (scaleValue (getMinMaxVal pr)) val
      where
        val = map (\actNr -> M.findWithDefault 0 (st, actNr) (_proxyNNStartup pr)) [0 .. _proxyNrActions pr]
          -- map snd $ M.toList $ M.filterWithKey (\(x, _) _ -> x == st) (_proxyNNStartup pr)
