{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TupleSections       #-}
{-# LANGUAGE TypeFamilies        #-}

module ML.BORL.Type where

import           ML.BORL.Action
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import           ML.BORL.Proxy.Type
import           ML.BORL.Types

import           Control.DeepSeq
import           Control.Lens
import           Control.Monad                (zipWithM)
import           Control.Monad.IO.Class       (MonadIO, liftIO)
import qualified Data.Map.Strict              as M
import qualified Data.Proxy                   as Type
import           Data.Singletons.Prelude.List
import qualified Data.Vector.Mutable          as V
import           GHC.TypeLits
import           Grenade
import qualified TensorFlow.Core              as TF
import qualified TensorFlow.Session           as TF


type ActionIndexed s = (ActionIndex, Action s) -- ^ An action with index.
type Decay = Period -> Parameters -> Parameters -- ^ Function specifying the decay of the parameters at time t.


-------------------- Main RL Datatype --------------------


data BORL s = BORL
  { _actionList    :: ![ActionIndexed s]    -- ^ List of possible actions in state s.
  , _actionFilter  :: !(s -> [Bool])        -- ^ Function to filter actions in state s.
  , _s             :: !s                    -- ^ Current state.
  , _sRef          :: !(Maybe (s,ActionIndex)) -- ^ Reference state.
  , _t             :: !Integer              -- ^ Current time t.
  , _parameters    :: !Parameters           -- ^ Parameter setup.
  , _decayFunction :: !Decay                -- ^ Decay function at period t.

  -- discount factors
  , _gammas        :: !(Double, Double) -- ^ Two gamma values in ascending order

  -- Values:
  , _rho           :: !(Either Double (Proxy s)) -- ^ Either unichain or multichain y_{-1} values.
  , _psis          :: !(Double, Double, Double)                -- ^ Exponentially smoothed psi values.
  , _v             :: !(Proxy s)                 -- ^ Bias values (y_0).
  , _w             :: !(Proxy s)                 -- ^ y_1 values.
  , _r0            :: !(Proxy s)                 -- ^ Discounted values with first gamma value.
  , _r1            :: !(Proxy s)                 -- ^ Discounted values with second gamma value.

  -- Stats:
  , _visits        :: !(M.Map s Integer)                       -- ^ Counts the visits of the states
  }
makeLenses ''BORL

instance NFData s => NFData (BORL s) where
  rnf (BORL as af s sRef t par dec gam rho psis v w r r1 vis) = rnf as `seq` rnf af `seq` rnf s `seq` rnf sRef `seq` rnf t `seq` rnf par `seq` rnf dec `seq` rnf gam `seq` rnf rho `seq` rnf psis `seq` rnf v `seq` rnf w `seq` rnf r `seq` rnf r1 `seq` rnf s


default_gamma0, default_gamma1 :: Double
default_gamma0 = 0.25
default_gamma1 = 0.99

idxStart :: Int
idxStart = 0


-------------------- Constructors --------------------

-- Tabular representations

mkBORLUnichainTabular :: (Ord s) => InitialState s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> BORL s
mkBORLUnichainTabular initialState as asFilter params decayFun =
  BORL (zip [idxStart ..] as) asFilter initialState Nothing 0 params decayFun (default_gamma0, default_gamma1) (Left 0) (0, 0, 0) tabSA tabSA tabSA tabSA mempty
  where
    tabSA = Table mempty

mkBORLUnichainTensorflow :: forall s m . (NFData s, Ord s) => InitialState s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> TF.Session TensorflowModel -> NNConfig s -> IO (BORL s)
mkBORLUnichainTensorflow initialState as asFilter params decayFun modelBuilder nnConfig = do
  nnConfig' <- mkNNConfigSA as asFilter nnConfig
  -- Initialization for all NNs
  let nnTypes = [VTable, VTable, WTable, WTable, R0Table, R0Table, R1Table, R1Table]
      scopes = concat $ replicate 4 ["_target", "_worker"]
  let fullModelInit = sequenceA (zipWith3 (\tp sc fun -> TF.withNameScope (name tp <> sc) fun) nnTypes scopes (repeat modelBuilder))
  let netInpInitState = (nnConfig' ^. toNetInp) initialState
      nnSA :: ProxyType -> Int -> IO (Proxy s)
      nnSA tp idx = do
        nnT <- runMonadBorl $ mkModel tp "_target" netInpInitState ((!! idx) <$> fullModelInit)
        nnW <- runMonadBorl $ mkModel tp "_worker" netInpInitState ((!! (idx + 1)) <$> fullModelInit)
        return $ TensorflowProxy nnT nnW mempty tp nnConfig' (length as)
  v <- nnSA VTable 0
  w <- nnSA WTable 2
  r0 <- nnSA R0Table 4
  r1 <- nnSA R1Table 6
  return $ force $ BORL (zip [idxStart ..] as) asFilter initialState Nothing 0 params decayFun (default_gamma0, default_gamma1) (Left 0) (0, 0, 0) v w r0 r1 mempty
  where
    mkModel tp scope netInpInitState modelBuilderFun = do
      !model <- prependName (name tp <> scope) <$> Tensorflow modelBuilderFun
      saveModel (TensorflowModel' model Nothing (Just (map realToFrac netInpInitState, replicate (length as) 0)) modelBuilderFun) [map realToFrac netInpInitState] [replicate (length as) 0]
    prependName txt model =
      model {inputLayerName = txt <> "/" <> inputLayerName model, outputLayerName = txt <> "/" <> outputLayerName model, labelLayerName = txt <> "/" <> labelLayerName model}
    name VTable  = "v"
    name WTable  = "w"
    name R0Table = "r0"
    name R1Table = "r1"


mkBORLMultichainTabular :: (Ord s) => InitialState s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> BORL s
mkBORLMultichainTabular initialState as asFilter params decayFun =
  BORL (zip [0 ..] as) asFilter initialState Nothing 0 params decayFun (default_gamma0, default_gamma1) (Right tabSA) (0, 0, 0) tabSA tabSA tabSA tabSA mempty
  where
    tabSA = Table mempty

-- Neural network approximations

mkBORLUnichainGrenade ::
     forall nrH nrL s layers shapes. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, Ord s, NFData (Tapes layers shapes), NFData (Network layers shapes))
  => InitialState s
  -> [Action s]
  -> (s -> [Bool])
  -> Parameters
  -> Decay
  -> Network layers shapes
  -> NNConfig s
  -> IO (BORL s)
mkBORLUnichainGrenade initialState as asFilter params decayFun net nnConfig = do
  nnConfig' <- mkNNConfigSA as asFilter nnConfig
  let nnSA tp = Grenade net net mempty tp nnConfig' (length as) :: Proxy s
  return $
    checkGrenade net nnConfig $
    BORL
      (zip [idxStart ..] as)
      asFilter
      initialState
      Nothing
      0
      params
      decayFun
      (default_gamma0, default_gamma1)
      (Left 0)
      (0, 0, 0)
      (nnSA VTable)
      (nnSA WTable)
      (nnSA R0Table)
      (nnSA R1Table)
      mempty


mkBORLMultichainGrenade ::
     forall nrH nrL s layers shapes. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, Ord s, NFData (Tapes layers shapes), NFData (Network layers shapes))
  => InitialState s
  -> [Action s]
  -> (s -> [Bool])
  -> Parameters
  -> Decay
  -> Network layers shapes
  -> NNConfig s
  -> IO (BORL s)
mkBORLMultichainGrenade initialState as asFilter params decayFun net nnConfig = do
  nnConfig' <- mkNNConfigSA as asFilter nnConfig
  let nnSA tp = Grenade net net mempty tp nnConfig' (length as) :: Proxy s
  return $ checkGrenade net nnConfig $ BORL
    (zip [0 ..] as)
    asFilter
    initialState
    Nothing
    0
    params
    decayFun
    (default_gamma0, default_gamma1)
    (Right $ nnSA VTable)
    (0, 0, 0)
    (nnSA VTable)
    (nnSA WTable)
    (nnSA R0Table)
    (nnSA R1Table)
    mempty

-------------------- Other Constructors --------------------

-- noScaling :: ScalingNetOutParameters
-- noScaling = ScalingNetOutParameters

-- | Infer scaling by maximum reward.
scalingByMaxReward :: Bool -> Double -> ScalingNetOutParameters
scalingByMaxReward onlyPositive maxR = ScalingNetOutParameters (-maxV) maxV (-maxW) maxW (if onlyPositive then 0 else -maxR0) maxR0 (if onlyPositive then 0 else -maxR1) maxR1
  where maxDiscount g = sum $ take 10000 $ map (\p -> (g^p) * maxR) [(0::Int)..]
        maxV = 0.8 * maxR
        maxW = 300 * maxR
        maxR0 = 2 * maxDiscount default_gamma0
        maxR1 = 0.8 * maxDiscount default_gamma1


-------------------- Helpers --------------------

-- | Checks the neural network setup and throws an error in case of a faulty number of input or output nodes.
checkGrenade ::
     forall layers shapes nrH nrL s. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, Ord s)
  => Network layers shapes
  -> NNConfig s
  -> BORL s
  -> BORL s
checkGrenade _ nnConfig borl
  | nnInpNodes /= stInp = error $ "Number of input nodes for neural network is " ++ show nnInpNodes ++ " but should be " ++ show stInp
  | nnOutNodes /= fromIntegral nrActs = error $ "Number of output nodes for neural network is " ++ show nnOutNodes ++ " but should be " ++ show nrActs
  | otherwise = borl
  where
    nnInpNodes = fromIntegral $ natVal (Type.Proxy :: Type.Proxy nrH)
    nnOutNodes = natVal (Type.Proxy :: Type.Proxy nrL)
    stInp = length ((nnConfig ^. toNetInp) (borl ^. s))
    nrActs = length (borl ^. actionList)


-- | Converts the neural network state configuration to a state-action configuration.
mkNNConfigSA :: forall s . [Action s] -> (s -> [Bool]) -> NNConfig s -> IO (NNConfig s)
mkNNConfigSA as asFilter (NNConfig inp (ReplayMemory _ sz) bs lp pp sc c mse) = do
  vec <- V.new sz
  let rm' = ReplayMemory vec sz
  return $ NNConfig inp rm' bs lp pp sc c mse
  where
    -- maxVal = fromIntegral (length as)
    -- ppSA :: [s] -> [(s, ActionIndex)]
    -- ppSA = concatMap (\k -> map ((k,) . snd) (filter fst $ zip (asFilter k) [idxStart .. idxStart + length as - 1]))
