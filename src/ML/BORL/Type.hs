{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TupleSections       #-}
{-# LANGUAGE TypeFamilies        #-}

module ML.BORL.Type where

import           ML.BORL.Action
import           ML.BORL.NeuralNetwork
import           ML.BORL.Parameters
import           ML.BORL.Proxy

import           Control.Lens
import qualified Data.Map.Strict              as M
import qualified Data.Proxy                   as Type
import           Data.Singletons.Prelude.List
import           GHC.TypeLits
import           Grenade


-------------------- Types --------------------

type Period = Integer
type ActionIndexed s = (ActionIndex, Action s) -- ^ An action with index.
type ActionIndex = Int
type InitialState s = s                            -- ^ Initial state
type Decay = Period -> Parameters -> Parameters -- ^ Function specifying the decay of the parameters at time t.


-------------------- Main RL Datatype --------------------


data BORL s = BORL
  { _actionList    :: ![ActionIndexed s]    -- ^ List of possible actions in state s.
  , _actionFilter  :: !(s -> [Bool])        -- ^ Function to filter actions in state s.
  , _s             :: !s                    -- ^ Current state.
  , _t             :: !Integer              -- ^ Current time t.
  , _parameters    :: !Parameters           -- ^ Parameter setup.
  , _decayFunction :: !Decay                -- ^ Decay function at period t.

  -- discount factors
  , _gammas        :: !(Double, Double) -- ^ Two gamma values in ascending order

  -- Values:
  , _rho           :: !(Either Double (Proxy (s,ActionIndex))) -- ^ Either unichain or multichain y_{-1} values.
  , _psis          :: !(Double, Double, Double)                -- ^ Exponentially smoothed psi values.
  , _v             :: !(Proxy (s,ActionIndex))                 -- ^ Bias values (y_0).
  , _w             :: !(Proxy (s,ActionIndex))                 -- ^ y_1 values.
  , _r0            :: !(Proxy (s,ActionIndex))                 -- ^ Discounted values with first gamma value.
  , _r1            :: !(Proxy (s,ActionIndex))                 -- ^ Discounted values with second gamma value.

  -- Stats:
  , _visits        :: !(M.Map s Integer)                       -- ^ Counts the visits of the states
  }
makeLenses ''BORL

default_gamma0, default_gamma1 :: Double
default_gamma0 = 0.25
default_gamma1 = 0.99

idxStart :: Int
idxStart = 0


-------------------- Constructors --------------------

-- Tabular representations

mkBORLUnichainTabular :: (Ord s) => InitialState s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> BORL s
mkBORLUnichainTabular initialState as asFilter params decayFun =
  BORL (zip [idxStart ..] as) asFilter initialState 0 params decayFun (default_gamma0, default_gamma1) (Left 0) (0, 0, 0) tabSA tabSA tabSA tabSA mempty
  where
    tabSA = Table mempty

mkBORLMultichainTabular :: (Ord s) => InitialState s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> BORL s
mkBORLMultichainTabular initialState as asFilter params decayFun =
  BORL (zip [0 ..] as) asFilter initialState 0 params decayFun (default_gamma0, default_gamma1) (Right tabSA) (0, 0, 0) tabSA tabSA tabSA tabSA mempty
  where
    tabSA = Table mempty

-- Neural network approximations

mkBORLUnichain ::
     forall nrH nrL s layers shapes. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, Ord s)
  => InitialState s
  -> [Action s]
  -> (s -> [Bool])
  -> Parameters
  -> Decay
  -> Network layers shapes
  -> NNConfig s
  -> BORL s
mkBORLUnichain initialState as asFilter params decayFun net nnConfig =
  checkNN net nnConfig $
  BORL
    (zip [idxStart ..] as)
    asFilter
    initialState
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
  where
    nnSA tp = NN net tp (mkNNConfigSA as asFilter nnConfig) :: Proxy (s, ActionIndex)


mkBORLMultichain ::
     forall nrH nrL s layers shapes. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, Ord s)
  => InitialState s
  -> [Action s]
  -> (s -> [Bool])
  -> Parameters
  -> Decay
  -> Network layers shapes
  -> NNConfig s
  -> BORL s
mkBORLMultichain initialState as asFilter params decayFun net nnConfig =
  checkNN net nnConfig $
  BORL
    (zip [0 ..] as)
    asFilter
    initialState
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
  where
    nnSA tp = NN net tp (mkNNConfigSA as asFilter nnConfig) :: Proxy (s, ActionIndex)


-------------------- Other Constructors --------------------

-- | Infer scaling by maximum reward.
scalingByMaxReward :: Double -> ScalingNetOutParameters
scalingByMaxReward maxR = ScalingNetOutParameters (3*maxR) (500*maxR) (1.5*maxDiscount default_gamma0) (1.5*maxDiscount default_gamma1)
  where maxDiscount g = sum $ take 1000 $ map (\p -> (g^p) * maxR) [(0::Int)..]


-------------------- Helpers --------------------

-- | Checks the neural network setup and throws an error in case of a faulty number of input or output nodes.
checkNN ::
     forall layers shapes nrH nrL s. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, Ord s)
  => Network layers shapes
  -> NNConfig s
  -> BORL s
  -> BORL s
checkNN _ nnConfig borl
  | nnInpNodes /= stInp + 1 = error $ "Number of input nodes for neural network is " ++ show nnInpNodes ++ " but should be " ++ show (stInp + 1)
  | nnOutNodes /= 1 = error $ "Number of output nodes for neural network is " ++ show nnOutNodes ++ " but should be 1"
  | otherwise = borl
  where
    nnInpNodes = fromIntegral $ natVal (Type.Proxy :: Type.Proxy nrH)
    nnOutNodes = natVal (Type.Proxy :: Type.Proxy nrL)
    stInp = length ((nnConfig ^. toNetInp) (borl ^. s))


-- | Converts the neural network state configuration to a state-action configuration.
mkNNConfigSA :: forall s . [Action s] -> (s -> [Bool]) -> NNConfig s -> NNConfig (s, ActionIndex)
mkNNConfigSA as asFilter (NNConfig inp _ bs lp pp sc) = NNConfig (toSA inp) [] bs lp (ppSA pp) sc
  where
    maxVal = fromIntegral (length as)
    toSA :: (s -> [Double]) -> (s, ActionIndex) -> [Double]
    toSA f (state, a) = f state ++ [scaleNegPosOne maxVal (fromIntegral a)]
    ppSA :: [s] -> [(s, ActionIndex)]
    ppSA = concatMap (\k -> map ((k,) . snd) (filter fst $ zip (asFilter k) [idxStart .. idxStart + length as - 1]))
