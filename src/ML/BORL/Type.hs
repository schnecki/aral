{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeFamilies        #-}

module ML.BORL.Type where

import           ML.BORL.Parameters
import           ML.BORL.Proxy

import           Control.Arrow                (first)
import           Control.Lens
import qualified Data.Map.Strict              as M
import           Data.Singletons.Prelude.List
import qualified Data.Text                    as T
import           GHC.TypeLits
import           Grenade


-- Types
type Period = Integer
type Reward = Double
type ActionIndexed s = (ActionIndex, Action s) -- ^ An action with index.
type ActionIndex = Int
type InitialState s = s                            -- ^ Initial state
type Decay = Period -> Parameters -> Parameters -- ^ Function specifying the decay of the parameters at time t.

data Action s = Action { actionFunction :: s -> IO (Reward, s) -- ^ An action which returns a reward r and a new state s'
                       , actionName     :: T.Text
                       }
instance Eq (Action s) where
  (Action _ t1) == (Action _ t2) = t1 == t2

instance Ord (Action s) where
  compare a1 a2 = compare (actionName a1) (actionName a2)

instance Show (Action s) where
  show a = show (actionName a)


-------------------- Main RL Datatype --------------------


data BORL s = BORL
  { _actionList    :: ![ActionIndexed s]    -- ^ List of possible actions in state s.
  , _actionFilter  :: s -> [Bool]           -- ^ Function to filter actions in state s.
  , _s             :: !s                    -- ^ Current state.
  , _t             :: !Integer              -- ^ Current time t.
  , _parameters    :: !Parameters           -- ^ Parameter setup.
  , _decayFunction :: !Decay                -- ^ Decay function at period t.

  -- discount factors
  , _gammas        :: !(Double, Double) -- ^ Two gamma values in ascending order

  -- Values:
  , _rho           :: !(Either Double (Proxy (s,ActionIndex))) -- ^ Either unichain or multichain y_{-1} values.
  , _psis          :: !(Double, Double, Double)                -- ^ Exponentially smoothed psi values.
  , _psiStates     :: !(Proxy s, Proxy s, Proxy s)             -- ^ Psi values for the states.
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

mkBORLUnichainTabular :: (Ord s) => InitialState s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> BORL s
mkBORLUnichainTabular initialState as asFilter params decayFun =
  BORL (zip [0 ..] as) asFilter initialState 0 params decayFun (default_gamma0, default_gamma1) (Left 0) (0, 0, 0) (tabS, tabS, tabS) tabSA tabSA tabSA tabSA mempty
  where
    tabS = Table mempty
    tabSA = Table mempty

mkBORLUnichain :: forall nrH nrL s layers shapes .
     (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, Ord s)
  => InitialState s
  -> [Action s]
  -> (s -> [Bool])
  -> Parameters
  -> Decay
  -> Network layers shapes
  -> NNConfig s
  -> BORL s
mkBORLUnichain initialState as asFilter params decayFun net nnConfig =
  BORL (zip [0 ..] as) asFilter initialState 0 params decayFun (default_gamma0, default_gamma1) (Left 0) (0, 0, 0) (tabS, tabS, tabS) tabSA tabSA tabSA tabSA mempty
  where
    tabS = NN net nnConfig
    tabSA = NN net (mkNNConfigSA nnConfig) :: Proxy (s, ActionIndex)
    mkNNConfigSA :: NNConfig s -> NNConfig (s, ActionIndex)
    mkNNConfigSA (NNConfig inp _ bs lp) = NNConfig (toSA inp) [] bs lp :: NNConfig (s, ActionIndex)
    toSA :: (s -> [Double]) -> (s, ActionIndex) -> [Double]
    toSA f (state,a) = f state ++ [fromIntegral (2 * a) / divisor - 1]
    divisor = fromIntegral (length as)

mkBORLMultichainTabular :: (Ord s) => InitialState s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> BORL s
mkBORLMultichainTabular initialState as asFilter params decayFun =
  BORL (zip [0 ..] as) asFilter initialState 0 params decayFun (default_gamma0, default_gamma1) (Right tabSA) (0, 0, 0) (tabS, tabS, tabS) tabSA tabSA tabSA tabSA mempty
  where
    tabS = Table mempty
    tabSA = Table mempty


isMultichain :: BORL s -> Bool
isMultichain borl =
  case borl ^. rho of
    Left {}  -> False
    Right {} -> True

isUnichain :: BORL s -> Bool
isUnichain = not . isMultichain

