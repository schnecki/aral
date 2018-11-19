{-# LANGUAGE TemplateHaskell #-}

module ML.BORL.Type where

import           Control.Lens
import qualified Data.Map.Strict    as M
import qualified Data.Text          as T
import           ML.BORL.Parameters

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
  , _rho           :: !(Either Double (M.Map (s,ActionIndex) Double))   -- ^ Either unichain or multichain y_{-1} values.
  , _psis          :: !(Double, Double, Double)                         -- ^ Exponentially smoothed psi values.
  , _psiStates     :: !(M.Map s Double, M.Map s Double, M.Map s Double) -- ^ Psi values for the states.
  , _v             :: !(M.Map (s,ActionIndex) Double)                   -- ^ Bias values (y_0).
  , _w             :: !(M.Map (s,ActionIndex) Double)                   -- ^ y_1 values.
  , _r0            :: !(M.Map (s,ActionIndex) Double)                   -- ^ Discounted values with first gamma value.
  , _r1            :: !(M.Map (s,ActionIndex) Double)                   -- ^ Discounted values with second gamma value.

  -- Stats:
  , _visits        :: !(M.Map s Integer)                                -- ^ Counts the visits of the states
  }
makeLenses ''BORL

default_gamma0, default_gamma1 :: Double
default_gamma0 = 0.25
default_gamma1 = 0.99

mkBORLUnichain :: (Ord s) => InitialState s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> BORL s
mkBORLUnichain initialState as asFilter params decayFun =
  BORL (zip [0 ..] as) asFilter initialState 0 params decayFun (default_gamma0, default_gamma1) (Left 0) (0, 0, 0) (mempty, mempty, mempty) mempty mempty mempty mempty mempty

mkBORLMultichain :: (Ord s) => InitialState s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> BORL s
mkBORLMultichain initialState as asFilter params decayFun =
  BORL (zip [0 ..] as) asFilter initialState 0 params decayFun (default_gamma0, default_gamma1) (Right mempty) (0, 0, 0) (mempty, mempty, mempty) mempty mempty mempty mempty mempty


isMultichain :: BORL s -> Bool
isMultichain borl =
  case borl ^. rho of
    Left {}  -> False
    Right {} -> True

isUnichain :: BORL s -> Bool
isUnichain = not . isMultichain

