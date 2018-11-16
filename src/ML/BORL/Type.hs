{-# LANGUAGE TemplateHaskell #-}

module ML.BORL.Type where

import           Control.Lens
import qualified Data.Map.Strict    as M
import           Data.Monoid
import           ML.BORL.Parameters

-- Types
type Period = Integer
type Reward = Double
type Action s = s -> IO (Reward, s)            -- ^ An action which returns a reward r and a new state s'
type ActionIndexed s = (ActionIndex, Action s) -- ^ An action with index.
type ActionIndex = Int
type InitState s = s                            -- ^ Initial state
type Decay = Period -> Parameters -> Parameters -- ^ Function specifying the decay of the parameters at time t.


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
  , _rho           :: !(Either Double (M.Map (s,ActionIndex) Double))                                             -- ^ Either unichain or multichain y_{-1} values.
  , _psis          :: !(Double, Double, Double)                                                                   -- ^ Exponentially smoothed psi values.
  , _psiStates     :: !(M.Map (s,ActionIndex) Double, M.Map (s,ActionIndex) Double, M.Map (s,ActionIndex) Double) -- ^ Psi values for the states.
  , _v             :: !(M.Map (s,ActionIndex) Double)                                                             -- ^ Bias values (y_0).
  , _w             :: !(M.Map (s,ActionIndex) Double)                                                             -- ^ y_1 values.
  , _r0            :: !(M.Map (s,ActionIndex) Double)                                                             -- ^ Discounted values with first gamma value.
  , _r1            :: !(M.Map (s,ActionIndex) Double)                                                             -- ^ Discounted values with second gamma value.

  -- Stats:
  , _visits        :: !(M.Map (s,ActionIndex) Integer)                                                            -- ^ Counts the visits of the states
  }
makeLenses ''BORL


mkBORLUnichain :: (Ord s) => InitState s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> BORL s
mkBORLUnichain init as asFilter params decayFunction = BORL (zip [0..] as) asFilter init 0 params decayFunction (0.25, 0.75) (Left 0) (0, 0, 0) (mempty, mempty, mempty) mempty mempty mempty mempty mempty

mkBORLMultichain :: (Ord s) => InitState s -> [Action s] -> (s -> [Bool]) -> Parameters -> Decay -> BORL s
mkBORLMultichain init as asFilter params decayFunction = BORL (zip [0..] as) asFilter init 0 params decayFunction (0.25, 0.75) (Right mempty) (0, 0, 0) (mempty, mempty, mempty) mempty mempty mempty mempty mempty


isMultichain :: BORL s -> Bool
isMultichain borl =
  case borl ^. rho of
    Left {}  -> False
    Right {} -> True

isUnichain :: BORL s -> Bool
isUnichain = not . isMultichain

