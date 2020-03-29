{-# LANGUAGE BangPatterns #-}
module ML.BORL.Action.Type where

import           Control.DeepSeq
import qualified Data.Text       as T

import           ML.BORL.Reward
import           ML.BORL.Types

-- | Agent type. There is only one main agent, but there could be multiple workers (configured via NNConfig).
data AgentType = MainAgent | WorkerAgent
  deriving (Show, Eq, Ord)

-- | An action is a function returning a reward and a new state, and has a name for pretty printing.
data Action s = Action
  { actionFunction :: !(AgentType -> s -> IO (Reward s, s, EpisodeEnd)) -- ^ An action which returns a reward r and a new state s' and if s' is the episode end
  , actionName     :: !T.Text                                         -- ^ Name of the action.
  }

instance NFData (Action s) where
  rnf (Action !_ n) = rnf n


instance Eq (Action s) where
  (Action _ t1) == (Action _ t2) = t1 == t2


instance Ord (Action s) where
  compare a1 a2 = compare (actionName a1) (actionName a2)


instance Show (Action s) where
  show a = show (actionName a)
