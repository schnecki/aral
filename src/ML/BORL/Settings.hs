{-# LANGUAGE DeriveAnyClass       #-}
{-# LANGUAGE DeriveGeneric        #-}
{-# LANGUAGE TemplateHaskell      #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE TypeSynonymInstances #-}
module ML.BORL.Settings where

import           Control.DeepSeq
import           Control.Lens
import           Data.Default
import           Data.Serialize
import           GHC.Generics

import           ML.BORL.Exploration

-- Parameters
data Settings = Settings
  { _explorationStrategy   :: !ExplorationStrategy -- ^ Strategy for exploration.
  , _workersMinExploration :: ![Float]             -- ^ Set worker minimum exploration values.
  , _workersUpdateInterval :: !Int                 -- ^ Number of periods before workers report results and are updated with the latest.
  , _nStep                 :: !Int                 -- ^ N-Step Q-Learning. 1 means no N-step Q-learning. Only works with @ReplayMemorySingle@!
  , _disableAllLearning    :: !Bool                -- ^ Completely disable learning (e.g. for evaluation). Enabling increases performance.
  , _useForking            :: !Bool                -- ^ Fork where possible.
  } deriving (Show, Eq, Ord, NFData, Generic, Serialize)
makeLenses ''Settings


instance Default Settings where
  def = Settings EpsilonGreedy [] 1000 1 False True

