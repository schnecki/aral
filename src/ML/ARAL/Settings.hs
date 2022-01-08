{-# LANGUAGE DeriveAnyClass       #-}
{-# LANGUAGE DeriveGeneric        #-}
{-# LANGUAGE TemplateHaskell      #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE TypeSynonymInstances #-}
module ML.ARAL.Settings where

import           Control.DeepSeq
import           Control.Lens
import           Data.Default
import           Data.Serialize
import           GHC.Generics

import           ML.ARAL.Exploration

-- Parameters
data Settings = Settings
  { _explorationStrategy           :: !ExplorationStrategy -- ^ Strategy for exploration.
  , _workersMinExploration         :: ![Double]             -- ^ Set worker minimum exploration values. [Default: No workers]
  , _mainAgentSelectsGreedyActions :: !Bool                -- ^ Let the main agent always choose the greedy action. [Default: False]
  , _nStep                         :: !Int                 -- ^ N-Step Q-Learning. 1 means no N-step Q-learning. Only works with @ReplayMemorySingle@! [Default: 1]
  , _disableAllLearning            :: !Bool                -- ^ Completely disable learning (e.g. for evaluation). Enabling increases performance. [Default: False]
  , _useProcessForking             :: !Bool                -- ^ Use actual process forking where possible. [Default: True]
  , _overEstimateRho               :: !Bool                -- ^ Overestimate the average reward to find better policies. This may lead to incorrect state value estimates! [Default: False]
  , _independentAgents             :: !Int                 -- ^ Split action space into independent X agents. At least 1. Changes have no effect after initialisation of the agent.
  , _independentAgentsSharedRho    :: !Bool                -- ^ Share the average reward over all independent agents. Default: True
  } deriving (Show, Eq, Ord, NFData, Generic, Serialize)
makeLenses ''Settings


instance Default Settings where
  def = Settings EpsilonGreedy [] False 1 False True False 1 True
