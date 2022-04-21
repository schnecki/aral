{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module ML.ARAL.NeuralNetwork.Hasktorch
    ( MLP (..)
    , MLPSpec (..)
    , HasktorchActivation (..)
    , runHasktorch
    , trainHasktorch
    ) where

import           Control.DeepSeq
import           Data.List                        (foldl', genericLength, intersperse)
import           Data.Maybe
import           Data.Serialize
import qualified Data.Vector.Storable             as V
import           GHC.Generics
import qualified Torch                            as Torch
import qualified Torch.Functional.Internal        as Torch (gather)
import qualified Torch.Serialize                  as Torch
import qualified Torch.Tensor                     as Torch
import qualified Torch.Typed.Vision               as Torch (initMnist)
import qualified Torch.Vision                     as Torch.V

import           ML.ARAL.NeuralNetwork.Conversion
import           ML.ARAL.NeuralNetwork.NNConfig
import           ML.ARAL.Types

import           Debug.Trace


data HasktorchActivation
  = HasktorchRelu
  | HasktorchTanh
  deriving (Show, Eq, Generic, Serialize, NFData)

mkHasktorchActivation :: HasktorchActivation -> Torch.Tensor -> Torch.Tensor
mkHasktorchActivation HasktorchRelu = Torch.relu
mkHasktorchActivation HasktorchTanh = Torch.tanh

data MLPSpec = MLPSpec
  { hasktorchFeatureCounts    :: [Integer]
  , hasktorchHiddenActivation :: HasktorchActivation
  , hasktorchOutputActivation :: Maybe HasktorchActivation
  } deriving (Show, Eq, Generic, Serialize, NFData)

-- Actual model fuction

data MLP = MLP
  { mlpLayers           :: [Torch.Linear]
  , mlpHiddenActivation :: Torch.Tensor -> Torch.Tensor
  , mlpOutputActivation :: Maybe (Torch.Tensor -> Torch.Tensor)
  }
  deriving (Generic, Torch.Parameterized)

instance Show MLP where
  show (MLP layers _ _) = show layers


hasktorchModel :: MLP -> Torch.Tensor -> Torch.Tensor
hasktorchModel (MLP layers hiddenAct outputAct) input = foldl' revApply input $ intersperse hiddenAct $ map Torch.linear layers ++ maybe [] pure outputAct
  where
    revApply x f = f x


instance Torch.Randomizable MLPSpec MLP where
  sample (MLPSpec featCounts hiddenActivation outputActivation) = do
    let layerSizes = mkLayerSizes (map fromInteger featCounts)
    linears <- mapM (Torch.sample . uncurry Torch.LinearSpec) layerSizes
    return $ MLP {mlpLayers = linears, mlpHiddenActivation = mkHasktorchActivation hiddenActivation, mlpOutputActivation = mkHasktorchActivation <$> outputActivation}
    where
      mkLayerSizes (a:(b:t)) = scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)
      mkLayerSizes _ = error "Need at least 2 layers in MLPSpec"


toFloat :: Double -> Float
toFloat = realToFrac

toDouble :: Float -> Double
toDouble = realToFrac

toFloatList :: V.Vector Double -> [Float]
toFloatList = V.toList . V.map realToFrac

toDoubleList :: [Float] -> V.Vector Double
toDoubleList = V.map realToFrac . V.fromList

runHasktorch :: MLP -> NrActions -> NrAgents -> StateFeatures -> [Values]
runHasktorch mlp nrAs nrAgents st =
  let input = Torch.asTensor $ map toFloat $ V.toList st
      output = V.map toDouble $ V.fromList $ Torch.asValue $ hasktorchModel mlp input
   in [toAgents nrAs nrAgents output]

trainHasktorch :: (Torch.Optimizer optimizer) => Period -> Double -> optimizer -> NNConfig -> MLP -> [[((StateFeatures, ActionIndex), Double)]] -> IO (MLP, optimizer)
trainHasktorch period lRate optimizer nnConfig model chs = do
  -- map (makeLoss model) chs

  let loss :: Torch.Tensor
      loss =
        -- trace ("inputs: " ++ show (Torch.asTensor $ map (map (map toFloat . V.toList . fst . fst)) chs))
        -- trace ("actions: " ++ show actions)
        -- trace ("outputs: " ++ show (hasktorchModel model inputs) ) $
        -- trace ("gather outputs: " ++ show (gather (hasktorchModel model inputs)) ) $
        -- trace ("targets: " ++ show targets)
        -- trace ("loss: " ++ show (mkLoss targets $ gather $ hasktorchModel model inputs) ) $
        -- (`Torch.div` (1 / sum (map genericLength chs))) $
        mkLoss targets $ gather $ hasktorchModel model inputs
      inputs = Torch.asTensor $ map (map (map toFloat . V.toList . fst . fst)) chs
      actions = Torch.asTensor $ map ((: []) . map (snd . fst)) chs
      targets = Torch.asTensor $ map (map (toFloat . snd)) chs
      gather x = Torch.reshape [Torch.size 0 targets, Torch.size 1 targets] $ Torch.gather x 0 actions False
      -- mkLoss t o = Torch.mseLoss t o
      mkLoss t o = Torch.smoothL1Loss Torch.ReduceMean o t
      lRateTensor = Torch.asTensor (toFloat lRate)
  Torch.runStep model optimizer loss lRateTensor
