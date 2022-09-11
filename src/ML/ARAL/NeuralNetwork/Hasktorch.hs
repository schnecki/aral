{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TupleSections         #-}
module ML.ARAL.NeuralNetwork.Hasktorch
    ( MLP (..)
    , MLPSpec (..)
    , HasktorchActivation (..)
    , runHasktorch
    , trainHasktorch
    ) where

import           Control.Applicative              ((<|>))
import           Control.DeepSeq
import           Data.List                        (foldl', genericLength, intersperse)
import           Data.Maybe
import           Data.Serialize
import qualified Data.Vector.Storable             as V
import           GHC.Generics
import qualified Torch                            as Torch hiding (dropout)
import qualified Torch.Functional.Internal        as Torch (dropout, gather)
import qualified Torch.HList                      as Torch
import qualified Torch.Initializers               as Torch
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
  | HasktorchLeakyRelu (Maybe Float)
  | HasktorchTanh
  deriving (Show, Eq, Generic, Serialize, NFData)

mkHasktorchActivation :: HasktorchActivation -> Torch.Tensor -> Torch.Tensor
mkHasktorchActivation HasktorchRelu               = Torch.relu
mkHasktorchActivation (HasktorchLeakyRelu mAlpha) = Torch.leakyRelu (fromMaybe 0.02 mAlpha)
mkHasktorchActivation HasktorchTanh               = Torch.tanh

data MLPSpec
  = MLPSpec
      { hasktorchFeatureCounts    :: [Integer]
      , hasktorchHiddenActivation :: HasktorchActivation
      , hasktorchOutputActivation :: Maybe HasktorchActivation
      }
  | MLPSpecWDroput
      { hasktorchFeatureCounts      :: [Integer]
      , hasktorchHiddenActivation   :: HasktorchActivation
      , hasktorchHiddenDropoutAlpha :: Maybe (Bool, Double) -- Droput later after every hidden layer with given probability, e.g. 0.5
      , hasktorchOutputActivation   :: Maybe HasktorchActivation
      }
  deriving (Show, Eq, Generic, Serialize, NFData)

-- instance Serialize MLPSpec where
--   get = force <$> ((MLPSpec <$> get <*> get <*> get) <|> (MLPSpecWDroput <$> get <*> get <*> get <*> get))


-- Actual model fuction

data MLP =
  MLP
    { mlpLayers             :: [Torch.Linear]
    , mlpHiddenActivation   :: Torch.Tensor -> Torch.Tensor
    , mlpHiddenDropoutAlpha :: Maybe (Bool, Double) -- Nothing: no droput, Just: (active, alpha)
    , mlpOutputActivation   :: Maybe (Torch.Tensor -> Torch.Tensor)
    }
  deriving (Generic, Torch.Parameterized)

instance Show MLP where
  show (MLP layers _ Just{} _)  = show layers <> " with dropout"
  show (MLP layers _ Nothing _) = show layers <> " without dropout"


hasktorchModel :: MLP -> Torch.Tensor -> Torch.Tensor
hasktorchModel (MLP layers hiddenAct Nothing outputAct) input = foldl' revApply input $ (++ maybe [] pure outputAct) $ intersperse hiddenAct $ map Torch.linear layers
  where
    revApply x f = f x
hasktorchModel (MLP layers hiddenAct (Just (active, dropoutAlpha)) outputAct) input =
  foldl' revApply input $ (++ maybe [] pure outputAct) $ intersperse ((\t -> Torch.dropout t dropoutAlpha active) . hiddenAct) $ map Torch.linear layers
  where
    revApply x f = f x


instance Torch.Randomizable MLPSpec MLP where
  sample (MLPSpec featCounts hiddenActivation outputActivation) = Torch.sample (MLPSpecWDroput featCounts hiddenActivation Nothing outputActivation)
  sample (MLPSpecWDroput featCounts hiddenActivation mDropout outputActivation) = do
    let layerSizes = mkLayerSizes (map fromInteger featCounts)
    linears <- mapM sampleDouble layerSizes
    return $
      MLP
        { mlpLayers = linears
        , mlpHiddenActivation = mkHasktorchActivation hiddenActivation
        , mlpHiddenDropoutAlpha = mDropout
        , mlpOutputActivation = mkHasktorchActivation <$> outputActivation
        }
    where
      mkLayerSizes (a:(b:t)) = scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)
      mkLayerSizes _ = error "Need at least 2 layers in MLPSpec"
      sampleDouble (in_features, out_features) = do
        w <-
          Torch.makeIndependent . mkDbl =<< -- Torch.kaimingUniform Torch.FanIn (Torch.LeakyRelu $ Prelude.sqrt (5.0 :: Float)) [out_features, in_features]
          Torch.xavierUniform 1.0 [out_features, in_features]
        init <- Torch.randIO' [out_features]
        let bound = (1 :: Double) / Prelude.sqrt (fromIntegral (Torch.getter Torch.FanIn $ Torch.calculateFan [out_features, in_features]) :: Double)
        b <- Torch.makeIndependent . mkDbl =<< pure (Torch.subScalar bound $ Torch.mulScalar (bound * 2.0 :: Double) init)
        return $ Torch.Linear w b
      mkDbl = Torch.toDType Torch.Double


runHasktorch :: MLP -> NrActions -> NrAgents -> StateFeatures -> [Values]
runHasktorch mlp nrAs nrAgents st =
  -- trace ("mlp: " ++ show mlp) $
  let input = Torch.asTensor $ V.toList st
      output = -- V.map toDouble $
        V.fromList $ Torch.asValue $ hasktorchModel mlp input
   in [toAgents nrAs nrAgents output]

trainHasktorch :: (Torch.Optimizer optimizer) => Period -> Double -> optimizer -> NNConfig -> MLP -> [[((StateFeatures, ActionIndex), Double)]] -> IO (MLP, optimizer)
trainHasktorch period lRate optimizer nnConfig model chs = do
  -- map (makeLoss model) chs

  let grads :: Torch.Gradients
      grads =
        -- trace ("inputs: " ++ show (Torch.asTensor $ map (map (V.toList . fst . fst)) chs))
        -- trace ("actions: " ++ show actions)
        -- trace ("outputs: " ++ show outputs ) $
        -- trace ("gather outputs: " ++ show (gather outputs)) $
        -- trace ("targets: " ++ show targets)
        -- trace ("loss: " ++ show (mkLoss targets $ gather $ hasktorchModel model inputs) ) $
        -- (`Torch.div` (1 / sum (map genericLength chs))) $
        -- computeGrads $ mkLossDirect
        mkIndepLossAndGrads
        targets $ gather outputs
      outputs = hasktorchModel model inputs
      inputs = Torch.asTensor $ map (map (V.toList . fst . fst)) chs
      actions = Torch.asTensor $ map ((: []) . map (snd . fst)) chs
      targets = Torch.asTensor $ map (map snd) chs
      -- zeros = Torch.zerosLike targets
      nrRows = Torch.size 0 targets
      nrNStep = Torch.size 1 targets
      nrAs = Torch.size 2 targets
      gather x = Torch.reshape [nrRows, nrNStep] $ Torch.gather x 2 actions False
      computeGrads l = Torch.Gradients $ Torch.grad l (Torch.flattenParameters model)
      mkIndepLossAndGrads t o =
        Torch.Gradients $
        foldl1 (zipWith (+)) $
        map (\l -> Torch.grad l (Torch.flattenParameters model)) $
        zipWith lossFun (Torch.split nrRows (Torch.Dim 0) t) (Torch.split nrRows (Torch.Dim 0) o)
      lossFun t o = Torch.mseLoss t o
      -- lossFun t o = Torch.smoothL1Loss Torch.ReduceMean o t
      -- mkLossDirect t o = Torch.mseLoss t o
      -- mkLossDirect t o = Torch.smoothL1Loss Torch.ReduceMean o t
      lRateTensor = Torch.asTensor lRate
  Torch.runStep' model optimizer grads lRateTensor


mkSqrdLoss :: Double -> Double -> Double
mkSqrdLoss t o =
  let l = o - t
   in signum l * l ^ (2 :: Int)
