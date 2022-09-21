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

import           Control.Applicative                         ((<|>))
import           Control.DeepSeq
import           Control.Monad                               (when)
import           Data.List                                   (foldl', genericLength, intersperse)
import           Data.Maybe
import           Data.Serialize
import qualified Data.Vector.Storable                        as V
import           GHC.Generics
import           Statistics.Sample.WelfordOnlineMeanVariance
import qualified Torch                                       as Torch hiding (dropout)
import qualified Torch.Autograd                              as Torch (toDependent)
import qualified Torch.Functional                            as Torch hiding (dropout)
import qualified Torch.Functional.Internal                   as Torch (dropout, gather)
import qualified Torch.HList                                 as Torch
import qualified Torch.Initializers                          as Torch
import qualified Torch.NN.Recurrent.Cell.LSTM                as LSTM
import qualified Torch.Serialize                             as Torch
import qualified Torch.Tensor                                as Torch
import qualified Torch.Typed.Vision                          as Torch (initMnist)
import qualified Torch.Vision                                as Torch.V

import           ML.ARAL.NeuralNetwork.Conversion
import           ML.ARAL.NeuralNetwork.NNConfig
import           ML.ARAL.NeuralNetwork.Normalisation
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
  | MLPSpecWDropoutLSTM
      { hasktorchFeatureCounts      :: [Integer]
      , hasktorchHiddenActivation   :: HasktorchActivation
      , hasktorchHiddenDropoutAlpha :: Maybe (Bool, Double) -- Dropout later after every hidden layer with given probability, e.g. 0.5
      , hasktorchLSTM               :: Maybe (Int, Int)      -- After first FF layer. Number of layers and hidden nr.
      , hasktorchOutputActivation   :: Maybe HasktorchActivation
      }
  deriving (Show, Eq, Generic, Serialize, NFData)


-- Actual model fuction

batchSize = 1

sequenceLength = 1


data MLP_LSTM =
  MLP_LSTM
    { lstmNrLayers  :: Int
    , lstmNrInputs  :: Int
    , lstmNrHidden  :: Int
    , lstmBatchSize :: Int
    , lstmH0        :: Torch.Parameter
    , lstmC0        :: Torch.Parameter
    , lstmCell      :: LSTM.LSTMCell
    }
  deriving (Generic, Torch.Parameterized)

data MLP =
  MLP
    { mlpLayers             :: [Torch.Linear]
    , mlpHiddenActivation   :: Torch.Tensor -> Torch.Tensor
    , mlpHiddenDropoutAlpha :: Maybe (Bool, Double) -- Nothing: no dropout, Just: (active, alpha)
    , mlpLSTM               :: Maybe MLP_LSTM -- Nothing: no LSTM, Just: numLayers, h0, c0, LSTM weights and biases
    , mlpOutputActivation   :: Maybe (Torch.Tensor -> Torch.Tensor)
    }
  deriving (Generic, Torch.Parameterized)

instance Show MLP where
  show (MLP layers _ mDr mLSTM _)  = show layers <> maybe "" (const " with dropout") mDr <> maybe "" (const " with LSTM") mLSTM

hasktorchModel :: Bool -> MLP -> Torch.Tensor -> (Torch.Tensor, Maybe (Torch.Tensor, Torch.Tensor))
hasktorchModel _ (MLP layers hiddenAct mDropout Nothing outputAct) input =
  (, Nothing) $ foldl' revApply input $ (++ maybe [] pure outputAct) $ intersperse hiddenLayer $ map Torch.linear layers
  where
    hiddenLayer = case mDropout of
      (Just (active, dropoutAlpha)) -> ((\t -> Torch.dropout t dropoutAlpha active) . hiddenAct)
      Nothing                       -> hiddenAct
    revApply x f = f x
hasktorchModel train (MLP layers hiddenAct mDropout (Just (MLP_LSTM numLayers nrInputs _ batchSize h0 c0 (LSTM.LSTMCell wih whh bih bhh))) outputAct) input =
  runLstmAndRest $ revApply input layer1
  where
    dropout = 0 :: Double
    (layer1:layerRest) = (++ maybe [] pure outputAct) $ intersperse hiddenLayer $ map Torch.linear layers
    toDep = Torch.toDependent
    runLstmAndRest x = trace ("dim: " ++ show (Torch.dim x)) $
          -- (toDep inpW) (toDep hiddenW) (toDep inpBias) (toDep hiddenBias) (h0, c0) x
      let (out, hn, cn) = Torch.lstm (Torch.reshape [batchSize, sequenceLength, nrInputs] x) [toDep h0, toDep c0] [toDep wih, toDep whh, toDep bih, toDep bhh] True numLayers dropout train False True
          -- inp = torch.randn(batch_size, seq_len, input_dim)

           -- see https://github.com/pytorch/pytorch/blob/e3900d2ba5c9f91a24a9ce34520794c8366d5c54/benchmarks/fastrnns/factory.py
           --
           -- hx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
           -- cx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
           -- lstm = torch.nn.LSTM(inputSize, hiddenSize, numLayers).to(device)
       in (foldl' revApply out layerRest, Just (hn, cn))
    hiddenLayer =
      case mDropout of
        (Just (active, dropoutAlpha)) -> ((\t -> Torch.dropout t dropoutAlpha active) . hiddenAct)
        Nothing                       -> hiddenAct
    revApply x f = f x

randLSTMCell :: Int -> Int -> IO LSTM.LSTMCell
randLSTMCell inputSize hiddenSize = do
  -- x4 dimension calculations - see https://pytorch.org/docs/master/generated/torch.nn.LSTMCell.html
  weightsIH' <- Torch.makeIndependent . mkDbl =<< initScale <$> Torch.randIO' [4 * hiddenSize, inputSize]
  weightsHH' <- Torch.makeIndependent . mkDbl =<< initScale <$> Torch.randIO' [4 * hiddenSize, hiddenSize]
  biasIH'    <- Torch.makeIndependent . mkDbl =<< initScale <$> Torch.randIO' [4 * hiddenSize]
  biasHH'    <- Torch.makeIndependent . mkDbl =<< initScale <$> Torch.randIO' [4 * hiddenSize]
  pure $
    LSTM.LSTMCell
      { LSTM.weightsIH = weightsIH',
        LSTM.weightsHH = weightsHH',
        LSTM.biasIH = biasIH',
        LSTM.biasHH = biasHH'
      }
  where
    scale = Prelude.sqrt $ 1.0 / fromIntegral hiddenSize :: Double
    initScale = Torch.subScalar scale . Torch.mulScalar scale . Torch.mulScalar (2.0 :: Double)


instance Torch.Randomizable MLPSpec MLP where
  sample (MLPSpec featCounts hiddenActivation outputActivation) = Torch.sample (MLPSpecWDropoutLSTM featCounts hiddenActivation Nothing Nothing outputActivation)
  sample (MLPSpecWDropoutLSTM featCounts hiddenActivation mDropout mLSTM outputActivation) = do
    let layerSizes = mkLayerSizes (map fromInteger featCounts)
    linears <- mapM sampleDouble layerSizes
    when (length featCounts < 2) $ error "Need at least 1 layer (2 inputs) for ANN"
    let sampleLstm (numLayers, nrHidden) = do
          -- let h0 = Torch.zeros' [numLayers, 1, nrHidden]
          -- let c0 = Torch.zeros' [numLayers, 1, nrHidden]
          h0 <- Torch.makeIndependent . mkDbl =<< Torch.randIO' [numLayers, batchSize, nrHidden]
          c0 <- Torch.makeIndependent . mkDbl =<< Torch.randIO' [numLayers, batchSize, nrHidden]
          -- hidden = (hidden_state, cell_state)

          let nrInput = fromInteger $ featCounts !! 1
          -- cell <- Torch.sample $ LSTM.LSTMSpec nrInput nrHidden
          cell <- randLSTMCell nrInput nrHidden
          return $
            trace ("nrInput: " ++ show nrInput) $
            Just $ MLP_LSTM numLayers nrInput nrHidden batchSize h0 c0 cell
    mLSTMLayer <- maybe (return Nothing) sampleLstm mLSTM
    return $
      MLP
        { mlpLayers = linears
        , mlpHiddenActivation = mkHasktorchActivation hiddenActivation
        , mlpHiddenDropoutAlpha = mDropout
        , mlpLSTM = mLSTMLayer
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


runHasktorch :: MLP -> NrActions -> NrAgents -> Maybe (WelfordExistingAggregate StateFeatures) -> StateFeatures -> [Values]
runHasktorch mlp nrAs nrAgents mWel st =
  -- trace ("mlp: " ++ show mlp) $
  let input = Torch.asTensor $ V.toList $ maybe id normaliseStateFeature mWel $ st
      (outputTensor, _) = -- V.map toDouble $
        hasktorchModel False mlp input
      output = V.fromList $ Torch.asValue outputTensor
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
      (outputs, mH0C0s) = hasktorchModel True model inputs
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
