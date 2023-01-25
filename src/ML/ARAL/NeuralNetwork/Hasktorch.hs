{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TupleSections         #-}
module ML.ARAL.NeuralNetwork.Hasktorch
    ( MLP (..)
    , MLPSpec (..)
    , HasktorchLoss (..)
    , HasktorchActivation (..)
    , HasktorchActivationFun (..)
    , runHasktorch
    , trainHasktorch
    ) where

import           Control.Applicative                         ((<|>))
import           Control.Arrow                               (second)
import           Control.DeepSeq
import           Control.Lens                                ((^.))
import           Control.Monad                               (when, zipWithM, (>=>))
import           Data.Default
import           Data.List                                   (foldl', genericLength, intersperse, transpose)
import           Data.Maybe
import           Data.Serialize
import qualified Data.Vector                                 as VB
import qualified Data.Vector.Storable                        as V
import           Data.Word
import           GHC.Generics
import           Statistics.Sample.WelfordOnlineMeanVariance
import           System.IO.Unsafe                            (unsafePerformIO)
import           Text.Printf
import qualified Torch                                       as Torch hiding (dropout)
import qualified Torch.Autograd                              as Torch (toDependent)
import qualified Torch.DType                                 as Torch
import qualified Torch.Functional                            as Torch hiding (dropout)
import qualified Torch.Functional.Internal                   as Torch (dropout, gather, linalg_vector_norm, normAll, row_stack)
import qualified Torch.HList                                 as Torch
import qualified Torch.Initializers                          as Torch
import qualified Torch.Internal.Cast                         as Cast
import qualified Torch.Internal.Managed.Native               as ATen
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

-- | Activation funtion for all layers plus chaning specific activation functions for given index (0 is after input, 1 is after first hidden, etc).
data HasktorchActivation = HasktorchActivation HasktorchActivationFun [(Int, HasktorchActivationFun)]
  deriving (Show, Eq, Generic, Serialize, NFData)

data HasktorchLoss
  = HasktorchHuber
  | HasktorchMSE
  deriving (Show, Eq, Generic, Serialize, NFData, Torch.Parameterized)

instance Default HasktorchLoss where
  def = HasktorchHuber

data HasktorchActivationFun
  = HasktorchRelu
  | HasktorchLeakyRelu (Maybe Float) -- Default alpha = 0.02
  | HasktorchTanh
  | HasktorchSigmoid
  | HasktorchLogSigmoid
  | HasktorchId
  deriving (Show, Eq, Generic, Serialize, NFData)

mkHasktorchDefActivation :: HasktorchActivation -> Torch.Tensor -> Torch.Tensor
mkHasktorchDefActivation (HasktorchActivation def spec) = mkHasktorchActivation def

mkHasktorchSpecActivation :: HasktorchActivation -> [(Int, Torch.Tensor -> Torch.Tensor)]
mkHasktorchSpecActivation (HasktorchActivation _ spec) = map (second mkHasktorchActivation) spec


mkHasktorchActivation :: HasktorchActivationFun -> Torch.Tensor -> Torch.Tensor
mkHasktorchActivation HasktorchId                 = id
mkHasktorchActivation HasktorchRelu               = Torch.relu
mkHasktorchActivation (HasktorchLeakyRelu mAlpha) = Torch.leakyRelu (fromMaybe 0.02 mAlpha)
mkHasktorchActivation HasktorchTanh               = Torch.tanh
mkHasktorchActivation HasktorchSigmoid            = Torch.sigmoid
mkHasktorchActivation HasktorchLogSigmoid         = Torch.logSigmoid

data MLPSpec
  = MLPSpec
      { hasktorchFeatureCounts    :: [Integer]
      , hasktorchHiddenActivation :: HasktorchActivation
      , hasktorchOutputActivation :: Maybe HasktorchActivationFun
      }
  | MLPSpecWDropoutLSTM
      { hasktorchLossFunction       :: Maybe HasktorchLoss -- Default: Huber
      , hasktorchFeatureCounts      :: [Integer]
      , hasktorchHiddenActivation   :: HasktorchActivation
      , hasktorchInputDropoutAlpha  :: Maybe (Bool, Double) -- Dropout layer for input layer with given probability, e.g. 0.2
      , hasktorchHiddenDropoutAlpha :: Maybe (Bool, Double) -- Dropout layer after every hidden layer with given probability, e.g. 0.5
      , hasktorchLSTM               :: Maybe (Int, Int)     -- After first FF layer. Number of layers and hidden nr.
      , hasktorchOutputActivation   :: Maybe HasktorchActivationFun
      }
  deriving (Show, Eq, Generic, NFData)

data MLPSpec_V1
  = MLPSpecWDropoutLSTM_V1
      { v1_hasktorchFeatureCounts      :: [Integer]
      , v1_hasktorchHiddenActivation   :: HasktorchActivation
      , v1_hasktorchInputDropoutAlpha  :: Maybe (Bool, Double) -- Dropout layer for input layer with given probability, e.g. 0.2
      , v1_hasktorchHiddenDropoutAlpha :: Maybe (Bool, Double) -- Dropout layer after every hidden layer with given probability, e.g. 0.5
      , v1_hasktorchLSTM               :: Maybe (Int, Int)     -- After first FF layer. Number of layers and hidden nr.
      , v1_hasktorchOutputActivation   :: Maybe HasktorchActivationFun
      }
  deriving (Show, Eq, Generic, Serialize, NFData)


instance Serialize MLPSpec where
  get = do
    nr <- get :: Get Word8 -- uses smalles of Word8, Word16, Word32, Word64, see https://hackage.haskell.org/package/cereal-0.5.8.3/docs/src/Data.Serialize.html#line-625
    case nr of
      0 -> MLPSpec <$> get <*> get <*> get
      1 -> (MLPSpecWDropoutLSTM <$> get <*> get <*> get <*> get <*> get <*> get <*> get) <|> (fromV1 <$> get)
        where fromV1 (MLPSpecWDropoutLSTM_V1 a b c d e f) = MLPSpecWDropoutLSTM Nothing a b c d e f
      _ -> error $ "Unkown nr: " ++ show nr


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
    { mlpLayers                   :: [Torch.Linear]
    , mlpHiddenActivation         :: Torch.Tensor -> Torch.Tensor -- Default activation
    , mlpSpecificHiddenActivation :: [(Int, Torch.Tensor -> Torch.Tensor)] -- possible specific activations at specified index
    , mlpInputDropoutAlpha        :: Maybe (Bool, Double) -- Nothing: no dropout, Just: (active, alpha)
    , mlpHiddenDropoutAlpha       :: Maybe (Bool, Double) -- Nothing: no dropout, Just: (active, alpha)
    -- , mlpBatchNorm                :: [Torch.BatchNorm]    -- Batchnorm layers if they exist
    , mlpLSTM                     :: Maybe MLP_LSTM -- Nothing: no LSTM, Just: numLayers, h0, c0, LSTM weights and biases
    , mlpOutputActivation         :: Maybe (Torch.Tensor -> Torch.Tensor)
    , mlpLossFun                  :: HasktorchLoss
    }
  deriving (Generic, Torch.Parameterized)

-- instance Torch.Scalar Torch.BatchNorm


instance Show MLP where
  show (MLP layers _ _ mDrI mDr mLSTM _ loss) =
    show layers <>
    maybe "" (\(_, x) -> " w/ Inp Dropout(" ++ printf "%.3f" x ++ ")") mDrI <> maybe "" (\(_, x) -> " w/ Dropout" ++ printf "%.3f" x) mDr <> maybe "" (const " w/ LSTM") mLSTM <> "; Loss: " <> show loss

replaceIdx :: [a] -> (Int, a) -> [a]
replaceIdx xs (idx, x)
  | idx < length xs = take idx xs ++ (x : drop (idx + 1) xs)
  | otherwise = xs

mergeLayers :: [a] -> [a] -> [a] -> [a]
mergeLayers (l:ls) (h:hs) (d:ds) = l : h : d : mergeLayers ls hs ds
mergeLayers (l:ls) (h:hs) []     = l : h : mergeLayers ls hs []
mergeLayers [] [] []             = []
mergeLayers ls hs ds             = error $ "mergeLayers: Unexpected number of layers: " ++ show (length ls, length hs, length ds)

hasktorchModel :: Bool -> MLP -> Maybe Int -> Torch.Tensor -> (Torch.Tensor, Maybe (Torch.Tensor, Torch.Tensor))
hasktorchModel _ (MLP layers hiddenActDef hiddenActSpec mDropoutInput mDropoutHidden Nothing outputAct _) Nothing input =
  (, Nothing) $
  foldl' revApply input $ inputDropoutLayer ++ mergeLayers (map Torch.linear inpAndHiddenLayers) hiddenActs hiddenDropout ++ [Torch.linear outputLayer] ++ maybe [] pure outputAct
  where
    inpAndHiddenLayers = init layers
    outputLayer = last layers
    inputDropoutLayer :: [Torch.Tensor -> Torch.Tensor]
    inputDropoutLayer =
      case mDropoutInput of
        (Just (active, dropoutAlpha)) -> [(\t -> Torch.dropout t dropoutAlpha active)]
        Nothing                       -> []
    hiddenActs :: [Torch.Tensor -> Torch.Tensor]
    hiddenActs
      | null hiddenActSpec = replicate (length inpAndHiddenLayers) hiddenActDef
      | otherwise = foldl' replaceIdx (replicate (length inpAndHiddenLayers) hiddenActDef) hiddenActSpec
    hiddenDropout :: [Torch.Tensor -> Torch.Tensor]
    hiddenDropout =
      case mDropoutHidden of
        (Just (active, dropoutAlpha)) -> replicate (length inpAndHiddenLayers) (\t -> Torch.dropout t dropoutAlpha active)
        Nothing                       -> []
    revApply :: Torch.Tensor -> (Torch.Tensor -> Torch.Tensor) -> Torch.Tensor
    revApply x f = f x
hasktorchModel train (MLP layers hiddenActDef hiddenActSpec mDropoutInput mDropoutHidden (Just (MLP_LSTM numLayers nrInputs _ batchSize h0 c0 (LSTM.LSTMCell wih whh bih bhh))) outputAct _) Nothing input =
  runLstmAndRest $ revApply input layer1
  where
    inpAndHiddenLayers = init layers
    outputLayer = last layers
    inputDropoutLayer :: [Torch.Tensor -> Torch.Tensor]
    inputDropoutLayer =
      case mDropoutInput of
        (Just (active, dropoutAlpha)) -> [(\t -> Torch.dropout t dropoutAlpha active)]
        Nothing                       -> []
    hiddenActs :: [Torch.Tensor -> Torch.Tensor]
    hiddenActs
      | null hiddenActSpec = replicate (length inpAndHiddenLayers) hiddenActDef
      | otherwise = foldl' replaceIdx (replicate (length inpAndHiddenLayers) hiddenActDef) hiddenActSpec
    hiddenDropout :: [Torch.Tensor -> Torch.Tensor]
    hiddenDropout =
      case mDropoutHidden of
        (Just (active, dropoutAlpha)) -> replicate (length inpAndHiddenLayers) (\t -> Torch.dropout t dropoutAlpha active)
        Nothing                       -> []
    revApply :: Torch.Tensor -> (Torch.Tensor -> Torch.Tensor) -> Torch.Tensor
    revApply x f = f x
    dropout = 0 :: Double
    (layer1:layerRest) = inputDropoutLayer ++ mergeLayers (map Torch.linear inpAndHiddenLayers) hiddenActs hiddenDropout ++ [Torch.linear outputLayer] ++ maybe [] pure outputAct
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

randLSTMCell :: Int -> Int -> IO LSTM.LSTMCell
randLSTMCell inputSize hiddenSize = do
  -- x4 dimension calculations - see https://pytorch.org/docs/master/generated/torch.nn.LSTMCell.html
  weightsIH' <- Torch.makeIndependent . mkDbl . initScale =<< Torch.randIO' [4 * hiddenSize, inputSize]
  weightsHH' <- Torch.makeIndependent . mkDbl . initScale =<< Torch.randIO' [4 * hiddenSize, hiddenSize]
  biasIH'    <- Torch.makeIndependent . mkDbl . initScale =<< Torch.randIO' [4 * hiddenSize]
  biasHH'    <- Torch.makeIndependent . mkDbl . initScale =<< Torch.randIO' [4 * hiddenSize]
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
  sample (MLPSpec featCounts activations outputActivation) = Torch.sample (MLPSpecWDropoutLSTM Nothing featCounts activations Nothing Nothing Nothing outputActivation)
  sample (MLPSpecWDropoutLSTM mLossFun featCounts activations mDropoutInput mDropoutHidden mLSTM outputActivation) = do
    let layerSizes = mkLayerSizes (map fromInteger featCounts)
    linears <- mapM sampleDouble layerSizes
    when (length featCounts < 2) $ error "Need at least 1 layer (2 inputs) for ANN"
    let nrAs = last layerSizes
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
        , mlpHiddenActivation = mkHasktorchDefActivation activations
        , mlpSpecificHiddenActivation = mkHasktorchSpecActivation activations
        , mlpInputDropoutAlpha = mDropoutInput
        , mlpHiddenDropoutAlpha = mDropoutHidden
        , mlpLSTM = mLSTMLayer
        , mlpOutputActivation = mkHasktorchActivation <$> outputActivation
        , mlpLossFun = fromMaybe def mLossFun
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
runHasktorch mlp@MLP{} nrAs nrAgents mWel st =
  -- trace ("mlp: " ++ show mlp) $
  let input = Torch.asTensor $ V.toList $ maybe id normaliseStateFeature mWel $ st
      (outputTensor, _) = hasktorchModel False mlp Nothing input
      output = V.fromList $ Torch.asValue outputTensor
   in [toAgents nrAs nrAgents output]


trainHasktorch :: (Torch.Optimizer optimizer) => Period -> Maybe (Int, Double) -> Double -> optimizer -> NNConfig -> MLP -> [[((StateFeatures, ActionIndex), Double)]] -> IO (MLP, optimizer)
trainHasktorch period mSAM lRate optimizer nnConfig model@MLP{} chs
  | nnConfig ^. trainingIterations <= 1 = trainHasktorch' period mSAM lRate optimizer nnConfig model chs
  | otherwise = run (nnConfig ^. trainingIterations) (model, optimizer)
  where run 0 res        = return res
        run n (mdl, opt) = trainHasktorch' period mSAM lRate opt nnConfig mdl chs >>= run (n-1)

trainHasktorch' :: (Torch.Optimizer optimizer) => Period -> Maybe (Int, Double) -> Double -> optimizer -> NNConfig -> MLP -> [[((StateFeatures, ActionIndex), Double)]] -> IO (MLP, optimizer)
trainHasktorch' period mSAM lRate optimizer nnConfig model chs
  --
  -- for LSTM see snd component of output from hasktorchModel
  --
 = do
  let inputs = Torch.asTensor $ map (map (V.toList . fst . fst)) chs
      actions = Torch.asTensor $ map ((: []) . map (snd . fst)) chs
      targets = Torch.asTensor $ map (map snd) chs
      nrRows = Torch.size 0 targets
      nrNStep = Torch.size 1 targets
      gather x = Torch.reshape [nrRows, nrNStep] $ Torch.gather x 2 actions False
      mkLossAndGrads :: MLP -> IO [Torch.Tensor]
      mkLossAndGrads mdl = return . mkLossAndGrads' nnConfig mdl nrRows (Torch.flattenParameters mdl) targets . gather . fst . hasktorchModel True mdl Nothing $ inputs
  grads <- mkLossAndGrads model
  case mRho of
    Nothing -> Torch.runStep' model optimizer (Torch.Gradients grads) lRateTensor
    Just rho -> do
      let params = fmap Torch.toDependent . Torch.flattenParameters $ model
      gradsDetached <- mapM (Torch.clone >=> Torch.detach) grads
      -- compute \hat{\epsilonW}=\rho / \norm{g} \|g\|
      let norm v
            | Torch.dim v == 1 = Torch.linalg_vector_norm v samNorm 0 False Torch.Double
            | otherwise = linalg_matrix_norm_tslbs v samNorm [0 .. dim - 1] False Torch.Double
            where
              dim = Torch.dim v
      gradNorm <- Torch.asValue . norm . Torch.stack (Torch.Dim 0) <$> mapM (fmap norm . Torch.detach) gradsDetached
      let epsilonW = map (Torch.mulScalar (rho / (gradNorm + 1e-12))) gradsDetached
      --
      --
      paramsWEps <- zipWithM (\p -> Torch.makeIndependent . Torch.add p) params epsilonW                              -- virtual step toward \epsilon

      -- let flatParameters = Torch.flattenParameters model
      --     depParameters = fmap Torch.toDependent flatParameters
      -- newFlatParam <- mapM makeIndependent flatParameters'

      -- paramsWEpsFlattened <- mapM Torch.makeIndependent paramsWEps
      -- putStrLn $ "Sizes paramsWEps: " ++ show (map sizes paramsWEps)
      -- putStrLn $ "Sizes params: " ++ show (map sizes params)
      -- putStrLn $ "flattened: " ++ show (map (sizes. Torch.toDependent) $ Torch.flattenParameters paramsWEps)
      --
      -- putStrLn $ "MDL Params BEF: " ++ show (Torch.flattenParameters model) ++ "\n\n"
      let mdl' = Torch.replaceParameters model paramsWEps
      -- paramsWEpsFlattened -- (Torch.flattenParameters paramsWEps)                                   --
      -- putStrLn $ "MDL Params AFT: " ++ show (Torch.flattenParameters mdl') ++ "\n\n"
      gradsNew <- mkLossAndGrads mdl' -- recalculate loss and gradients
      -- let recreateParams = zipWith (Torch.sub . Torch.toDependent) paramsWEps epsilonW      -- virtual step back to the original point
      -- paramsOrig <- zipWithM (\p -> Torch.makeIndependent . Torch.sub p) params epsilonW                              -- virtual step toward \epsilon
      paramsOrig <- mapM Torch.makeIndependent params
      let mdl'' = Torch.replaceParameters mdl' paramsOrig
      -- putStrLn $ "MDL Params AFT AFT: " ++ show (Torch.flattenParameters mdl'') ++ "\n\n"
      --
      -- perform step with new gradients but from old parameters
      Torch.runStep' mdl'' optimizer (Torch.Gradients gradsNew) lRateTensor
  where
    samNorm = 2 -- Experimentally optimized value, see Foret, et al. "Sharpness-aware
                    -- minimization for efficiently improving generalization." (2020).
    samActive = maybe False ((== 0) . (period `mod`) . fst) mSAM
    mRho
      | samActive = snd <$> mSAM
      | otherwise = Nothing
    lRateTensor = Torch.asTensor lRate


mkLossAndGrads' :: NNConfig -> MLP -> Int -> [Torch.Parameter] -> Torch.Tensor -> Torch.Tensor -> [Torch.Tensor]
mkLossAndGrads' nnConfig model nrRows params t o =
  let lss = computeLoss t o
   in mClipGradNorm . foldl1 (zipWith (+)) . map (`Torch.grad` params) $ lss
  where
    computeLoss t o = zipWith lossFun (Torch.split nrRows (Torch.Dim 0) t) (Torch.split nrRows (Torch.Dim 0) o)
    lossFun t o =
      case mlpLossFun model of
        HasktorchHuber -> Torch.smoothL1Loss Torch.ReduceMean o t -- also known as the Huber loss (see docs)
        HasktorchMSE   -> Torch.mseLoss t o
    mClipGradNorm =
      case nnConfig ^. clipGradients of
        ClipByGlobalNorm maxNorm -> unsafePerformIO . clipGradNorm maxNorm
        ClipByValue val          -> unsafePerformIO . clipGradNormType (Just val) 1.0
        _                        -> id


clipGradNorm :: Double -> [Torch.Tensor] -> IO [Torch.Tensor]
clipGradNorm = clipGradNormType (Just 2.0)

clipGradNormType :: Maybe Double -> Double -> [Torch.Tensor] -> IO [Torch.Tensor]
clipGradNormType mNormType maxNorm grads
  -- totalNorm :: Double
 = do
  totalNorm <-
    case mNormType of
      Nothing -> do
        norms <- mapM (fmap (Torch.max . Torch.abs) . Torch.detach) grads
        return $ maximum $ map (Torch.asValue . Torch.max) norms
      Just nt -> Torch.asValue . flip normAll (realToFrac nt) . Torch.row_stack <$> mapM (fmap (`normAll` nt) . Torch.detach) grads
          -- torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
      -- divisor = sqrt . sum $ squaredSums grads
  let clip_coef :: Double
      clip_coef = maxNorm / (totalNorm + 1e-6)
  return $ map (clamp (-clip_coef) clip_coef) grads

  where
    clamp :: Double -> Double -> Torch.Tensor -> Torch.Tensor
    clamp minV maxV input = unsafePerformIO $ Cast.cast3 ATen.clamp_tss input minV maxV
    normAll :: Torch.Tensor -> Double -> Torch.Tensor
    normAll self p = unsafePerformIO $ Cast.cast2 ATen.norm_ts self p

mkSqrdLoss :: Double -> Double -> Double
mkSqrdLoss t o =
  let l = o - t
   in signum l * l ^ (2 :: Int)


sizes :: Torch.Tensor -> [Int]
sizes v = map (`Torch.size` v) [0.. Torch.dim v - 1]


linalg_matrix_norm_tslbs ::
  -- | self
  Torch.Tensor ->
  -- | ord
  Float ->
  -- | dim
  [Int] ->
  -- | keepdim
  Bool ->
  -- | dtype
  Torch.DType ->
  Torch.Tensor
linalg_matrix_norm_tslbs _self _ord _dim _keepdim _dtype = unsafePerformIO $ (Cast.cast5 ATen.linalg_matrix_norm_tslbs) _self _ord _dim _keepdim _dtype
