module ML.BORL.NeuralNetwork.Tensorflow
    ( backwardRunRepMemData
    ) where

import           Control.Monad        (replicateM, void)
import           Data.List            (foldl')
import qualified Data.Map.Strict      as M
import qualified Data.Vector.Storable as V
import qualified HighLevelTensorflow  as TF

import           ML.BORL.Types

backwardRunRepMemData :: (MonadBorl' m) => Int -> TF.TensorflowModel' -> [[((NetInputWoAction, ActionIndex), Float)]] -> m ()
backwardRunRepMemData iterations model values =
  liftTf $ do
    let valueMap = foldl' (\m ((inp, act), out) -> M.insertWith (++) inp [(act, out)] m) mempty (concat values)
    let inputs = M.keys valueMap
    outputs <- TF.forwardRun model inputs
    let labels = zipWith (flip (foldl' (\vec (idx, groundTruth) -> vec V.// [(idx, groundTruth)]))) (M.elems valueMap) outputs
    void $ replicateM iterations (TF.backwardRun model inputs labels)
