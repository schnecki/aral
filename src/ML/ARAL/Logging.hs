{-# LANGUAGE TemplateHaskell #-}
module ML.ARAL.Logging
    ( enableARALLogging
    , disableARALLogging
    ) where

import           EasyLogger

enableARALLogging :: LogDestination -> IO ()
enableARALLogging = $(initLogger)

disableARALLogging :: IO ()
disableARALLogging = $(finalizeLogger)
