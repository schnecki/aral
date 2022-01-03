{-# LANGUAGE TemplateHaskell #-}
module ML.BORL.Logging
    ( enableARALLogging
    , disableARALLogging
    ) where

import           EasyLogger

enableARALLogging :: LogDestination -> IO ()
enableARALLogging = $(initLogger)

disableARALLogging :: IO ()
disableARALLogging = $(finalizeLogger)


