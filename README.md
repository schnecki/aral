# BORL

Blackwell Optimal Reinforcement Learning

(c) Manuel Schneckenreither

## Prerequisites

You need git, stack (for Haskell, GHC) and tensorflow version 1.09. For ArchLinux using `sudo`:

    sudo pacman -S protobuf snappy
    cd /tmp
    wget https://archive.archlinux.org/packages/t/tensorflow/tensorflow-1.9.0-3-x86_64.pkg.tar.xz
    sudo pacman -U tensorflow-1.9.0-3-x86_64.pkg.tar.xz

## Cloning & Building

Ensure to clone all submodules:

    git clone --recursive git@git.uibk.ac.at:c4371143/blackwell_optimal_rl.git
    cd blackwell_optimal_rl/borl
    stack build --install-ghc


To build in debug mode use (this keeps tracks of the number of visits of the states, thus use with
care):

    stack build --ghc-options "-DDEBUG"
