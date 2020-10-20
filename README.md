# BORL

Blackwell Optimal Reinforcement Learning

(c) Manuel Schneckenreither

## Prerequisites

You need git, stack (for Haskell, GHC) and gym (see gym-haskell folder).
For ArchLinux using `sudo` and `yay`:

    sudo pacman -S postgresql-libs snappy blas lapack protobuf glpk
    cd /tmp

    # pacman -S python python-pip cmake
    pip install gym --user
    pip install gym[atari] --user


## Cloning & Building

Ensure to *clone all submodules*:

    git clone --recursive git@git.uibk.ac.at:c4371143/borl.git
    cd borl
    stack build --install-ghc

To build in debug mode use (this writes several files, thus use with care):

    stack build --ghc-options "-DDEBUG"


## Debug in GHCI

    stack ghci --flag borl:debug borl borl-examples:exe:gridworld-step
