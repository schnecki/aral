# ARAL

Near-Blackwell Optimal *A*verage *R*eward *A*djusted Reinforcement *L*earning

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

    git clone --recursive git@github.com:schnecki/aral.git
    cd aral
    stack build --install-ghc

To build in debug mode use (this writes several files, thus use with care):

    stack build --ghc-options "-DDEBUG"


## Debug in GHCI

    stack ghci --flag aral:debug aral aral-examples:exe:gridworld-step

## Profiling

    cd examples/
    stack build --profile --library-profiling --flag aral:debug
    stack exec --profile -- gridworld-mini +RTS -p
