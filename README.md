# BORL

Blackwell Optimal Reinforcement Learning

(c) Manuel Schneckenreither

## Prerequisites

You need git, stack (for Haskell, GHC) and tensorflow version 1.09 and gym (see gym-haskell folder).
For ArchLinux using `sudo` and `yay`:

    sudo pacman -S postgresql-libs snappy blas lapack protobuf glpk
    cd /tmp
    # wget https://archive.archlinux.org/packages/t/tensorflow/tensorflow-1.9.0-3-x86_64.pkg.tar.xz
    # sudo pacman -U tensorflow-1.9.0-3-x86_64.pkg.tar.xz
    wget https://archive.archlinux.org/packages/t/tensorflow/tensorflow-1.14.0-6-x86_64.pkg.tar.xz
    sudo pacman -U tensorflow-1.14.0-6-x86_64.pkg.tar.xz

    # pacman -S python python-pip cmake
    pip install gym --user
    pip install gym[atari] --user


## Cloning & Building

Ensure to *clone all submodules*:

    git clone --recursive git@git.uibk.ac.at:c4371143/borl.git
    cd borl
    stack build --install-ghc

    # In case tensorflow-haskell/tensorflow is empty do:
    # rm -rf tensorflow-haskell
    # git clone --recursive https://github.com/schnecki/haskell.git tensorflow-haskell
    # cd tensorflow-haskell


To build in debug mode use (this writes several files, thus use with care):

    stack build --ghc-options "-DDEBUG"


## Debug in GHCI

    stack ghci --flag borl:debug borl borl-examples:exe:gridworld-step
