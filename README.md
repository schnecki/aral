# BORL

Blackwell Optimal Reinforcement Learning

(c) Manuel Schneckenreither

## Prerequisites

You need git, stack (for Haskell, GHC) and tensorflow version 1.09 and gym (see gym-haskell folder).
For ArchLinux using `sudo` and `yay`:

    sudo pacman -S postgresql-libs snappy blas lapack protobuf glpk
    cd /tmp
    wget https://archive.archlinux.org/packages/t/tensorflow/tensorflow-1.9.0-3-x86_64.pkg.tar.xz
    sudo pacman -U tensorflow-1.9.0-3-x86_64.pkg.tar.xz

    yay -S python34                # for yay see https://wiki.archlinux.org/index.php/AUR_helpers
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3.4 get-pip.py --user
    pip3.4 install gym --user
    pip3.4 install atari-py --user


## Cloning & Building

Ensure to clone all submodules:

    git clone --recursive git@git.uibk.ac.at:c4371143/borl.git
    cd borl/tensorflow-haskell
    git checkout 3cd2e15
    cd ..
    stack build --install-ghc


To build in debug mode use (this writes several files, thus use with care):

    stack build --ghc-options "-DDEBUG"


