# BORL

Blackwell Optimal Reinforcement Learning

(c) Manuel Schneckenreither

## Prerequisites

You need git, stack (for Haskell, GHC) and tensorflow version 1.09 and gym (see gym-haskell folder).
For ArchLinux using `sudo` and `yay`:

    sudo pacman -S protobuf snappy
    cd /tmp
    # wget https://archive.archlinux.org/packages/t/tensorflow/tensorflow-1.9.0-3-x86_64.pkg.tar.xz
    # sudo pacman -U tensorflow-1.9.0-3-x86_64.pkg.tar.xz
    wget https://archive.archlinux.org/packages/t/tensorflow/tensorflow-1.14.0-6-x86_64.pkg.tar.xz
    sudo pacman -U tensorflow-1.14.0-6-x86_64.pkg.tar.xz

    yay -S python34                # for yay see https://wiki.archlinux.org/index.php/AUR_helpers
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3.4 get-pip.py --user
    pip3.4 install gym --user
    pip3.4 install atari-py --user


## Cloning & Building

Ensure to clone all submodules:

    git clone --recursive git@git.uibk.ac.at:c4371143/borl.git
    cd borl
    git checkout 3cd2e15
    cd ..
    stack build --install-ghc

    # In case tensorflow-haskell/tensorflow is empty do:
    # rm -rf tensorflow-haskell
    # git clone --recursive https://github.com/schnecki/haskell.git tensorflow-haskell
    # cd tensorflow-haskell


To build in debug mode use (this keeps tracks of the number of visits of the states, thus use with
care):

    stack build --ghc-options "-DDEBUG"


