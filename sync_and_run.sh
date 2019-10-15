#!/bin/bash

USER=schnecki
PC=c437-pc169
DIR=`pwd`
BUILDARGS=--ghc-options -DDEBUG --fast


# Sync & Run
echo "RUNNING ON  $USER@$PC:$DIR"
rsync -tarz --del --force --exclude=.git --exclude=.stack-work --exclude=state* --exclude=psiValues --exclude=episodeLength --exclude=queueLength --exclude=reward . $USER@$PC:$DIR/
if [ $? -ne 0 ]; then
    ssh -t $USER@$PC "mkdir -p $DIR 1>/dev/null"
    rsync -tarz --del --force --exclude=.git --exclude=.stack-work --exclude=state* --exclude=psiValues --exclude=episodeLength --exclude=queueLength --exclude=reward . $USER@$PC:$DIR/
fi
echo "Synced data via rsync. Result $?"
if [ $? -eq 0 ]; then
    printf "Building and running code on $PC...\n----------------------------------------\n"
    ssh -t $USER@$PC "source ~/.bashrc; cd $DIR; stack build $BUILDARGS && stack exec $@ 1>&1; wait" || true
    printf "Execution stopped...\n----------------------------------------\n"
    scp $USER@$PC:$DIR/state* .
    scp $USER@$PC:$DIR/psiValues .
    scp $USER@$PC:$DIR/episodeLength .
    scp $USER@$PC:$DIR/queueLength .
else
    echo "Something went wrong while syncing. Check the connection"
fi
