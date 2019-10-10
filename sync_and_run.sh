#!/bin/bash

set -e                          # exit on any failure

USER=schnecki
PC=c437-pc169
DIR=`pwd`

echo "RUNNING ON  $USER@$PC:$DIR"

# Sync & Run
ssh -t $USER@$PC "mkdir -p $DIR" 1>/dev/null
rsync -tarz --del --force --exclude=.git --exclude=.stack-work --exclude=state* --exclude=psiValues --exclude=episodeLength --exclude=queueLength --exclude=reward . $USER@$PC:$DIR/
echo "Synced data via rsync. Result $?"
if [ $? -eq 0 ]; then
    printf "Building and running code on $PC...\n----------------------------------------\n"
    ssh -t $USER@$PC "cd $DIR; stack build && stack exec $@ 1>&1; wait" || true
    printf "Execution stopped...\n----------------------------------------\n"
else
    echo "Something went wrong while syncing. Check the connection"
fi
