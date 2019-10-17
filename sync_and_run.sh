#!/bin/bash


USER=schnecki
PC=c437-pc169
DIR=`pwd`
SYNC_TIMEOUT=8
BUILDARGS=--flag=borl:debug


function syncLoop() {
    while true; do
        rsync -tarz $USER@$PC:$DIR/{statePsiVAllStates,statePsiWAllStates,stateValues,stateValuesAllStates,stateValuesAllStatesCount,psiValues,reward,costs,episodeLength,queueLength} . 2>/dev/null
        sleep $SYNC_TIMEOUT;
        wait $!
    done
}


# Sync & Run
echo "RUNNING ON  $USER@$PC:$DIR"
rsync -tarz --del --force --exclude=.git --exclude=.stack-work --exclude=state* --exclude=psiValues --exclude=episodeLength --exclude=queueLength --exclude=reward . $USER@$PC:$DIR/
if [ $? -ne 0 ]; then
    ssh -t $USER@$PC "mkdir -p $DIR 1>/dev/null"
    rsync -tarz --del --force --exclude=.git --exclude=.stack-work --exclude=state* --exclude=psiValues --exclude=episodeLength --exclude=queueLength --exclude=reward . $USER@$PC:$DIR/
fi
echo "Synced data via rsync. Result $?"
if [ $? -eq 0 ]; then
    printf "Starting file sync fork"
    syncLoop &
    rsync -tarz $USER@$PC:$DIR/{state*,psiValues,episodeLength,queueLength} . &
    printf "Building and running code on $PC...\n----------------------------------------\n"
    ssh -t $USER@$PC "source ~/.bashrc; cd $DIR; stack build $BUILDARGS && stack exec $@ 1>&1; wait" || true
    printf "Execution stopped...\n----------------------------------------\n"

else
    echo "Something went wrong while syncing. Check the connection"
fi

# Kill all childs
pkill -P $$
