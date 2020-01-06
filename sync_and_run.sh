#!/bin/bash


USER=schnecki
PC=c437-pc169
DIR=`pwd`/
SYNC_TIMEOUT=8
BUILDARGS=--flag=borl:debug
FIN=0

if [ ${PWD##*/} == "test" ]; then
    DIR=$DIR../
fi

if [ "$2" != "" ]; then
    PC="$2"
fi

# exit script
exit_script() {
    echo "Exit Signal received. Cleaning up..."
    trap - SIGINT SIGTERM # clear the trap
    kill -- -$$ # Sends SIGTERM to child/sub processes
    # pkill -P $$
    echo "Bye!"
}

# register exit handler
trap exit_script SIGINT SIGTERM SIGKILL

# Loop function for syncing files
function syncLoop() {
    startup=1
    while [ $FIN -eq 0 ]; do
        rsync -tarz $USER@$PC:$DIR/{statePsiVAllStates,costs,plts,statePsiWAllStates,statePsiW2AllStates,stateValues,stateVAllStates,stateWAllStates,stateW2AllStates,stateValuesAllStatesCount,reward,costs,episodeLength,queueLength} . 2>/dev/null
        size=`stat --printf="%s" stateValues`
        i=1
        if [ $startup -eq 1 ] && [ $size -ge 20000000 ]; then
            i=3
        fi
        while [ $i -gt 0 ]; do
            i=$((i-1));
            sleep $SYNC_TIMEOUT;
            wait $!
        done
        startup=0
    done
}

# Sync  & Run
echo "RUNNING ON  $USER@$PC:$DIR"
rsync -tarz --del --force --exclude=.git --exclude=results --exclude=.stack-work --exclude=state* --exclude=costs --exclude=plts --exclude=.state* --exclude=psiValues --exclude=.psiValues  --exclude=episodeLength --exclude=queueLength --exclude=reward $DIR $USER@$PC:$DIR
if [ $? -ne 0 ]; then
    ssh -t $USER@$PC "mkdir -p $DIR 1>/dev/null"
    rsync -tarz --del --force --exclude=.git --exclude=.stack-work --exclude=state*  --exclude=costs --exclude=plts --exclude=.state* --exclude=psiValues --exclude=.psiValues --exclude=episodeLength --exclude=queueLength --exclude=reward $DIR $USER@$PC:$DIR/
fi
echo "Synced data via rsync. Result $?"
if [ $? -eq 0 ]; then
    printf "Starting file sync fork"
    syncLoop &
    printf "Building and running code on $PC...\n----------------------------------------\n"
    ssh -t $USER@$PC "source ~/.bashrc; cd $DIR; stack build $BUILDARGS && stack exec $1 1>&1; wait" || true
    printf "Execution stopped...\n----------------------------------------\n"
else
    echo "Something went wrong while syncing. Check the connection"
fi

# Kill all childs (in case of a normal exit)
FIN=1
pkill -P $$
