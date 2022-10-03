#!/bin/bash


USER=schnecki
PCS="c437-pc141" # "192.168.0.104" # " c437-pc161 c437-pc147"
# PCS="147.156.22.170"
DIR=`pwd`/
if [ ${PWD##*/} == "test" ]; then
    DIR=$DIR../
fi
if [ ${PWD##*/} == "exec" ]; then
    DIR=$DIR../
fi


ADD_PROJECTS="aral easy-logger grenade experimenter hasktorch welford-online-mean-variance gym-haskell haskell-cpython regress"

# Sync  & Run
for PC in $PCS; do
    for pr in $ADD_PROJECTS; do
        PRDIR="~/Documents/projects/$pr/"
        echo "SYNCING POJECT $pr TO $USER@$PC:$PRDIR"
        rsync -tar -zz --del --force --exclude=.git --exclude=data.new --exclude=results --exclude=ordering --exclude=orders --exclude=.stack-work --exclude=.stack-work.prof --exclude=state* --exclude=costs --exclude=plts --exclude=.state* --exclude=psiValues --exclude=.psiValues  --exclude=episodeLength --exclude=queueLength --exclude=reward* ../$pr/ $USER@$PC:$PRDIR
        if [ $? -ne 0 ]; then
            ssh -t $USER@$PC "mkdir -p $PRDIR 1>/dev/null"
            rsync -tar -zz --del --force --exclude=.git --exclude=data.new --exclude=.stack-work --exclude=state*  --exclude=costs --exclude=plts --exclude=.state* --exclude=psiValues --exclude=.psiValues --exclude=episodeLength --exclude=queueLength --exclude=reward* ../$pr/ $USER@$PC:$PRDIR
        fi
        # .git folder for hasktorch
        if [ "$pr" == "hasktorch" ]; then
            rsync -tar -zz --del --force --exclude=results --exclude=ordering --exclude=orders --exclude=data.new --exclude=.stack-work --exclude=.stack-work.prof --exclude=state* --exclude=costs --exclude=plts --exclude=.state* --exclude=psiValues --exclude=.psiValues  --exclude=episodeLength --exclude=queueLength --exclude=reward* ../hasktorch/.git/ $USER@$PC:$PRDIR/.git
        fi
    done

done
