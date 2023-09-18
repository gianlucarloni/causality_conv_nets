#!/bin/bash

#training convnet
NUM_GPUS=$(($1))
NUM_CPUS=$((24*$NUM_GPUS))
#NUM_CPUS=$((24*$NUM_GPUS))

#############
arrVar="0,2,3,4,5,6,7"
#arrVar="0,1,2,3"
#arrVar="4,5,6,7"
#############

echo "The top $NUM_GPUS free GPUs are: $arrVar"
arrVar2="device="$arrVar
arrVar3="\""$arrVar2"\""

docker run \
    --interactive --tty \
    --rm \
    --user $(id -u):$(id -g) \
    --cpus $NUM_CPUS \
    --gpus $arrVar3 \
    --volume $PWD:$PWD \
    --workdir $PWD \
    gianlucacarloni/conv:train_convnet $NUM_GPUS $arrVar

#$@ \
# --user $(id -u):$(id -g) \ ##sarebbero uid=1021(gianlucacarloni) gid=1021(gianlucacarloni)
# tried to add: -v $(pwd)/cache:/local/cache to map the cache folder...but nothing
