#!/bin/bash

#test, inference convnet
NUM_GPUS=$(($1))
NUM_CPUS=$((16*$NUM_GPUS))

SAVED_MODEL_PATH=$2

####
SEED=42
EXPERIMENT="prostate" 
CONDITIONING_FEATURE="disease_yes_no"
IMAGE_SIZE=96 #64 
BATCH_SIZE_TEST=1
MODEL_TYPE="resnet18" #"EqualCNN"
#############
causality_setting='cat' #cat,mulcat,mulcatbool
CAUSALITY_AWARENESS_METHOD='lehmer' #'None' #'lehmer' #'max'
LEHMER_PARAM=-100 #it is used only if the above is "lehmer"
MULCAT_CAUSES_OR_EFFECTS='causes' #'causes','effects', #TODO ################################################
##########
arrVar="4"
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
    gianlucacarloni/conv:test_convnet $NUM_GPUS $arrVar $SAVED_MODEL_PATH $SEED $EXPERIMENT $CONDITIONING_FEATURE $IMAGE_SIZE $BATCH_SIZE_TEST $MODEL_TYPE \
                                      $causality_setting $CAUSALITY_AWARENESS_METHOD $LEHMER_PARAM $MULCAT_CAUSES_OR_EFFECTS

   

#$@ \
# --user $(id -u):$(id -g) \
# tried to add: -v $(pwd)/cache:/local/cache to map the cache folder...but nothing