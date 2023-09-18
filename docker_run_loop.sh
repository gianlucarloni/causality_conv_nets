#!/bin/bash

#training convnet in a loop setting to test different hyperparameters: note that trainings have earlystopping criterion
## GPU ids to be used:
arrVar="0,2,3,4,5,6,7"

## Required arguments
NUM_GPUS=$(($1))
SEED=42
EXPERIMENT="imagenette" #imagenette| prostate
CONDITIONING_FEATURE="imagenette" #imagenette | aggressiveness, no_tumour, scanner_vendor, disease_yes_no
IMAGE_SIZE=64 #imagenette 64, prostate 128
BATCH_SIZE_TRAIN=200
BATCH_SIZE_VALID=1
BATCH_SIZE_TEST=1
NUMBER_OF_EPOCHS=100
MODEL_TYPE="EqualCNN" #eg, EqualCNN, alexnet, SimpleCNN, EqualCNN, BaseCNN, LightCNN, resnet18, resnet34, resnet50

## Optional arguments (which have their respective defaults)
CAUSALITY_AWARE="--CAUSALITY_AWARE" #if specified, is True ################# LOOK AT THE BOTTOM ######################################
CAUSALITY_METHOD="lehmer" #max, lehmer ##################################### LOOK AT THE BOTTOM ######################################

echo "The top $NUM_GPUS free GPUs are: $arrVar"
arrVar2="device="$arrVar
arrVar3="\""$arrVar2"\""
NUM_CPUS=$((16*$NUM_GPUS))




## non causality aware run: ################################################
docker run \
    --interactive --tty \
    --rm \
    --user $(id -u):$(id -g) \
    --cpus $NUM_CPUS \
    --gpus $arrVar3 \
    --volume $PWD:$PWD \
    --workdir $PWD \
    gianlucacarloni/conv:train_convnet_loop $NUM_GPUS $arrVar $SEED $EXPERIMENT $CONDITIONING_FEATURE $IMAGE_SIZE $BATCH_SIZE_TRAIN $BATCH_SIZE_VALID $BATCH_SIZE_TEST $NUMBER_OF_EPOCHS 0.001 $MODEL_TYPE                                     
#
docker run \
    --interactive --tty \
    --rm \
    --user $(id -u):$(id -g) \
    --cpus $NUM_CPUS \
    --gpus $arrVar3 \
    --volume $PWD:$PWD \
    --workdir $PWD \
    gianlucacarloni/conv:train_convnet_loop $NUM_GPUS $arrVar $SEED $EXPERIMENT $CONDITIONING_FEATURE $IMAGE_SIZE $BATCH_SIZE_TRAIN $BATCH_SIZE_VALID $BATCH_SIZE_TEST $NUMBER_OF_EPOCHS 0.0003 $MODEL_TYPE                                     
#

## causality aware run ################################################
# for cslty_mthd in {"max","lehmer"}; do
#     if [[ $cslty_mthd == "max" ]]; then
#         echo "salto"
#         for adam_lr in {0.001,0.0003}; do #ADAM LEARNING RATE
#             echo "LOOP: causality method=$cslty_mthd, with adam_lr=$adam_lr"
#             docker run \
#                 --interactive --tty \
#                 --rm \
#                 --user $(id -u):$(id -g) \
#                 --cpus $NUM_CPUS \
#                 --gpus $arrVar3 \
#                 --volume $PWD:$PWD \
#                 --workdir $PWD \
#                 gianlucacarloni/conv:train_convnet_loop $NUM_GPUS $arrVar $SEED $EXPERIMENT $CONDITIONING_FEATURE $IMAGE_SIZE $BATCH_SIZE_TRAIN $BATCH_SIZE_VALID $BATCH_SIZE_TEST $NUMBER_OF_EPOCHS $adam_lr $MODEL_TYPE --CAUSALITY_AWARE --CAUSALITY_METHOD $cslty_mthd                                     
#             # --user $(id -u):$(id -g) \ ##sarebbero uid=1021(gianlucacarloni) gid=1021(gianlucacarloni)
#         done 
#     elif [[ $cslty_mthd == "lehmer" ]]; then
#         for lehmer_param in {-100,-1,0,1,2,100}; do #LEHMER MEAN PARAMETER
#         #for lehmer_param in {-1,0,1,2,100}; do #LEHMER MEAN PARAMETER
#             for adam_lr in {0.001,0.0003}; do #ADAM LEARNING RATE
#                 echo "LOOP: causality method=$cslty_mthd, with adam_lr=$adam_lr and lehmer_param=$lehmer_param"
#                 docker run \
#                     --interactive --tty \
#                     --rm \
#                     --user $(id -u):$(id -g) \
#                     --cpus $NUM_CPUS \
#                     --gpus $arrVar3 \
#                     --volume $PWD:$PWD \
#                     --workdir $PWD \
#                     gianlucacarloni/conv:train_convnet_loop $NUM_GPUS $arrVar $SEED $EXPERIMENT $CONDITIONING_FEATURE $IMAGE_SIZE $BATCH_SIZE_TRAIN $BATCH_SIZE_VALID $BATCH_SIZE_TEST $NUMBER_OF_EPOCHS $adam_lr $MODEL_TYPE --CAUSALITY_AWARE --CAUSALITY_METHOD $cslty_mthd --LEHMER_PARAM $lehmer_param                                     
#                 # --user $(id -u):$(id -g) \ ##sarebbero uid=1021(gianlucacarloni) gid=1021(gianlucacarloni)
#             done
#         done
#     fi
# done
echo "END OF ALL TRAINING COMBINATIONS (LOOP)"
#gianlucacarloni/conv:train_convnet $NUM_GPUS $arrVar