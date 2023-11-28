#!/bin/bash

#### Run conv:test_convnet, that is the inference of a trained convnet on the external test set

# The first ($1) input argument passed to this bash script is the number of GPU devices to be used, typically 1 for inference...
NUM_GPUS=$(($1))

arrVar="0" # Here specify the GPU ID of the device to be used, e.g. "0" or "2"
echo "The top $NUM_GPUS free GPUs are: $arrVar"
arrVar2="device="$arrVar
arrVar3="\""$arrVar2"\""
NUM_CPUS=$((24*$NUM_GPUS))

# The second ($2) input argument passed to this bash script must correspond to the absolute path of the trained model to be tested
SAVED_MODEL_PATH=$2

####
SEED=42
EXPERIMENT="breakhis" #"prostate" 
CONDITIONING_FEATURE="aggressiveness" #"disease_yes_no"
IMAGE_SIZE=128 #96 
BATCH_SIZE_TEST=1

######## Be careful to choose the right one depending on what you need###############
MODEL_TYPE="resnet18" # "resnet18", "resnet18_ablation" 
##

causality_setting='cat' #Specify the string of the model type you need based on the saved model you're using: cat,mulcat,mulcatbool
MULCAT_CAUSES_OR_EFFECTS='causes' #'causes','effects'
CAUSALITY_AWARENESS_METHOD='lehmer' #'None' #'lehmer' #'max'
LEHMER_PARAM=1 #it is used only if the above is "lehmer"


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