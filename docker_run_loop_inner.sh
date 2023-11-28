#!/bin/bash

#This bash script is used to run the causality-driven and baseline versions of the networks with all the hyperparameters (optimization) in a loop (inside the python script)
# Indeed, it executes a docker run command which utilizes the conv:train_convnet_loop_inner_B2 docker image. Please customise the arguments below.

################# EXPERIMENT SETTINGS ################
# Set the random seed:
SEED=0 #42,1,0,2
# Hyperparameters tuning
LR="[1e-5]" #"[0.008,0.003]" #"[0.01,0.001]"
WD="[1e-4]" #"[0.01,0.001,0.0001]"
# Causality parameters
CAUSALITY_SETTING="['mulcatbool','mulcat','cat']"
CAUSALITY_AWARENESS_METHOD="['max',None,'lehmer']" #'None' is the baseline
LEHMER_PARAM="[-2,-1,0,1,2]" #"[-100,-2,-1,0,1,100]"
MULCAT_CAUSES_OR_EFFECTS="['causes','effects']"
## other arguments
EXPERIMENT="breakhis" #Depending on the dataset: imagenette| prostate | procancer | breakhis
CONDITIONING_FEATURE="aggressiveness" #imagenette | disease_yes_no, aggressiveness, no_tumour, scanner_vendor
IMAGE_SIZE=128 #96 #224
BATCH_SIZE_TRAIN=12 #24
BATCH_SIZE_VALID=1
BATCH_SIZE_TEST=1
NUMBER_OF_EPOCHS=150 #200
MODEL_TYPE="resnet18" #eg, resnet18, resnext        [alexnet, SimpleCNN, EqualCNN, BaseCNN, LightCNN, , resnet34, resnet50]
which_resnext_to_use="tiny" #eg, "tiny", "base" it is used only when MODEL_TYPE="resnext"
is_pretrained=True









################# SYSTEM GPU/CPU SETTINGS: ################################################################
## The first argument ($1) passed after the bash script corresponds to the number of desired GPU devices to be used, e.g., 1 or 3.
NUM_GPUS=$(($1))

## CHOOSE ONE OF THE TWO FOLLOWING SECTIONS ################################

################ SECTION a) If you use regular (physical) GPU devices, then use this one and comment the other:
arrVar="5" #e.g., "1,2,5" ## GPU ids to be used (single int, or string list of integers)
## create new variables with a specific string formatting necessary to use it in Docker command
arrVar2="device="$arrVar
arrVar3="\""$arrVar2"\""
NUM_CPUS=$((24*$NUM_GPUS)) #generally, it is recommended to use 16-24 times the number of utilized GPUs
echo "Requested $NUM_GPUS GPU(s), utilizing the one(s) with ID: $arrVar"
docker run \
    --interactive --tty \
    --rm \
    --user $(id -u):$(id -g) \
    --cpus $NUM_CPUS \
    --gpus $arrVar3 \
    --volume $PWD:$PWD \
    --workdir $PWD \
    gianlucacarloni/conv:train_convnet_loop_inner_B2 \
                                                    --number_of_gpus $NUM_GPUS \
                                                    --gpus_ids $arrVar \
                                                    --SEED $SEED \
                                                    --EXPERIMENT $EXPERIMENT \
                                                    --CONDITIONING_FEATURE $CONDITIONING_FEATURE \
                                                    --IMAGE_SIZE $IMAGE_SIZE \
                                                    --BATCH_SIZE_TRAIN $BATCH_SIZE_TRAIN \
                                                    --BATCH_SIZE_VALID $BATCH_SIZE_VALID \
                                                    --BATCH_SIZE_TEST $BATCH_SIZE_TEST \
                                                    --NUMBER_OF_EPOCHS $NUMBER_OF_EPOCHS \
                                                    --MODEL_TYPE $MODEL_TYPE \
                                                    --LR $LR \
                                                    --WD $WD \
                                                    --CAUSALITY_AWARENESS_METHOD $CAUSALITY_AWARENESS_METHOD \
                                                    --LEHMER_PARAM $LEHMER_PARAM \
                                                    --CAUSALITY_SETTING $CAUSALITY_SETTING \
                                                    --MULCAT_CAUSES_OR_EFFECTS $MULCAT_CAUSES_OR_EFFECTS \
                                                    --which_resnext_to_use $which_resnext_to_use \
                                                    --is_pretrained $is_pretrained


################ SECTION b) Else, if you want to use NVIDIA Multi-Instance GPU (MIG)(www.nvidia.com/it-it/technologies/multi-instance-gpu/)
## i.e., some physical GPUs are replaced by 'virtual' ones typically used to execute smaller jobs, then you need
## to find their GPU ID with the command 'nvidia-smi -L'.
## E.g., to set the MIG GPU with ID "MIG-5e802a29-4df5-573c-bcad-d27aa83a4bdc", we set the arrVar (bash) to that value:
# arrVar="MIG-5e802a29-4df5-573c-bcad-d27aa83a4bdc" 
## create new variables with a specific string formatting necessary to use it in Docker command
# arrVar2="device="$arrVar
# arrVar3="\""$arrVar2"\""
# NUM_CPUS=$((24*$NUM_GPUS)) #generally, it is recommended to use 16-24 times the number of utilized GPUs
# echo "Requested $NUM_GPUS GPU(s), utilizing the one(s) with ID: $arrVar"
##In this case, we need to specify the flags --runtime nvidia and -e NVIDIA_VISIBLE_DEVICE=$arrVar3, in the docker run command 
# docker run \
#     --interactive --tty \
#     --rm \
#     --user $(id -u):$(id -g) \
#     --cpus $NUM_CPUS \
#     --runtime nvidia \
#     -e NVIDIA_VISIBLE_DEVICE=$arrVar3 \
#     --volume $PWD:$PWD \
#     --workdir $PWD \
#     gianlucacarloni/conv:train_convnet_loop_inner_B2 \
#                                                     --number_of_gpus $NUM_GPUS \
#                                                     --gpus_ids $arrVar \
#                                                     --SEED $SEED \
#                                                     --EXPERIMENT $EXPERIMENT \
#                                                     --CONDITIONING_FEATURE $CONDITIONING_FEATURE \
#                                                     --IMAGE_SIZE $IMAGE_SIZE \
#                                                     --BATCH_SIZE_TRAIN $BATCH_SIZE_TRAIN \
#                                                     --BATCH_SIZE_VALID $BATCH_SIZE_VALID \
#                                                     --BATCH_SIZE_TEST $BATCH_SIZE_TEST \
#                                                     --NUMBER_OF_EPOCHS $NUMBER_OF_EPOCHS \
#                                                     --MODEL_TYPE $MODEL_TYPE \
#                                                     --LR $LR \
#                                                     --WD $WD \
#                                                     --CAUSALITY_AWARENESS_METHOD $CAUSALITY_AWARENESS_METHOD \
#                                                     --LEHMER_PARAM $LEHMER_PARAM \
#                                                     --CAUSALITY_SETTING $CAUSALITY_SETTING \
#                                                     --MULCAT_CAUSES_OR_EFFECTS $MULCAT_CAUSES_OR_EFFECTS \
#                                                     --which_resnext_to_use $which_resnext_to_use \
#                                                     --is_pretrained $is_pretrained
    
echo "END OF ALL TRAINING COMBINATIONS (LOOP)"
