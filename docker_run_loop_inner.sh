#!/bin/bash

#training convnet in a loop setting to test different hyperparameters: note that trainings have earlystopping criterion
## GPU ids to be used:
arrVar="0"

# arrVar="0,3,5,7"
# arrVar="1,4,6,7"

# arrVar="0"

### Causality parameters
CAUSALITY_SETTING="['mulcatbool']" #"['cat','mulcatbool','mulcat']"
CAUSALITY_AWARENESS_METHOD="['lehmer']" #"['lehmer','max',None]"
LEHMER_PARAM="[1]" #"[-100,-2,-1,0,1,100]"
# added 21 luglio
MULCAT_CAUSES_OR_EFFECTS="['causes']" # just one, that is valid for both, to run the ablation studies


#### Hyperparameters tuning
ADAM_LR="[0.001]" #"[0.01,0.001]"
WD="[0.01]" #"[0.01,0.001,0.0001]"


## fixed (for now) arguments
NUM_GPUS=$(($1))
SEED=42
EXPERIMENT="prostate" #imagenette| prostate
CONDITIONING_FEATURE="disease_yes_no" #imagenette | disease_yes_no, aggressiveness, no_tumour, scanner_vendor
IMAGE_SIZE=128 #256 #96   #128 #64 #imagenette 64, prostate 128
BATCH_SIZE_TRAIN=50 #20 #200 #200
BATCH_SIZE_VALID=1
BATCH_SIZE_TEST=1
NUMBER_OF_EPOCHS=300 #200, 100
MODEL_TYPE="resnext" #eg, EqualCNN, resnet18, resnext_base          alexnet, SimpleCNN, EqualCNN, BaseCNN, LightCNN, , resnet34, resnet50
which_resnext_to_use="tiny" #eg, "tiny", "base"


echo "The top $NUM_GPUS free GPUs are: $arrVar"
arrVar2="device="$arrVar
arrVar3="\""$arrVar2"\""
NUM_CPUS=$((23*$NUM_GPUS))



docker run \
    --interactive --tty \
    --rm \
    --cpus $NUM_CPUS \
    --gpus $arrVar3 \
    --volume $PWD:$PWD \
    --workdir $PWD \
    gianlucacarloni/conv:train_convnet_loop_inner_B2 $NUM_GPUS $arrVar $SEED $EXPERIMENT $CONDITIONING_FEATURE $IMAGE_SIZE $BATCH_SIZE_TRAIN $BATCH_SIZE_VALID $BATCH_SIZE_TEST $NUMBER_OF_EPOCHS $MODEL_TYPE $ADAM_LR $WD $CAUSALITY_AWARENESS_METHOD $LEHMER_PARAM $CAUSALITY_SETTING $MULCAT_CAUSES_OR_EFFECTS --which_resnext_to_use $which_resnext_to_use
    
    # gianlucacarloni/conv:train_convnet_loop_inner_B2_ablation $NUM_GPUS $arrVar $SEED $EXPERIMENT $CONDITIONING_FEATURE $IMAGE_SIZE $BATCH_SIZE_TRAIN $BATCH_SIZE_VALID $BATCH_SIZE_TEST $NUMBER_OF_EPOCHS $MODEL_TYPE $ADAM_LR $WD $CAUSALITY_AWARENESS_METHOD $LEHMER_PARAM $CAUSALITY_SETTING $MULCAT_CAUSES_OR_EFFECTS
    #gianlucacarloni/conv:train_convnet_loop_inner_B $NUM_GPUS $arrVar $SEED $EXPERIMENT $CONDITIONING_FEATURE $IMAGE_SIZE $BATCH_SIZE_TRAIN $BATCH_SIZE_VALID $BATCH_SIZE_TEST $NUMBER_OF_EPOCHS $MODEL_TYPE $ADAM_LR $WD $CAUSALITY_AWARENESS_METHOD $LEHMER_PARAM $CAUSALITY_SETTING
    #gianlucacarloni/conv:train_convnet_loop_inner $NUM_GPUS $arrVar $SEED $EXPERIMENT $CONDITIONING_FEATURE $IMAGE_SIZE $BATCH_SIZE_TRAIN $BATCH_SIZE_VALID $BATCH_SIZE_TEST $NUMBER_OF_EPOCHS $MODEL_TYPE $ADAM_LR $WD $CAUSALITY_AWARENESS_METHOD $LEHMER_PARAM 
    #gianlucacarloni/conv:train_convnet_loop_inner_dict $NUM_GPUS $arrVar $SEED $EXPERIMENT $CONDITIONING_FEATURE $IMAGE_SIZE $BATCH_SIZE_TRAIN $BATCH_SIZE_VALID $BATCH_SIZE_TEST $NUMBER_OF_EPOCHS $MODEL_TYPE "[0.001]" "['lehmer']" "[1]"

#     gianlucacarloni/conv:train_convnet_loop_inner $NUM_GPUS $arrVar $SEED $EXPERIMENT $CONDITIONING_FEATURE $IMAGE_SIZE $BATCH_SIZE_TRAIN $BATCH_SIZE_VALID $BATCH_SIZE_TEST $NUMBER_OF_EPOCHS $MODEL_TYPE "[0.001]" "['lehmer']" "[-2]" 
#$ADAM_LR $CAUSALITY_AWARENESS_METHOD $LEHMER_PARAM

# --user $(id -u):$(id -g) \ ##sarebbero uid=1021(gianlucacarloni) gid=1021(gianlucacarloni)

echo "END OF ALL TRAINING COMBINATIONS (LOOP)"
#gianlucacarloni/conv:train_convnet $NUM_GPUS $arrVar