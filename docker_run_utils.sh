#!/bin/bash

#After launching this, it will be created an interactive container where you can run
# some fast and easy scripts, such us utils_resizePNGimages_andSave.py, with their input argument from CLI.
docker run \
    --interactive --tty \
    --rm \
    --cpus 48\
    --gpus '"device=1"' \
    --volume $PWD:$PWD \
    --workdir $PWD \
    gianlucacarloni/utils_interactive:v0.1 \
    $@
#--gpus '"device=1,3"' \
## --user $(id -u):$(id -g) \