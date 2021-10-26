#!/bin/bash

DATASET_ROOT="/hdd/mliuzzolino/datasets"  # Specify location of datasets
EXPERIMENT_ROOT="/hdd/mliuzzolino/cascaded_nets"  # Specify experiment root
SPLIT_IDXS_ROOT="/hdd/mliuzzolino/split_idxs"  # Specify root of dataset split_idxs

# Experiment name to evaluate
MODEL="resnet18"  # resnet18, resnet34, resnet50, densenet_cifar
DATASET_NAME="ImageNet2012"  # CIFAR10, CIFAR100, TinyImageNet, ImageNet2012
EXPERIMENT_NAME="${MODEL}_${DATASET_NAME}"

TRAIN_MODE="cascaded"  # baseline, cascaded_seq, cascaded
CASCADED_SCHEME="parallel"  # parallel, serial (used for train_mode=cascaded_seq)
DATASET_KEY="val"  # used for train_mode=cascaded_seq
BATCH_SIZE=128

TDL_MODE="OSD"  # OSD, EWS, noise
TDL_ALPHA=0.9
NOISE_VAR=0.0  # Used for noise kernel only
N_TIMESTEPS=70  # Used for EWS kernel only

DEVICE=1
KEEP_LOGITS=true
KEEP_EMBEDDINGS=true
FORCE_OVERWRITE=true
DEBUG=false

cmd=( python eval.py )   # create array with one element
cmd+=( --device $DEVICE )
cmd+=( --dataset_root $DATASET_ROOT )
cmd+=( --dataset_name $DATASET_NAME )
cmd+=( --dataset_key $DATASET_KEY )
cmd+=( --split_idxs_root $SPLIT_IDXS_ROOT )
cmd+=( --experiment_root $EXPERIMENT_ROOT )
cmd+=( --experiment_name $EXPERIMENT_NAME )
cmd+=( --train_mode $TRAIN_MODE )
cmd+=( --batch_size $BATCH_SIZE )
cmd+=( --cascaded_scheme $CASCADED_SCHEME )
cmd+=( --tdl_mode $TDL_MODE )
cmd+=( --tdl_alpha $TDL_ALPHA )
cmd+=( --noise_var $NOISE_VAR )
cmd+=( --n_timesteps $N_TIMESTEPS )

${KEEP_LOGITS} && cmd+=( --keep_logits )
${KEEP_EMBEDDINGS} && cmd+=( --keep_embeddings )
${DEBUG} && cmd+=( --debug ) && echo "DEBUG MODE ENABLED"
${FORCE_OVERWRITE} && cmd+=( --force_overwrite ) && echo "FORCE OVERWRITE"

# Run command
"${cmd[@]}"