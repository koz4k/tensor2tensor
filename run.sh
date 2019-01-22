#!/bin/sh

# usage: sh run.sh model_id dropout gpu_id


. ~/mbrl_env/bin/activate
export CUDA_VISIBLE_DEVICES=$3
python tensor2tensor/rl/trainer_model_based.py \
    --loop_hparams_set=rlmb_long_stochastic_discrete \
    "--loop_hparams=game=qbert" \
    "--world_model_dir=$HOME/$1/world_model" \
    "--dropout=$2" \
    "--output_dir=$HOME/$1-$2" \
    2>&1 | tee "$HOME/log-$1-$2"
