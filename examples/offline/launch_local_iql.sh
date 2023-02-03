#!/usr/bin/env bash

ds_version=v2
NOW=$(date +"%Y-%m-%d_%H-%M-%S")
export WANDB_RUN_GROUP=iql_offline_mujoco_$NOW
for seed in 0 1 2; do
for domain in walker2d hopper halfcheetah; do
for kind in medium medium-replay medium-expert; do
    env=$domain-$kind-$ds_version
    workdir=./tmp/iql_mujoco_offline_$NOW/${env}_${seed}
    echo Running $env, save to $workdir
    mkdir -p $workdir
    export WANDB_NAME=$env
    python magi/projects/baselines/offline_iql.py \
        --config magi/projects/baselines/configs/iql_mujoco_offline.py \
        --config.env_name=$env \
        --config.eval_episodes=10 \
        --config.eval_interval=10000 \
        --workdir $workdir \
        --log_to_wandb \
        --config.seed $seed
done
done
done
