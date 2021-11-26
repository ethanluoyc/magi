#!/usr/bin/env bash

NOW=$(date +"%Y-%m-%d_%H-%M-%S")
export WANDB_RUN_GROUP=cql_offline_mujoco_$NOW
for seed in 0 42; do
for env in antmaze-medium-play-v0 antmaze-umaze-v0 antmaze-umaze-diverse-v0 antmaze-medium-diverse-v0 antmaze-large-play-v0 antmaze-large-diverse-v0; do
    workdir=./tmp/cql_mujoco_offline_$NOW/${env}_${seed}
    echo Running $env, save to $workdir
    mkdir -p $workdir
    export WANDB_NAME=$env
    python magi/projects/baselines/offline_cql.py \
        --config magi/projects/baselines/configs/cql_antmaze_offline.py \
        --config.env_name=$env \
        --config.eval_episodes=50 \
        --config.eval_interval=10000 \
        --config.log_to_wandb \
        --config.seed $seed \
        --workdir $workdir
done
done
