#!/usr/bin/env bash

NOW=$(date +"%Y-%m-%d_%H-%M-%S")
export WANDB_RUN_GROUP=cql_offline_mujoco_$NOW
for seed in 0 1; do
for env in antmaze-medium-play-v0 antmaze-umaze-v0 antmaze-umaze-diverse-v0 antmaze-medium-diverse-v0 antmaze-large-play-v0 antmaze-large-diverse-v0; do
    workdir=./tmp/cql_mujoco_offline_$NOW/${env}_${seed}
    echo Running $env, save to $workdir
    mkdir -p $workdir
    export WANDB_NAME=$env
    python magi/projects/baselines/offline_cql.py \
        --config magi/projects/baselines/configs/cql_antmaze_offline.py \
        --config.env_name=$env \
        --config.eval_episodes=100 \
        --config.eval_interval=10000 \
        --config.log_to_wandb \
        --config.seed $seed \
        --workdir $workdir
done
done

ds_version=v2
for seed in 0 1; do
for domain in walker2d hopper halfcheetah; do
for kind in medium medium-replay medium-expert; do
    env=$domain-$kind-$ds_version
    workdir=./tmp/cql_mujoco_offline_$NOW/${env}_${seed}
    echo Running $env, save to $workdir
    mkdir -p $workdir
    export WANDB_NAME=$env
    python magi/projects/baselines/offline_cql.py \
        --config magi/projects/baselines/configs/cql_mujoco_offline.py \
        --config.env_name=$env \
        --config.eval_episodes=10 \
        --config.eval_interval=10000 \
        --workdir $workdir \
        --log_to_wandb \
        --config.seed $seed
done
done
done
