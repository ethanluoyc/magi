# Baseline results for IQL

## Mujoco locomotion v2

```bash
ds_version=v2
NOW=$(date +"%Y-%m-%d_%H-%M-%S")
export WANDB_RUN_GROUP=iql_offline_mujoco_$NOW
for domain in walker2d hopper halfcheetah; do
for kind in medium medium-replay medium-expert; do
    env=$domain-$kind-$ds_version
    workdir=./tmp/iql_mujoco_offline_$NOW/${env}_${seed}
    echo Running $env, save to $workdir
    mkdir -p $workdir
    export WANDB_NAME=$env
    python magi/projects/baselines/offline_iql.py \
        --config magi/projects/baselines/configs/iql_mujoco_offline.py \
        --env_name=$env \
        --eval_episodes=10 \
        --eval_interval=10000 \
        --workdir $workdir \
        --log_to_wandb \
        --seed 0
done
done
```

Example results running with _one_ seed
                           env          magi        original implementation
         halfcheetah-medium-v2          47.6597     47.4
              hopper-medium-v2          53.8381     66.3
            walker2d-medium-v2          86.4784     78.3
  halfcheetah-medium-replay-v2          44.7481     44.2
       hopper-medium-replay-v2          78.0457     94.7
     walker2d-medium-replay-v2          77.5224     73.9
  halfcheetah-medium-expert-v2          91.7748     86.7
       hopper-medium-expert-v2          103.5875    91.5
     walker2d-medium-expert-v2          108.1659    109.6
