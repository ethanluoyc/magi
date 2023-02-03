# PETS on Mujoco and Cartpole

This directory includes examples on running the PETS agent on
Mujoco benchmarks described in the original paper and a classical cartpole environment.

Under the hood, it uses `ml_collections` to manage configurations. Optionally
you can log results to wandb.

## Running Mujoco benchmarks
Run
```
python run_cartpole.py --config=configs/cartpole_continuous.py
```
Note that you need should use the `cartpole_continuous.py` config instead of
`cartpole.py` which is used by the Mujoco benchmark.

Run
```
python run_cartpole.py --helpshort
```
For a list of additional configuration besides those that can be overriden with
ml_collections.config_flags

For the classical cartpole, you should consistently get an episode return of 200 (the maximum)
within 10 episodes of training.
On a machine with _NVIDIA GeForce RTX 3080_, 10 episodes
should take less than 2 minutes. If your running time significantly exceeds this value,
make sure that you are installing the GPU version of JAX.

## Running Mujoco benchmarks
Run
```
python run_mujoco.py --config=configs/halfcheetah.py
```
You can change the configuration file to point to any file in `configs/`
