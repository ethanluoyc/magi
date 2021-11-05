"""Dataset loader and utilities for D4RL."""
from acme import types
import numpy as np
import reverb
import tensorflow as tf
import tree


def normalize_obs(dataset, eps=1e-3):
    mean = dataset["observations"].mean(axis=0)
    std = dataset["observations"].std(axis=0) + eps
    o_t = (dataset["observations"] - mean) / std
    o_tp1 = (dataset["next_observations"] - mean) / std
    normalized_data = {
        "observations": o_t,
        "next_observations": o_tp1,
        "rewards": dataset["rewards"],
        "actions": dataset["actions"],
        "terminals": dataset["terminals"],
    }
    return (normalized_data, mean, std)


def _process_fn(data):
    info = reverb.SampleInfo(
        key=tf.constant(0, tf.uint64),
        probability=tf.constant(1.0, tf.float64),
        table_size=tf.constant(0, tf.int64),
        priority=tf.constant(1.0, tf.float64),
    )
    data = types.Transition(
        observation=data["observations"],
        action=data["actions"],
        reward=data["rewards"],
        discount=1 - tf.cast(data["terminals"], tf.float32),
        next_observation=data["next_observations"],
        extras=(),
    )
    return reverb.ReplaySample(info, data)


def make_tf_data_iterator(data, batch_size: int):
    dtypes = tree.map_structure(lambda x: x.dtype, data)
    shapes = tree.map_structure(lambda x: (batch_size, *x.shape[1:]), data)
    dataset_size = data["rewards"].shape[0]

    def gen_tensor():
        while True:
            ind = np.random.randint(0, dataset_size, size=batch_size)
            batch = tree.map_structure(lambda x: x[ind], data)
            yield batch

    ds = tf.data.Dataset.from_generator(gen_tensor, dtypes, shapes)
    ds = ds.repeat().map(_process_fn)
    return ds
