import jax


def rand_seed(key):
    return int(jax.random.randint(key, (), minval=0, maxval=2 ** 31 - 1))
