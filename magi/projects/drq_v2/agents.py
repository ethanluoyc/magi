import jax
import jax.numpy as jnp
import optax

from magi.agents import drq_v2


def drq_v2_agent(config, random_key, spec, logger):
    agent_networks = drq_v2.make_networks(spec, latent_size=config.latent_size)
    agent_key, eval_key = jax.random.split(random_key)
    agent = drq_v2.DrQV2(
        environment_spec=spec,
        networks=agent_networks,
        config=drq_v2.DrQV2Config(
            min_replay_size=config.min_replay_size,
            max_replay_size=config.max_replay_size,
            prefetch_size=config.prefetch_size,
            batch_size=config.batch_size,
            discount=config.discount,
            n_step=config.n_step,
            critic_q_soft_update_rate=config.tau,
            learning_rate=config.lr,
            noise_clip=config.noise_clip,
            sigma=(config.sigma_start, config.sigma_end, config.sigma_schedule_steps),
            samples_per_insert=config.samples_per_insert,
        ),
        seed=int(jax.random.randint(agent_key, (), 0, jnp.iinfo(jnp.int32).max)),
        logger=logger,
    )
    evaluator_network = drq_v2.get_default_behavior_policy(
        agent_networks,
        spec.actions,
        # Turn off action noise in evaluator_network
        optax.constant_schedule(0.0),
    )
    eval_actor = agent.builder.make_actor(
        eval_key, evaluator_network, variable_source=agent
    )
    return agent, eval_actor
