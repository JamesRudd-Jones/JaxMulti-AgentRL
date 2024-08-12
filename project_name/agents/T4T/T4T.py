import sys
import jax
import jax.numpy as jnp
from typing import Any
import jax.random as jrandom
from functools import partial
from project_name.agents.PPO.network import ActorCritic  # TODO sort out this class import ting
import optax
from flax.training.train_state import TrainState
from project_name.utils import MemoryState


class T4TAgent:
    def __init__(self,
                 env,
                 env_params,
                 key,
                 config):
        self.config = config
        self.env = env
        self.env_params = env_params

    def create_train_state(self):
        return ({"empty_train_state": 0},
                MemoryState(hstate=jnp.zeros((self.config.NUM_ENVS, 1)),
                            extras={
                                "values": jnp.zeros((self.config.NUM_ENVS, 1)),
                                "log_probs": jnp.zeros((self.config.NUM_ENVS, 1)),
                            })
                )

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):
        mem_state = mem_state._replace(extras={
            "values": jnp.zeros((self.config.NUM_ENVS, 1)),
            "log_probs": jnp.zeros((self.config.NUM_ENVS, 1)),
        },
            hstate=jnp.zeros((self.config.NUM_ENVS, 1)),
        )
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def meta_policy(self, mem_state):
        return mem_state

    @partial(jax.jit, static_argnums=(0))
    def act(self, train_state: Any, mem_state: Any, ac_in: Any, key: Any):  # TODO better implement checks
        state = ac_in[0]
        obs = state.argmax(axis=-1)
        obs = obs % 2
        action = jnp.where(obs > 0, 1, 0)

        log_prob = jnp.zeros((self.config.NUM_ENVS, 1))
        value = jnp.zeros((self.config.NUM_ENVS, 1))

        return mem_state, action, log_prob, value, key

    @partial(jax.jit, static_argnums=(0))
    def update(self, runner_state, agent, traj_batch):
        return runner_state

    @partial(jax.jit, static_argnums=(0,2))
    def meta_update(self, runner_state, agent, traj_batch):
        train_state, mem_state, env_state, ac_in, key = runner_state
        return train_state, mem_state, env_state, ac_in, key

    @partial(jax.jit, static_argnums=(0, 3))
    def update_encoding(self, train_state, mem_state, agent, obs_batch, action, reward, done, key):
        return mem_state

