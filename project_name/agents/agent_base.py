import jax
from typing import Tuple, Any
import chex
from functools import partial


class AgentBase:
    def create_train_state(self) -> Tuple[Any, Any]:
        raise NotImplementedError

    def reset_memory(self, mem_state) -> Any:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def meta_policy(self, mem_state):
        return mem_state

    def act(self, train_state: Any, mem_state: Any, ac_in: chex.Array, key: chex.PRNGKey) -> Tuple[Any, chex.Array, chex.Array, chex.Array, chex.PRNGKey]:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state: Any, agent: int, traj_batch: chex.Array) -> Tuple[Any, Any, Any, chex.Array, chex.PRNGKey]:
        return runner_state

    @partial(jax.jit, static_argnums=(0,))
    def meta_update(self, runner_state: Any, agent: int, traj_batch: chex.Array) -> Tuple[Any, Any, Any, chex.Array, chex.PRNGKey]:
        return runner_state

    @partial(jax.jit, static_argnums=(0,))
    def update_encoding(self, train_state: Any, mem_state: Any, agent: int, obs_batch: chex.Array, action: chex.Array,
                        reward: chex.Array, done: chex.Array, key: chex.PRNGKey) -> Any:
        return mem_state