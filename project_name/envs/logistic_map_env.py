import numpy as np
from os import path
import jax.numpy as jnp
import jax.random as jrandom
from gymnax.environments import environment
from gymnax.environments import spaces
from flax import struct
from typing import Any, Dict, Optional, Tuple, Union
import chex
import jax


@struct.dataclass
class EnvState(environment.EnvState):
    x: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    init_r: float = 3.8
    max_control: float = 0.1
    horizon: int = 200

class GymnaxLogisticMap(environment.Environment[EnvState, EnvParams]):

    def __init__(self):
        super().__init__()
        self.obs_dim = 1
        self.params = EnvParams()

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()

    def step_env(self,
                 key: chex.PRNGKey,
                 state: EnvState,
                 action: Union[int, float, chex.Array],
                 params: EnvParams) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        new_x = (action + params.init_r) * state.x * (1- state.x)

        reward = jax.lax.select(state.x == new_x, jnp.zeros(1, ), -jnp.ones(1, ))

        done = False  # jax.lax.select(jnp.squeeze(state.x == new_x), True, False)

        state = EnvState(x=new_x, time=state.time+1)

        delta_s = jnp.zeros(1,)  # TODO add in some delta s value

        return (jax.lax.stop_gradient(self.get_obs(state)),
                jax.lax.stop_gradient(state),
                reward,
                done,
                {"delta_obs": delta_s})

    def generative_step_env(self, key, obs, action, params):
        state = EnvState(x=obs[0], time=0)
        return self.step_env(key, state, action, params)

    def reward_function(self, x, next_obs, params: EnvParams):
        """
        As per the paper titled: Optimal chaos control through reinforcement learning
        """
        reward = jax.lax.select(x == next_obs, jnp.zeros(1,), -jnp.ones(1,))
        return reward

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        # high = jnp.array([jnp.pi, 1])
        # init_state = jrandom.uniform(key, shape=(2,), minval=-high, maxval=high)
        state = EnvState(x=jnp.array((0.4,)),
                         time=0)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        return state.x

    @property
    def name(self) -> str:
        """Environment name."""
        return "LogisticMap-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Discrete(3)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array([1.0])
        return spaces.Box(-high, high, (1,), dtype=jnp.float32)

    # TODO add in state space