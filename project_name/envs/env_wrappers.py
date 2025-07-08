from gymnax.environments import environment, spaces
import jax.numpy as jnp
from functools import partial
import jax
import chex
from typing import TypeVar, Optional, Tuple, Union, Dict, Any
import gymnax


TEnvState = TypeVar("TEnvState", bound="EnvState")
TEnvParams = TypeVar("TEnvParams", bound="EnvParams")


class FlattenObservationWrapper:
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__()

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


class NormalisedEnv(environment.Environment):
    def __init__(self, wrapped_env, env_params):
        """
        Normalises obs to be between -1 and 1
        """
        self._wrapped_env = wrapped_env
        self.unnorm_action_space = self._wrapped_env.action_space(env_params)
        self.unnorm_observation_space = self._wrapped_env.observation_space(env_params)
        self.unnorm_obs_space_size = self.unnorm_observation_space.high - self.unnorm_observation_space.low
        self.unnorm_action_space_size = self.unnorm_action_space.high - self.unnorm_action_space.low

    def action_space(self, params=None) -> spaces.Box:
        """Action space of the environment."""
        assert isinstance(
            self._wrapped_env.action_space(params), spaces.Box
        ), "Only Box spaces are supported for now, aka continuous action spaces."
        return spaces.Box(-1, 1, shape=(self.unnorm_action_space.shape[0],))

    def observation_space(self, params=None) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(low=-jnp.ones_like(self.unnorm_observation_space.low,),
                          high=np.ones_like(self.unnorm_observation_space.high,),
                          shape=(self.unnorm_observation_space.shape[0],))

    @property
    def wrapped_env(self):
        return self._wrapped_env

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key: chex.PRNGKey, params: TEnvParams) -> Tuple[chex.Array, TEnvState]:
        unnorm_obs, env_state = self._wrapped_env.reset(key, params)
        return self.normalise_obs(unnorm_obs), env_state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: chex.PRNGKey, state: TEnvState, action: Union[int, float, chex.Array],
             params: TEnvParams) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        unnorm_action = self.unnormalise_action(action)
        unnorm_obs, new_env_state, rew, done, info = self._wrapped_env.step_env(key, state, unnorm_action, params)

        unnorm_delta_obs = info["delta_obs"]
        norm_delta_obs = unnorm_delta_obs / self.unnorm_obs_space_size * 2
        info["delta_obs"] = norm_delta_obs

        return self.normalise_obs(unnorm_obs), new_env_state, rew, done, info

    @partial(jax.jit, static_argnums=(0,))
    def generative_step_env(self, key, norm_obs, action, params):
        unnorm_init_obs = self.unnormalise_obs(norm_obs)
        unnorm_action = self.unnormalise_action(action)
        unnorm_obs, new_env_state, rew, done, info = self._wrapped_env.generative_step_env(key, unnorm_init_obs, unnorm_action, params)

        unnorm_delta_obs = info["delta_obs"]
        norm_delta_obs = unnorm_delta_obs / self.unnorm_obs_space_size * 2
        info["delta_obs"] = norm_delta_obs

        return self.normalise_obs(unnorm_obs), new_env_state, rew, done, info

    def reward_function(self, x, next_obs, params):
        norm_obs = x[..., :self._wrapped_env.obs_dim]
        action = x[..., self._wrapped_env.obs_dim:]
        unnorm_action = self.unnormalise_action(action)
        unnorm_obs = self.unnormalise_obs(norm_obs)
        unnorm_x = jnp.concatenate([unnorm_obs, unnorm_action], axis=-1)
        unnorm_y = self.unnormalise_obs(next_obs)
        rewards = self._wrapped_env.reward_function(unnorm_x, unnorm_y, params)

        return rewards

    def __getattr__(self, attr):
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapped_env)

    def normalise_obs(self, obs):
        low = self.unnorm_observation_space.low
        size = self.unnorm_obs_space_size
        pos_obs = obs - low
        norm_obs = (pos_obs / size * 2) - 1
        return norm_obs

    def unnormalise_obs(self, obs):
        low = self.unnorm_observation_space.low
        size = self.unnorm_obs_space_size
        obs01 = (obs + 1) / 2
        obs_ranged = obs01 * size
        unnorm_obs = obs_ranged + low
        return unnorm_obs

    def unnormalise_action(self, action):
        low = self.unnorm_action_space.low
        size = self.unnorm_action_space_size
        act01 = (action + 1) / 2
        act_ranged = act01 * size
        unnorm_act = act_ranged + low
        return unnorm_act

    def normalise_action(self, action):
        low = self.unnorm_action_space.low
        size = self.unnorm_action_space_size
        pos_action = action - low
        norm_action = (pos_action / size * 2) - 1
        return norm_action


class GymnaxToJaxMARL(object):
    """
    Based off: https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/wrappers/gymnax.py
    """

    num_agents = 1
    agent = 0
    agents = [agent]

    def __init__(self, env_name: str, env_kwargs: dict = {}, env=None):
        self.env_name = env_name

        if env is None:
            self._env, self.env_params = gymnax.make(env_name, **env_kwargs)
        else:
            self._env = env
            self.env_params = env.default_params

    @property
    def default_params(self):
        return self.env_params

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions, params=None):
        # print('act', actions[self.agent])
        obs, state, reward, done, info = self._env.step(key, state, actions[self.agent], params)
        obs = (obs,)
        reward = (reward,)
        done = done  # {self.agent: done, "__all__": done}
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        obs = (obs,)
        return obs, state

    def observation_space(self, env_params):
        return self._env.observation_space(env_params)

    def action_space(self):
        return self._env.action_space(self.env_params)

    def __getattr__(self, attr):
        if attr == "_env":
            raise AttributeError()
        return getattr(self._env, attr)


class BifurcaGymToJaxMARL(object):
    """
    Based off: https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/wrappers/gymnax.py
    """

    num_agents = 1
    agent = 0
    agents = [agent]

    def __init__(self, env_name: str, env_kwargs: dict = {}, env=None):
        self.env_name = env_name

        if env is None:
            self._env = gymnax.make(env_name, **env_kwargs)
        else:
            self._env = env

    @partial(jax.jit, static_argnums=(0,))
    def step(self, actions, state, key, params=None):
        # print('act', actions[self.agent])
        obs, delta_obs, state, reward, done, info = self._env.step(actions[self.agent], state, key)
        obs = jnp.expand_dims(obs, axis=0)
        reward = jnp.expand_dims(reward, axis=0)
        done = done  # {self.agent: done, "__all__": done}
        return obs, delta_obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key)
        obs = jnp.expand_dims(obs, axis=0)
        return obs, state

    def observation_space(self):
        return self._env.observation_space()

    def action_space(self):
        return self._env.action_space()

    def __getattr__(self, attr):
        if attr == "_env":
            raise AttributeError()
        return getattr(self._env, attr)