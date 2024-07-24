import jax
import jax.numpy as jnp
import jax.random as jrandom
from project_name.agents.agent_main import Agent
import sys
from ..utils import import_class_from_folder, batchify
from functools import partial
from typing import Any


class MultiAgent(Agent):
    def __init__(self, env, env_params, config, key):
        super().__init__(env, env_params, config, key)
        self.agent_list = {agent: None for agent in range(config.NUM_AGENTS)}  # TODO is there a better way to do this?
        self.mem_state_list = {agent: None for agent in range(config.NUM_AGENTS)}
        self.train_state_list = {agent: None for agent in range(config.NUM_AGENTS)}
        for agent in range(config.NUM_AGENTS):
            self.agent_list[agent] = (
                import_class_from_folder(self.agent_types[agent])(env=env, env_params=env_params, key=key, config=config))
            train_state, init_mem_state = self.agent_list[agent].create_train_state()
            self.mem_state_list[agent] = init_mem_state
            self.train_state_list[agent] = train_state
            # TODO this is dodgy list train_state way, but can we make a custom sub class, is that faster?

    @partial(jax.jit, static_argnums=(0,))
    def initialise(self):
        return self.train_state_list, self.mem_state_list

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: Any, mem_state: Any, obs_batch: Any, last_done: Any, key: Any):  # TODO add better chex
        action_n = jnp.zeros((self.config.NUM_AGENTS, self.config["NUM_ENVS"]), dtype=int)
        value_n = jnp.zeros((self.config.NUM_AGENTS, self.config["NUM_ENVS"]))
        log_prob_n = jnp.zeros((self.config.NUM_AGENTS, self.config["NUM_ENVS"]))
        pi_n = jnp.zeros((self.config.NUM_AGENTS, self.config["NUM_ENVS"], self.env.action_space().n))
        spec_key_n = jnp.zeros((self.config.NUM_AGENTS, 2))
        for agent in range(self.config.NUM_AGENTS):
            ac_in = (obs_batch[jnp.newaxis, agent, :],
                     last_done[jnp.newaxis, agent],
                     )
            ind_mem_state, ind_action, ind_log_prob, ind_value, key, pi, spec_key = self.agent_list[agent].act(train_state[agent],
                                                                                              mem_state[
                                                                                              agent],
                                                                                              ac_in, key)
            action_n = action_n.at[agent].set(ind_action[0])
            value_n = value_n.at[agent].set(ind_value[0])
            log_prob_n = log_prob_n.at[agent].set(ind_log_prob[0])
            pi_n = pi_n.at[agent].set(pi[0])
            spec_key_n = spec_key_n.at[agent].set(spec_key)
            mem_state[agent] = ind_mem_state

        return mem_state, action_n, log_prob_n, value_n, key

    @partial(jax.jit, static_argnums=(0,))
    def update(self, update_state: Any, trajectory_batch: Any):  # TODO add better chex
        train_state, mem_state, env_state, last_obs, last_done, key = update_state
        last_obs_batch = batchify(last_obs, range(self.config.NUM_AGENTS), self.config.NUM_AGENTS, self.config["NUM_ENVS"])
        for agent in range(self.config.NUM_AGENTS):  # TODO this is probs mega slowsies
            # TODO be good to pmap agents as think this is the biggest storage draw, i,e. distribute agents between GPU
            individual_trajectory_batch = jax.tree_map(lambda x: x[:, agent], trajectory_batch)
            individual_train_state = (train_state[agent], env_state, last_obs_batch[agent, :],
                                      last_done[agent],
                                      mem_state[agent], key)
            individual_runner_list = self.agent_list[agent].update(individual_train_state, individual_trajectory_batch)
            train_state[agent] = individual_runner_list[0]
            mem_state[agent] = individual_runner_list[4]
            key = individual_runner_list[-1]

        return train_state, mem_state, env_state, last_obs, last_done, key
