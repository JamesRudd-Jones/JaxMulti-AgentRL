import distrax
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv,
                                                     )
from stable_baselines3.common.buffers import ReplayBuffer
import flax.linen as nn
import jax.numpy as jnp
import jax
import jax.random as jrandom
import flax
import random
import numpy as np
import optax
from project_name.config import get_config  # TODO dodge need to know how to fix this
from project_name.algos_vapor import SoftQNetwork, Actor, RandomisedPrior
import wandb
from project_name.buffer.prioritised_buffer import PrioritizedReplayBuffer
import sys
import bsuite
from bsuite.utils import gym_wrapper
import gymnax
from project_name.vapor_lite import VAPOR_Lite
from typing import NamedTuple
import flashbax as fbx
import chex
from project_name.utils import TransitionNoInfo


class Transition(NamedTuple):
    state: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    info: jnp.ndarray


def run_train(config):
    env, env_params = gymnax.make("DeepSea-bsuite", size=8)

    def train():
        key = jax.random.PRNGKey(config.SEED)

        actor = VAPOR_Lite(env, env_params, key, config)

        # ensrpr relates to ensembled randomised prior reward state
        actor_state, critic_state, ensrpr_state, buffer_state, key = actor.create_train_state(key)

        key, _key = jrandom.split(key)
        obs, env_state = env.reset(_key, env_params)

        runner_state = (
        actor_state, critic_state, ensrpr_state, buffer_state, env_state, obs, jnp.zeros((), dtype=bool), key)

        def _run_update(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _run_episode_step(runner_state, unused):
                # take initial env_state
                actor_state, critic_state, ensrpr_state, buffer_state, env_state, obs, last_done, key = runner_state

                # act on this initial env_state
                action, log_prob, action_probs, key = actor.act(actor_state.params, obs[jnp.newaxis, :, :, jnp.newaxis], key)

                # step in env
                key, _key = jrandom.split(key)
                nobs, nenv_state, reward, done, info = env.step(_key,
                                                                env_state,
                                                                action[0],
                                                                env_params)

                # update tings my dude
                transition = Transition(obs[:, :, jnp.newaxis], action, reward[jnp.newaxis], done[jnp.newaxis], info)

                return (actor_state, critic_state, ensrpr_state, buffer_state, nenv_state, nobs, done, key), transition

            # run for NUM_STEPS length rollout
            runner_state, trajectory_batch = jax.lax.scan(_run_episode_step, runner_state, None, config["NUM_STEPS"])
            _, _, _, buffer_state, env_state, obs, done, _ = runner_state

            buffer_state = actor.per_buffer.add(buffer_state, TransitionNoInfo(state=trajectory_batch.state,
                                                                               action=trajectory_batch.action,
                                                                               reward=trajectory_batch.reward,
                                                                               done=trajectory_batch.done))
            # TODO maybe refresh the above so a bit better

            # update agents here after rollout
            actor_state, critic_state, ensrpr_state, buffer_state, key = actor.update(runner_state)

            # metric handling
            # metric = jax.tree_map(lambda x: jnp.swapaxes(x, 1, 2), trajectory_batch.info)
            metric = trajectory_batch.info

            # def callback(metric, train_state):
            #     # if metric["update_steps"] >= 1000 and metric["update_steps"] <= 1250:  # TODO comment this out when don't want it
            #     #     checkpoint_manager.save(metric["update_steps"], train_state)
            #     metric_dict = {
            #         # the metrics have an agent dimension, but this is identical
            #         # for all agents so index into the 0th item of that dimension. not true anymore as hetero babY
            #         "returns": metric["returned_episode_returns"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),
            #         # This always follows the PB following agent_0
            #         "win_rate": metric["returned_won_episode"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),
            #         "env_step": metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"] + env_step_count_init
            #     }
            #
            #     for agent in env.agents:  # TODO this is cool but win rate is defined by only green fixed point so should be the same for all agents anyway??
            #         metric_dict[f"returns_{agent}"] = metric["returned_episode_returns"][:, :, env.agent_ids[agent]][
            #             metric["returned_episode"][:, :, env.agent_ids[agent]]].mean()
            #         metric_dict[f"win_rate_{agent}"] = metric["returned_won_episode"][:, :, env.agent_ids[agent]][
            #             metric["returned_episode"][:, :, env.agent_ids[agent]]].mean()
            #
            #     wandb.log(metric_dict)
            #
            # metric["update_steps"] = update_steps
            # jax.experimental.io_callback(callback, None, metric, train_state)

            update_steps = update_steps + 1

            return ((actor_state, critic_state, ensrpr_state, buffer_state, env_state, obs, done, key), update_steps), metric

        runner_state, metric = jax.lax.scan(_run_update, (runner_state, 0), None, config["NUM_UPDATES"])

        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    config = get_config()
    train = run_train(config)
    out = train()
