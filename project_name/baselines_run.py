import jax.numpy as jnp
import jax
import jax.random as jrandom
from project_name.config import get_config  # TODO dodge need to know how to fix this
import wandb
import gymnax
# from project_name.vapor_stuff.algos import VAPOR_Lite
from typing import NamedTuple
import chex
# from project_name.vapor_stuff.utils import TransitionNoInfo
from .pax.envs.iterated_matrix_game import IteratedMatrixGame, EnvParams
# from .pax.agents.ppo.ppo import make_agent
from .agents import Agent, MultiAgent
from .utils import batchify, unbatchify, Transition, EvalTransition
import sys


def run_train(config):
    env = IteratedMatrixGame(num_inner_steps=config.NUM_STEPS, num_outer_steps=1)
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]  # payoff matrix for the IPD
    env_params = EnvParams(payoff_matrix=payoff)

    def train():
        key = jax.random.PRNGKey(config.SEED)

        if config.NUM_AGENTS == 1:
            print("NOT RIGHT HERE")
            pass
            # actor = Agent(env=env, config=config, key=key)  # TODO fix this m8y
        else:
            actor = MultiAgent(env=env, env_params=env_params, config=config, key=key)
        train_state, mem_state = actor.initialise()

        reset_key = jrandom.split(key, config["NUM_ENVS"])
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None), axis_name="batch_axis")(reset_key, env_params)

        runner_state = (
        train_state, mem_state, env_state, obs, jnp.zeros((config.NUM_AGENTS, config["NUM_ENVS"]), dtype=bool), key)

        def _run_inner_update(update_runner_state, unused):
            # TODO needs to reset env here I think always? altho this sounds dodgy for trajectories
            runner_state, update_steps = update_runner_state

            def _run_episode_step(runner_state, unused):
                # take initial env_state
                train_state, mem_state, env_state, obs, last_done, key = runner_state
                obs_batch = batchify(obs, range(config.NUM_AGENTS), config.NUM_AGENTS, config["NUM_ENVS"])

                mem_state, action_n, log_prob_n, value_n, key = actor.act(train_state, mem_state, obs_batch, last_done, key)

                env_act = unbatchify(action_n, range(config.NUM_AGENTS), config.NUM_AGENTS, config["NUM_DEVICES"])
                env_act = {k: v for k, v in env_act.items()}
                env_act = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), env_act)

                # step in env
                key, _key = jrandom.split(key)
                key_step = jrandom.split(_key, config["NUM_ENVS"])
                obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None),
                                                              axis_name="batch_axis")(key_step,
                                                                                      env_state,
                                                                                      env_act,
                                                                                      env_params
                                                                                      )
                info = jax.tree_map(lambda x: jnp.swapaxes(jnp.tile(x[:, jnp.newaxis], (1, config.NUM_AGENTS)), 0, 1), info)  # TODO not sure if need this basically
                done_batch = jnp.swapaxes(jnp.tile(done[:, jnp.newaxis], (1, config.NUM_AGENTS)), 0, 1)

                transition = Transition(
                    done_batch,  # jnp.full((config.NUM_AGENTS, config["NUM_ENVS"]), done["__all__"].reshape((config["NUM_ENVS"]))),
                    done_batch,
                    action_n,
                    value_n,
                    batchify(reward, range(config.NUM_AGENTS), config.NUM_AGENTS, config["NUM_ENVS"]).squeeze(axis=2),
                    log_prob_n,
                    obs_batch,
                    # TODO add meta_actions here somehow as above maybe if there is MFOS around
                    info,
                )

                return (train_state, mem_state, env_state, obs, done_batch, key), transition

            # TODO maybe run for data collection for a while until learning starts

            # run for NUM_STEPS length rollout
            runner_state, trajectory_batch = jax.lax.scan(_run_episode_step, runner_state, None, config.NUM_INNER_STEPS)
            train_state, mem_state, env_state, obs, done, key = runner_state

            # TODO some if statement if MFOS agent to take a meta action too maybe make a mask or something

            # needs the below to add the new trajectory_buffer
            update_state = train_state, mem_state, env_state, obs, done, key
            train_state, mem_state, env_state, obs, done, key = actor.update(update_state, trajectory_batch)
            # TODO some statement in update so only updates if not MFOS or someting idk

            # metric handling
            # metric = jax.tree_map(lambda x: jnp.swapaxes(x, 1, 2), trajectory_batch.info)
            metric = trajectory_batch.info

            update_steps = update_steps + 1

            return ((train_state, mem_state, env_state, obs, done, key), update_steps), metric

        def _run_meta_update(runner_state):
            runner_state, metric = jax.lax.scan(_run_inner_update, (runner_state, 0), None, config.NUM_META_STEPS)
            train_state, mem_state, env_state, obs, done, key = runner_state

            update_steps = update_steps + 1  # TODO sort out update_steps

            # TODO if MFOS update meta agent here

            # TODO reset memory?

            return ((train_state, mem_state, env_state, obs, done, key), update_steps), metric

        # TODO some conditional only running the meta_update if required for meta-training at all
        runner_state, metric = jax.lax.scan(_run_meta_update, (runner_state, 0), None, config.NUM_UPDATES)
        # TODO else it just runs it num_updates times
        runner_state, metric = jax.lax.scan(_run_inner_update, (runner_state, 0), None, config.NUM_UPDATES)

        return {"runner_state": runner_state, "metrics": metric}

    return train


# def callback(metric, train_state):
            #     # if metric["update_steps"] >= 1000 and metric["update_steps"] <= 1250:  # TODO comment this out when don't want it
            #     #     checkpoint_manager.save(metric["update_steps"], train_state)
            #     metric_dict = {
            #         # the metrics have an agent dimension, but this is identical
            #         # for all agents so index into the 0th item of that dimension. not true anymore as hetero babY
            #         "returns": metric["returned_episode_returns"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),
            #         # This always follows the PB following agent_0
            #         "win_rate": metric["returned_won_episode"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),
            #         "env_step": metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]
            #     }
            #     wandb.log(metric_dict)
            #
            # metric["update_steps"] = update_steps
            # jax.experimental.io_callback(callback, None, metric, train_state)


if __name__ == "__main__":
    config = get_config()
    with jax.disable_jit(disable=True):
        train = run_train(config)
        out = train()
