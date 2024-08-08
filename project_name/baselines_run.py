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
from .pax.envs.in_the_matrix import InTheMatrix, EnvParams as MatrixEnvParams
from .pax.envs.iterated_matrix_game import IteratedMatrixGame, EnvParams
# from .pax.agents.ppo.ppo import make_agent
from .agents import Agent, MultiAgent
from .utils import Transition, EvalTransition, ipd_visitation, Utils, UtilsCNN
import sys


def run_train(config):
    if config.CNN:
        payoff = jnp.array([[[3, 0], [5, 1]], [[3, 5], [0, 1]]])  # TODO sort this out
        env = InTheMatrix(num_inner_steps=config.NUM_INNER_STEPS, num_outer_steps=config.NUM_META_STEPS,
                          fixed_coin_location=False)
        env_params = MatrixEnvParams(payoff_matrix=payoff, freeze_penalty=5)
        utils = UtilsCNN(config)  # TODO this a bit dodge
    else:
        payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]  # payoff matrix for the IPD
        env = IteratedMatrixGame(num_inner_steps=config.NUM_INNER_STEPS, num_outer_steps=config.NUM_META_STEPS)
        env_params = EnvParams(payoff_matrix=payoff)
        utils = Utils(config)  # TODO this a bit dodge

    def train():
        key = jax.random.PRNGKey(config.SEED)

        if config.NUM_AGENTS == 1:
            print("NOT RIGHT HERE")
            pass
            # actor = Agent(env=env, config=config, key=key)  # TODO fix this m8y
        else:
            actor = MultiAgent(env=env, env_params=env_params, config=config, utils=utils, key=key)
        train_state, mem_state = actor.initialise()

        reset_key = jrandom.split(key, config["NUM_ENVS"])
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None), axis_name="batch_axis")(reset_key, env_params)

        runner_state = (
            train_state, mem_state, env_state, obs, jnp.zeros((config.NUM_AGENTS, config["NUM_ENVS"]), dtype=bool), key)

        def _run_inner_update(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _run_episode_step(runner_state, unused):
                # take initial env_state
                train_state, mem_state, env_state, obs, last_done, key = runner_state
                obs_batch = utils.batchify_obs(obs, range(config.NUM_AGENTS), config.NUM_AGENTS, config["NUM_ENVS"])

                mem_state, action_n, log_prob_n, value_n, key = actor.act(train_state, mem_state, obs_batch, last_done,
                                                                          key)

                # for not cnn
                # env_act = utils.unbatchify(action_n, range(config.NUM_AGENTS), config.NUM_AGENTS, config["NUM_DEVICES"])
                # env_act = {k: v for k, v in env_act.items()}
                # env_act = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), env_act)

                # for cnn maybe
                env_act = jnp.swapaxes(action_n, 0, 1)

                # step in env
                key, _key = jrandom.split(key)
                key_step = jrandom.split(_key, config["NUM_ENVS"])
                obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None),
                                                              axis_name="batch_axis")(key_step,
                                                                                      env_state,
                                                                                      env_act,
                                                                                      env_params
                                                                                      )
                info = jax.tree_map(lambda x: jnp.swapaxes(jnp.tile(x[:, jnp.newaxis], (1, config.NUM_AGENTS)), 0, 1),
                                    info)  # TODO not sure if need this basically
                done_batch = jnp.swapaxes(jnp.tile(done[:, jnp.newaxis], (1, config.NUM_AGENTS)), 0, 1)
                reward_batch = utils.batchify(reward, range(config.NUM_AGENTS), config.NUM_AGENTS,
                                              config["NUM_ENVS"]).squeeze(axis=-1)

                nobs_batch = utils.batchify_obs(obs, range(config.NUM_AGENTS), config.NUM_AGENTS, config["NUM_ENVS"])
                # latent_sample, latent_mean, latent_log_var, encoder_hstate = the below maybe idk where better to put basos
                mem_state = actor.update_encoding(train_state,
                                                  mem_state,
                                                  nobs_batch,
                                                  action_n,
                                                  reward_batch,
                                                  done_batch)

                transition = Transition(done_batch,
                                        done_batch,
                                        action_n,
                                        value_n,
                                        reward_batch,
                                        log_prob_n,
                                        obs_batch,
                                        mem_state,
                                        info,
                                        )

                return (train_state, mem_state, env_state, obs, done_batch, key), transition

            runner_state, trajectory_batch = jax.lax.scan(_run_episode_step, runner_state, None, config.NUM_INNER_STEPS)
            train_state, mem_state, env_state, obs, done, key = runner_state

            # some if statement if MFOS agent to take a meta action too maybe make a mask or something
            mem_state = actor.meta_act(mem_state)  # TODO is there a better way than this?

            # needs the below to add the new trajectory_buffer

            last_obs_batch = utils.batchify_obs(obs, range(config.NUM_AGENTS), config.NUM_AGENTS, config.NUM_ENVS)
            train_state, mem_state, env_state, last_obs_batch, done, key = actor.update(train_state, mem_state,
                                                                                        env_state,
                                                                                        last_obs_batch, done, key,
                                                                                        trajectory_batch)

            def callback(metrics, env_stats):
                metric_dict = {
                    # "returns": metric["returned_episode_returns"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),
                    # This always follows the PB following agent_0
                    # "win_rate": metric["returned_won_episode"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),
                    # "env_step": update_steps * config.NUM_ENVS * config.NUM_INNER_STEPS,
                    "env_stats": env_stats
                    # TODO sort this out as a bit dodge
                }

                for idx, agent in enumerate(config.AGENT_TYPE):
                    metric_dict[f"avg_reward_{agent}_{idx}"] = metrics.reward[:, idx, :].mean()
                    # TODO above is so so dodgy

                wandb.log(metric_dict)

            env_stats = jax.tree_util.tree_map(lambda x: x.mean(), utils.visitation(env_state,
                                                                                    trajectory_batch,
                                                                                    obs))

            jax.experimental.io_callback(callback, None, trajectory_batch, env_stats)

            # update_steps = update_steps + 1

            return ((train_state, mem_state, env_state, obs, done, key), update_steps), trajectory_batch

        def _run_meta_update(meta_runner_state, unused):
            (train_state, mem_state, env_state, obs, last_done, key), update_steps = meta_runner_state

            # reset env here actually I think
            # TODOthis feels dodgy re ending of episodes in trajectories etc but seems what they have done
            key, _key = jrandom.split(key)  # TODO is this necessary?
            reset_key = jrandom.split(_key, config.NUM_ENVS)
            obs, env_state = jax.vmap(env.reset, in_axes=(0, None), axis_name="batch_axis")(reset_key, env_params)

            # reset agents memory apparently as well, do I need this?
            mem_state = actor.reset_memory(mem_state)

            # TODO initialise naive agents here or when using meta agents

            runner_state = (train_state, mem_state, env_state, obs,
                            jnp.zeros((config.NUM_AGENTS, config.NUM_ENVS), dtype=bool), key)

            update_state, meta_trajectory_batch = jax.lax.scan(_run_inner_update, (runner_state, update_steps), None,
                                                               config.NUM_META_STEPS)
            collapsed_trajectory_batch = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x,
                                      [config.NUM_META_STEPS * config.NUM_INNER_STEPS, ] + list(x.shape[2:])),
                meta_trajectory_batch)

            runner_state, update_steps = update_state
            train_state, mem_state, env_state, obs, done, key = runner_state

            last_obs_batch = utils.batchify_obs(obs, range(config.NUM_AGENTS), config.NUM_AGENTS, config.NUM_ENVS)
            train_state, mem_state, env_state, last_obs_batch, done, key = actor.meta_update(train_state, mem_state,
                                                                                             env_state,
                                                                                             last_obs_batch, done, key,
                                                                                             collapsed_trajectory_batch)
            # TODO benchmark if need to output last_obs_batch and env_state and done or not as they don't change?

            metric = meta_trajectory_batch.info
            # update_steps = update_steps + 1  # TODO should add separate update for inner and outer? not sure about this

            # TODO reset memory again?

            return ((train_state, mem_state, env_state, obs, done, key), update_steps), metric

        # TODO some conditional only running the meta_update if required for meta-training at all
        # TODO add something so if do meta or not then they run the same number of total timesteps basically
        runner_state, metric = jax.lax.scan(_run_meta_update, (runner_state, 1), None, config.NUM_UPDATES)
        # TODO else it just runs it num_updates times
        # runner_state, metric = jax.lax.scan(_run_inner_update, (runner_state, 0), None, config.NUM_UPDATES)

        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    config = get_config()
    with jax.disable_jit(disable=True):
        train = run_train(config)
        out = train()
