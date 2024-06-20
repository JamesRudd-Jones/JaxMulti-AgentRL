import jax.numpy as jnp
import jax
import jax.random as jrandom
from project_name.config import get_config  # TODO dodge need to know how to fix this
import wandb
import gymnax
from project_name.algos.vapor_lite import VAPOR_Lite
from project_name.algos.sac import SAC
from typing import NamedTuple
import chex
from project_name.utils import TransitionNoInfo


class Transition(NamedTuple):
    state: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    info: jnp.ndarray


def run_train(config):
    env, env_params = gymnax.make("DeepSea-bsuite", size=8)  # TODO edited the gymnax env for deepsea for info

    def train():
        key = jax.random.PRNGKey(config.SEED)

        actor = VAPOR_Lite(env, env_params, key, config)
        # actor = SAC(env, env_params, key, config)

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

                def random_act(actor_params, obs, key):
                    key, _key = jrandom.split(key)
                    action = env.action_space(env_params).sample(rng=_key)
                    return action[jnp.newaxis], jnp.zeros((1, 1)), jnp.zeros((1, 2)), key

                # act on this initial env_state if above certain outer steps
                action, _, _, key = jax.lax.cond(update_steps > config.LEARNING_STARTS,
                                                                   actor.act,
                                                                   random_act,
                                                                   actor_state.params,
                                                                   obs[jnp.newaxis, :, :, jnp.newaxis],
                                                                   key
                                                                   )
                # action, _, _, key = actor.act(actor_state.params, obs[jnp.newaxis, :, :, jnp.newaxis], key)

                # step in env
                key, _key = jrandom.split(key)
                nobs, nenv_state, reward, done, info = env.step(_key,
                                                                env_state,
                                                                action[0],
                                                                env_params)

                # update tings my dude
                transition = Transition(obs[:, :, jnp.newaxis], action, reward[jnp.newaxis], done[jnp.newaxis], info)

                return (actor_state, critic_state, ensrpr_state, buffer_state, nenv_state, nobs, done, key), transition

            # TODO maybe run for data collection for a while until learning starts

            # run for NUM_STEPS length rollout
            runner_state, trajectory_batch = jax.lax.scan(_run_episode_step, runner_state, None, config["NUM_STEPS"])
            actor_state, critic_state, ensrpr_state, buffer_state, env_state, obs, done, key = runner_state

            buffer_state = actor.per_buffer.add(buffer_state, TransitionNoInfo(state=trajectory_batch.state,
                                                                               action=trajectory_batch.action,
                                                                               reward=trajectory_batch.reward,
                                                                               done=trajectory_batch.done))
            # TODO maybe refresh the above so a bit better
            # needs the below to add the new trajectory_buffer
            runner_state = actor_state, critic_state, ensrpr_state, buffer_state, env_state, obs, done, key

            # update agents here after rollout
            # def fake_update_fn(runner_state):
            #     actor_state, critic_state, ensrpr_state, buffer_state, env_state, obs, done, key = runner_state
            #     return actor_state, critic_state, ensrpr_state, buffer_state, key
            # actor_state, critic_state, ensrpr_state, buffer_state, key = jax.lax.cond(actor.per_buffer.can_sample(buffer_state),
            #                                                                           actor.update,
            #                                                                           fake_update_fn,
            #                                                                           runner_state)

            actor_state, critic_state, ensrpr_state, buffer_state, actor_loss, critic_loss, mean_ensembled_loss, key = actor.update(
                runner_state)

            # metric handling
            # metric = jax.tree_map(lambda x: jnp.swapaxes(x, 1, 2), trajectory_batch.info)
            metric = trajectory_batch.info

            def callback(metric, actor_loss, critic_loss, mean_ensembled_loss):
                # print(metric["update_steps"])
                # print(train_state.params)#["Dense_2"])
                # if metric["update_steps"] == 2:
                #     sys.exit()
                metric_dict = {
                    "returns": metric["returned_episode_returns"][metric["returned_episode"]].mean(),
                    "env_step": metric["update_steps"] * config.NUM_ENVS * config.NUM_STEPS,
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "mean_ensembled_loss": mean_ensembled_loss
                    # TODO add new metric re the 90% thingo idk how to do that tbh but lets get training for now
                }
                wandb.log(metric_dict)

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None,
                                         metric,
                                         actor_loss,
                                         critic_loss,
                                         mean_ensembled_loss)

            update_steps = update_steps + 1

            # update target freq every x update steps
            def critic_fn(critic_state):
                return critic_state

            critic_state = jax.lax.cond(update_steps % config.TARGET_NETWORK_FREQ == 0,
                                        actor.update_target_network,
                                        critic_fn,
                                        critic_state)

            return (
            (actor_state, critic_state, ensrpr_state, buffer_state, env_state, obs, done, key), update_steps), metric

        runner_state, metric = jax.lax.scan(_run_update, (runner_state, 0), None, config["NUM_UPDATES"])

        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    config = get_config()
    with jax.disable_jit(disable=False):
        train = run_train(config)
        out = train()
