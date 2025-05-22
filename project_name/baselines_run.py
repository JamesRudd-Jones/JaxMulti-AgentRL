import jax.numpy as jnp
import jax
import jax.random as jrandom
import wandb
from project_name.config import get_config  # TODO dodge need to know how to fix this
from .utils import Transition, EvalTransition, Utils_IMG, Utils_IMPITM, Utils_CG, Utils_DEEPSEA, Utils_KS


def run_train(config, actor, env, env_params, utils):
    def train():
        key = jrandom.PRNGKey(config.SEED)

        train_state, mem_state = actor.initialise()

        key, _key = jrandom.split(key)
        reset_key = jrandom.split(_key, config.NUM_ENVS)
        obs_NO, env_state = jax.vmap(env.reset, in_axes=(0, None), axis_name="batch_axis")(reset_key, env_params)

        runner_state = (train_state, mem_state, env_state, obs_NO,
                        jnp.zeros((config.NUM_AGENTS, config.NUM_ENVS), dtype=bool), key)

        def _run_inner_update(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _run_episode_step(runner_state, unused):
                train_state, mem_state, env_state, obs, last_done_GN, key = runner_state
                obs_batch_GNO = utils.batchify_obs(obs, range(config.NUM_AGENTS), config.NUM_AGENTS, config.NUM_ENVS)

                key, _key = jrandom.split(key)
                mem_state, action_GNA, key = actor.act(train_state, mem_state, obs_batch_GNO, last_done_GN, _key)

                # for cnn maybe
                env_act_NGA = jnp.swapaxes(action_GNA, 0, 1)

                # step in env
                key, _key = jrandom.split(key)
                key_step = jrandom.split(_key, config.NUM_ENVS)
                obs, env_state, reward, done_N, info = jax.vmap(env.step, in_axes=(0, 0, 0, None),
                                                              axis_name="batch_axis")(key_step,
                                                                                      env_state,
                                                                                      env_act_NGA,
                                                                                      env_params
                                                                                      )
                info = jax.tree_util.tree_map(lambda x: jnp.swapaxes(jnp.tile(x[:, jnp.newaxis],
                                                                              (1, config.NUM_AGENTS)),
                                                                     0, 1),
                                    info)  # TODO not sure if need this basically
                done_batch_GN = jnp.swapaxes(jnp.tile(done_N[:, jnp.newaxis], (1, config.NUM_AGENTS)), 0, 1)
                reward_batch_GN = utils.batchify(reward, range(config.NUM_AGENTS), config.NUM_AGENTS,
                                              config["NUM_ENVS"]).squeeze(axis=-1)
                nobs_batch_GNO = utils.batchify_obs(obs, range(config.NUM_AGENTS), config.NUM_AGENTS, config.NUM_ENVS)

                mem_state = actor.update_encoding(train_state,
                                                  mem_state,
                                                  nobs_batch_GNO,
                                                  action_GNA,
                                                  reward_batch_GN,
                                                  done_batch_GN,
                                                  key)

                transition = Transition(done_batch_GN,
                                        done_batch_GN,  # TODO why are there two done batches?
                                        action_GNA,
                                        reward_batch_GN,
                                        obs_batch_GNO,
                                        mem_state,
                                        # env_state,  # TODO have added for info purposes
                                        info,
                                        )

                return (train_state, mem_state, env_state, obs, done_batch_GN, key), transition

            runner_state, trajectory_batch = jax.lax.scan(_run_episode_step, runner_state, None, config.NUM_INNER_STEPS)
            train_state, mem_state, env_state, obs, done, key = runner_state

            last_obs_batch = utils.batchify_obs(obs, range(config.NUM_AGENTS), config.NUM_AGENTS, config.NUM_ENVS)
            train_state, mem_state, env_state, last_obs_batch, done, agent_info, key = actor.update(train_state,
                                                                                                    mem_state,
                                                                                                    env_state,
                                                                                                    last_obs_batch,
                                                                                                    done,
                                                                                                    key,
                                                                                                    trajectory_batch)

            def callback(traj_batch, env_stats, agent_info, update_steps):
                metric_dict = {  # "env_step": update_steps * config.NUM_ENVS * config.NUM_INNER_STEPS,
                    "env_stats": env_stats
                }

                # TODO below must be sooo slow but maybe it works fine?
                for step_idx in range(config.NUM_META_STEPS):
                    step_metric_dict = {}
                    for idx, agent in enumerate(config.AGENT_TYPE):
                        # shape is [num_meta_steps, num_inner_steps, num_agents, num_envs]
                        step_metric_dict[f"avg_reward_{agent}"] = traj_batch.reward[step_idx, :, idx, :].mean()
                        # step_metric_dict[f"avg_reward_{agent}_{idx}"] = traj_batch.reward[step_idx, :, idx, :].mean()
                        # step_metric_dict[f"avg_reward_{agent}_{idx}"] = traj_batch.reward[step_idx, -1, idx, :].mean()
                        # TODO have added the -1 for deepsea
                        if agent != "MFOS":
                            for item in agent_info[idx]:
                                step_metric_dict[f"{agent}-{item}"] = agent_info[idx][item][step_idx]
                                # step_metric_dict[f"{agent}_{idx}-{item}"] = agent_info[idx][item][step_idx]
                    wandb.log(step_metric_dict)

                metric_dict["env_step"] = (update_steps + 1) * config.NUM_META_STEPS

                wandb.log(metric_dict)

            env_stats = jax.tree_util.tree_map(lambda x: x.mean(), utils.visitation(env_state,
                                                                                    trajectory_batch,
                                                                                    obs))
            # env_stats = env_state
            # TODO remove these stats if not using them and replace with deepsea stats somehow

            # jax.experimental.io_callback(callback, None, trajectory_batch,
            #                              env_stats, agent_info, update_steps)

            metric = trajectory_batch.info

            update_steps += 1

            return ((train_state, mem_state, env_state, obs, done, key), update_steps), (trajectory_batch, agent_info)

        runner_state, metric = jax.lax.scan(_run_inner_update, (runner_state, 0), None, config.NUM_UPDATES)

        return {"runner_state": runner_state, "metrics": metric}

    return train


def run_eval(config, actor, env, env_params, utils, train_state, mem_state):
    key = jrandom.PRNGKey(config.SEED)

    key, _key = jrandom.split(key)
    obs_O, env_state = env.reset(_key, env_params)
    done = False

    for _ in range(config.NUM_EVAL_STEPS):
        obs_10 = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), obs_O)
        # TODO a quick fix for the batchify_obs feature
        obs_batch_GO = utils.batchify_obs(obs_10, range(config.NUM_AGENTS), config.NUM_AGENTS, 1).squeeze(0)

        key, _key = jrandom.split(key)
        mem_state, action_GA, key = actor.act(train_state, mem_state, obs_batch_GO, done, _key)

        # for cnn maybe
        env_act_GA = action_GA

        # step in env
        key, _key = jrandom.split(key)
        obs, env_state, reward, done, info = env.step(_key, env_state, env_act_GA, env_params)

        env.render(env_state, env_params)


if __name__ == "__main__":
    config = get_config()
    with jax.disable_jit(disable=True):
        train = run_train(config)
        out = train()
