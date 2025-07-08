import jax.numpy as jnp
import jax
import jax.random as jrandom
import wandb
from project_name.config import get_config  # TODO dodge need to know how to fix this
from .utils import Transition, EvalTransition, Utils_IMG, Utils_IMPITM, Utils_CG, Utils_DEEPSEA, Utils_KS


def run_train(config, actor, env, env_params, utils):
    def train():
        key = jrandom.key(config.SEED)

        train_state, mem_state = actor.initialise()

        key, _key = jrandom.split(key)
        reset_key = jrandom.split(_key, config.NUM_ENVS)
        obs_NGO, env_state = jax.vmap(env.reset)(reset_key)

        runner_state = (train_state,
                        mem_state,
                        env_state,
                        jnp.swapaxes(obs_NGO, 0, 1),
                        jnp.zeros((config.NUM_AGENTS, config.NUM_ENVS), dtype=bool),
                        key)

        def _run_inner_update(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _run_episode_step(runner_state, unused):
                train_state, mem_state, env_state, obs_GNO, last_done_GN, key = runner_state

                key, _key = jrandom.split(key)
                mem_state, action_GNA, key = actor.act(train_state, mem_state, obs_GNO, last_done_GN, _key)

                # step in env
                key, _key = jrandom.split(key)
                key_step = jrandom.split(_key, config.NUM_ENVS)
                nobs_NGO, _, nenv_state, reward_NG, done_N, info = jax.vmap(env.step)(jnp.swapaxes(action_GNA, 0, 1),
                                                                                      env_state,
                                                                                      key_step,
                                                                                      )
                # TODO should there be an individual done and also a global done?
                info = jax.tree_util.tree_map(lambda x: jnp.swapaxes(jnp.tile(x[:, jnp.newaxis],
                                                                              (1, config.NUM_AGENTS)),
                                                                     0, 1),
                                    info)  # TODO not sure if need this basically
                done_batch_GN = jnp.swapaxes(jnp.tile(done_N[:, jnp.newaxis], (1, config.NUM_AGENTS)), 0, 1)
                nobs_GNO = jnp.swapaxes(nobs_NGO, 0, 1)
                reward_GN = jnp.swapaxes(reward_NG, 0, 1)

                mem_state = actor.update_encoding(train_state,
                                                  mem_state,
                                                  jnp.swapaxes(nobs_NGO, 0, 1),
                                                  action_GNA,
                                                  reward_GN,
                                                  done_batch_GN,
                                                  key)

                transition = Transition(done_batch_GN,
                                        done_batch_GN,  # TODO why are there two done batches? refer above re local and global
                                        action_GNA,
                                        reward_GN,
                                        obs_GNO,
                                        mem_state,
                                        # env_state,  # TODO have added for info purposes
                                        info,
                                        )

                return (train_state, mem_state, env_state, nobs_GNO, done_batch_GN, key), transition

            runner_state, trajectory_batch = jax.lax.scan(_run_episode_step, runner_state, None, config.NUM_INNER_STEPS)
            train_state, mem_state, env_state, last_obs_GNO, done_GN, key = runner_state

            train_state, mem_state, agent_info, key = actor.update(train_state,
                                                                   mem_state,
                                                                   env_state,
                                                                   last_obs_GNO,
                                                                   done_GN,
                                                                   key,
                                                                   trajectory_batch)
            # TODO can we remove some items from the update, perhaps env_state?

            def callback(traj_batch, env_stats, agent_info, update_steps):
                metric_dict = {"env_step": update_steps * config.NUM_ENVS * config.NUM_INNER_STEPS,
                               # "env_stats": env_stats,
                               }

                return_values = traj_batch.info["returned_episode_returns"][traj_batch.info["returned_episode"]]
                timesteps = traj_batch.info["timestep"][traj_batch.info["returned_episode"]] * config.NUM_ENVS
                # TODO this must be so slow can we improve the time taken
                for t in range(len(timesteps)):
                    metric_dict["global step"] = timesteps[t]
                    metric_dict["episodic return"] = return_values[t]
                    for idx, agent in enumerate(config.AGENT_TYPE):
                        # shape is [num_meta_steps, num_inner_steps, num_agents, num_envs]
                        metric_dict[f"avg_reward_{agent}"] = traj_batch.reward[t, :, idx, :].mean()
                        # step_metric_dict[f"avg_reward_{agent}_{idx}"] = traj_batch.reward[step_idx, :, idx, :].mean()
                        # step_metric_dict[f"avg_reward_{agent}_{idx}"] = traj_batch.reward[step_idx, -1, idx, :].mean()
                        # TODO have added the -1 for deepsea
                for idx, agent in enumerate(config.AGENT_TYPE):
                    for item in agent_info[idx]:
                        metric_dict[f"{agent}-{item}"] = agent_info[idx][item]
                        # TODO do we even need this here as surely the loss is not per step?

                wandb.log(metric_dict)

            env_stats = jax.tree.map(lambda x: x.mean(), utils.visitation(env_state,
                                                                          trajectory_batch,
                                                                          last_obs_GNO))
            # env_stats = env_state
            # TODO remove these stats if not using them and replace with deepsea stats somehow

            jax.experimental.io_callback(callback, None, trajectory_batch, env_stats, agent_info, update_steps)

            update_steps += 1

            return ((train_state, mem_state, env_state, last_obs_GNO, done_GN, key), update_steps), (trajectory_batch, agent_info)

        runner_state, metric = jax.lax.scan(_run_inner_update, (runner_state, 0), None, config.NUM_UPDATES)

        return {"runner_state": runner_state, "metrics": metric}

    return train


def run_eval(config, actor, env, env_params, utils, train_state, mem_state):
    key = jrandom.key(config.SEED)

    key, _key = jrandom.split(key)
    obs_GO, env_state = env.reset(_key)
    done = False

    for _ in range(config.NUM_EVAL_STEPS):

        key, _key = jrandom.split(key)
        mem_state, action_GA, key = actor.act(train_state, mem_state, obs_GO, done, _key)

        # step in env
        key, _key = jrandom.split(key)
        obs_GO, _, env_state, reward_G, _, info = env.step(action_GA, env_state, _key)

        env.render(env_state, env_params)


if __name__ == "__main__":
    config = get_config()
    with jax.disable_jit(disable=True):
        train = run_train(config)
        out = train()
