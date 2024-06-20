import distrax
import jax.numpy as jnp
import jax
import jax.random as jrandom
import flax
from flax.training.train_state import TrainState
import random
import numpy as np
import optax
from project_name.config import get_config  # TODO dodge need to know how to fix this
from project_name.algos.network_deepsea import SoftQNetwork, Actor, RandomisedPrior
import wandb
from project_name.buffer.prioritised_buffer import PrioritizedReplayBuffer
import gymnax


class TrainStateCritic(TrainState):
    target_params: flax.core.FrozenDict


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def main():
    config = get_config()

    wandb.init(project="ProbInfMarl",
               entity=config.WANDB_ENTITY,
               config=config,
               group="TESTS",
               mode=config.WANDB
               )

    random.seed(config.SEED)  # TODO remove random
    np.random.seed(config.SEED)  # TODO remove numpy random
    key = jax.random.PRNGKey(config.SEED)
    key, actor_key, critic_key = jrandom.split(key, 3)

    # envs = gym.vector.SyncVectorEnv(
    #     [make_env("BreakoutNoFrameskip-v4", config.SEED, i, False, "who knows m8") for i in range(config.NUM_ENVS)])

    # env = bsuite.load_and_record_to_csv("deep_sea/0", results_dir="results")
    # envs = gym_wrapper.GymFromDMEnv(env)

    envs, env_params = gymnax.make("DeepSea-bsuite", size=8)

    key, _key = jrandom.split(key)
    obs, state = envs.reset(_key, env_params)

    actor = Actor(action_dim=envs.action_space(env_params).n)
    critic = SoftQNetwork(action_dim=envs.action_space(env_params).n)
    randomised_prior = RandomisedPrior()

    actor_params = actor.init(actor_key, jnp.zeros((1, *envs.observation_space(env_params).shape, 1)))
    critic_params = critic.init(critic_key, jnp.zeros((1, *envs.observation_space(env_params).shape, 1)))

    actor_state = TrainState.create(apply_fn=None,
                                    params=actor_params,
                                    tx=optax.chain(optax.inject_hyperparams(optax.adam)(config.LR, eps=1e-4)),
                                    )
    critic_state = TrainStateCritic.create(apply_fn=None,
                                           params=critic_params,
                                           target_params=critic_params,
                                           tx=optax.chain(optax.inject_hyperparams(optax.adam)(config.LR, eps=1e-4)),
                                           )

    def create_reward_state(key, model, config):
        key, _key = jrandom.split(key)
        rp_params = model.init(_key, (jnp.zeros((1, *envs.observation_space(env_params).shape, 1)),
                                      jnp.zeros((1, envs.action_space(env_params).n))))["params"]
        prior_params, reward_params = rp_params["static_prior"], rp_params["trainable"]
        reward_state = TrainState.create(apply_fn=randomised_prior.apply,
                                         params=reward_params,
                                         tx=optax.adam(config.LR))
        return prior_params, reward_state

    ensemble_keys = jrandom.split(key, config.NUM_ENSEMBLE)
    ensembled_prior_params, ensembled_reward_state = jax.vmap(create_reward_state, in_axes=(0, None, None))(
        ensemble_keys, randomised_prior, config)

    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    rb = PrioritizedReplayBuffer(config.BUFFER_SIZE,
                                 jnp.zeros((*envs.observation_space(env_params).shape, 1)),  # added zeros due to shape requirements
                                 envs.action_space(env_params),
                                 config.GAMMA,
                                 nstep=config.NSTEP,
                                 alpha=1.0,
                                 beta_steps=(config.TOTAL_TIMESTEPS - config.LEARNING_STARTS) / 1)

    # rb = ReplayBuffer(config.BUFFER_SIZE,
    #                   envs.single_observation_space,
    #                   envs.single_action_space,
    #                   "cpu",
    #                   optimize_memory_usage=True,
    #                   handle_timeout_termination=False,
    #                   )

    def get_action(actor_params, next_obs: jnp.ndarray, key: jrandom.PRNGKey):  # TODO double check this
        key, _key = jrandom.split(key)
        logits = actor.apply(actor_params, next_obs)
        policy_dist = distrax.Categorical(logits=logits)
        action = policy_dist.sample(seed=_key)
        log_prob = policy_dist.log_prob(action)
        action_probs = policy_dist.probs

        return action, log_prob[:, jnp.newaxis], action_probs

    def get_reward_noise(ensembled_prior_params, ensembled_reward_state, obs, actions):
        ensemble_obs = jnp.repeat(obs[jnp.newaxis, :], config.NUM_ENSEMBLE, axis=0)
        ensemble_action = jnp.repeat(actions[jnp.newaxis, :], config.NUM_ENSEMBLE, axis=0)

        def single_reward_noise(reward_state, prior_params, ob, action):
            rew_pred = randomised_prior.apply({"params": {"static_prior": prior_params,
                                                          "trainable": reward_state.params}},
                                              (ob, action))
            return rew_pred

        ensembled_reward = jax.vmap(single_reward_noise)(ensembled_reward_state, ensembled_prior_params,
                                                         ensemble_obs,
                                                         ensemble_action)

        ensembled_reward = config.SIGMA_SCALE * jnp.std(ensembled_reward, axis=0)
        ensembled_reward = jnp.minimum(ensembled_reward, 1)

        return ensembled_reward

    def reward_noise_over_actions(ensembled_prior_params, ensembled_reward_state, obs):
        # run the get_reward_noise for each action choice, can probs vmap
        actions = jnp.arange(0, envs.action_space(env_params).n, step=1)[:, jnp.newaxis]
        actions = jnp.tile(actions, obs.shape[0])[:, :, jnp.newaxis]

        obs = jnp.repeat(obs[jnp.newaxis, :], envs.action_space(env_params).n, axis=0)

        reward_over_actions = jax.vmap(get_reward_noise, in_axes=(None, None, 0, 0))(ensembled_prior_params,
                                                                                     ensembled_reward_state, obs,
                                                                                     actions)
        reward_over_actions = jnp.sum(reward_over_actions, axis=0)

        return reward_over_actions

    # TRY NOT TO MODIFY: start the game
    key, _key = jrandom.split(key)
    obs, state = envs.reset(_key, env_params)
    for global_step in range(config.TOTAL_TIMESTEPS):
        # ALGO LOGIC: put action logic here
        # epsilon = linear_schedule(config.START_EPS, config.END_EPS, config.EPS_DECAY * config.TOTAL_TIMESTEPS,
        #                           global_step)
        key, _key = jrandom.split(key)
        if global_step < config.LEARNING_STARTS:
            actions = envs.action_space(env_params).sample(_key)
        else:
            actions, _, _ = get_action(actor_state.params, obs, key)

        # TRY NOT TO MODIFY: execute the game and log data.
        nobs, nstate, rewards, terminations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    wandb.log({"episodic_return": info["episode"]["r"],
                               "episodic_length": info["episode"]["l"]})

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        def get_mask(truncations, done):
            return done if not truncations else False

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        # rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        mask = get_mask(truncations, terminations)
        rb.append(obs, actions, rewards, mask, next_obs, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs  # issue with this is means obs gets overruled in the below functions :((

        # ALGO LOGIC: training.
        if global_step > config.LEARNING_STARTS:
            if global_step % config.TRAIN_FREQ == 0:
                # data = rb.sample(config.BATCH_SIZE)


            # update target network
            if global_step % config.TARGET_NETWORK_FREQ == 0:
                critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params,
                                                                                           critic_state.target_params,
                                                                                           config.TAU)
                                                    )
    envs.close()


if __name__ == "__main__":
    main()
