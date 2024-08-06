import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax
import chex
import sys

from functools import partial
from typing import Any, Tuple
import distrax

from project_name.vapor_stuff.algos.network_deepsea import SoftQNetwork, Actor, RandomisedPrior
from flax.training.train_state import TrainState
import optax
import flashbax as fbx
from project_name.vapor_stuff.utils import TransitionNoInfo


class TrainStateCritic(TrainState):  # TODO check gradients do not update target_params
    target_params: flax.core.FrozenDict


class TrainStateRP(TrainState):  # TODO check gradients do not update the static prior
    static_prior_params: flax.core.FrozenDict


class VAPOR_Lite:
    def __init__(self, env, env_params, key, config):
        self.config = config
        self.env = env
        self.env_params = env_params
        self.actor_network = Actor(action_dim=env.action_space(env_params).n)
        self.critic_network = SoftQNetwork(action_dim=env.action_space(env_params).n)
        self.rp_network = RandomisedPrior()

        key, actor_key, critic_key = jrandom.split(key, 3)

        self.actor_params = self.actor_network.init(actor_key,
                                                    jnp.zeros((1, *env.observation_space(env_params).shape, 1)))
        self.critic_params = self.critic_network.init(critic_key,
                                                      jnp.zeros((1, *env.observation_space(env_params).shape, 1)))

        self.per_buffer = fbx.make_prioritised_flat_buffer(max_length=config.BUFFER_SIZE,
                                                           min_length=config.BATCH_SIZE,
                                                           sample_batch_size=config.BATCH_SIZE,
                                                           add_sequences=True,
                                                           add_batch_size=None,
                                                           priority_exponent=config.REPLAY_PRIORITY_EXP,
                                                           device=config.DEVICE)

    def create_train_state(self, key: chex.Array) -> Tuple[
        type(flax.training.train_state), TrainStateCritic, TrainStateRP, Any, chex.PRNGKey]:  # TODO imrpove checks any
        actor_state = TrainState.create(apply_fn=self.actor_network.apply,
                                        params=self.actor_params,
                                        tx=optax.chain(optax.inject_hyperparams(optax.adam)(self.config.LR, eps=1e-4)),
                                        )
        critic_state = TrainStateCritic.create(apply_fn=self.critic_network.apply,  # TODO check this actually works
                                               params=self.critic_params,
                                               target_params=self.critic_params,
                                               # TODO does this need copying? worth checking to ensure params and target arent the same
                                               tx=optax.chain(
                                                   optax.inject_hyperparams(optax.adam)(self.config.LR, eps=1e-4)),
                                               )

        def create_reward_state(key: chex.PRNGKey) -> TrainStateRP:  # TODO is this the best place to put it all?
            key, _key = jrandom.split(key)
            rp_params = \
                self.rp_network.init(_key,
                                     (jnp.zeros((1, *self.env.observation_space(self.env_params).shape, 1)),
                                      jnp.zeros((1, 1))))["params"]
            reward_state = TrainStateRP.create(apply_fn=self.rp_network.apply,
                                               params=rp_params["trainable"],
                                               static_prior_params=rp_params["static_prior"],
                                               tx=optax.adam(self.config.LR))
            return reward_state

        ensemble_keys = jrandom.split(key, self.config.NUM_ENSEMBLE)
        ensembled_reward_state = jax.vmap(create_reward_state, in_axes=(0))(ensemble_keys)
        # TODO maybe update this to corax from yicheng

        buffer_state = self.per_buffer.init(
            TransitionNoInfo(state=jnp.zeros((*self.env.observation_space(self.env_params).shape, 1)),
                             action=jnp.zeros((1), dtype=jnp.int32),
                             reward=jnp.zeros((1)),
                             ensemble_reward=jnp.zeros((1)),
                             done=jnp.zeros((1), dtype=bool),
                             ))

        return actor_state, critic_state, ensembled_reward_state, buffer_state, key

    @partial(jax.jit, static_argnums=(0,))
    def act(self, actor_params: dict, obs: chex.Array, key: chex.PRNGKey) -> Tuple[
        chex.Array, chex.Array, chex.Array, chex.PRNGKey]:
        key, _key = jrandom.split(key)
        logits = self.actor_network.apply(actor_params, obs)
        policy_dist = distrax.Categorical(logits=logits)
        action = policy_dist.sample(seed=_key)
        log_prob = policy_dist.log_prob(action)
        # action_probs = policy_dist.prob(action)
        action_probs = policy_dist.probs
        z = action_probs == 0.0
        z = z * 1e-8
        log_prob = jnp.log(action_probs + z)  # TODO idk if this is right but eyo

        # pi_s, log_pi_s = self.actor_network.apply(actor_params, obs)
        # action = jnp.argmax(pi_s, axis=1)  # TODO is it an argmax or is it a sample?
        #
        # return action, log_pi_s, pi_s, key

        return action, log_prob, action_probs, key

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward_noise(self, ensrpr_state, obs: chex.Array, actions: chex.Array) -> chex.Array:
        ensemble_obs = jnp.repeat(obs[jnp.newaxis, :], self.config.NUM_ENSEMBLE, axis=0)
        ensemble_action = jnp.repeat(actions[jnp.newaxis, :], self.config.NUM_ENSEMBLE, axis=0)

        def single_reward_noise(ind_rpr_state: TrainStateRP, obs: chex.Array, action: chex.Array) -> chex.Array:
            rew_pred = ind_rpr_state.apply_fn({"params": {"static_prior": ind_rpr_state.static_prior_params,
                                                          "trainable": ind_rpr_state.params}},
                                              (obs, action))
            return rew_pred

        ensembled_reward = jax.vmap(single_reward_noise)(ensrpr_state,
                                                         ensemble_obs,
                                                         ensemble_action)

        ensembled_reward = self.config.SIGMA_SCALE * jnp.std(ensembled_reward, axis=0)
        ensembled_reward = jnp.minimum(ensembled_reward, 1)

        return ensembled_reward

    @partial(jax.jit, static_argnums=(0,))
    def _reward_noise_over_actions(self, ensrpr_state: TrainStateRP, obs: chex.Array) -> chex.Array:
        # run the get_reward_noise for each action choice, can probs vmap
        actions = jnp.arange(0, self.env.action_space(self.env_params).n, step=1)[:, jnp.newaxis]
        actions = jnp.tile(actions, obs.shape[0])[:, :, jnp.newaxis]

        obs = jnp.repeat(obs[jnp.newaxis, :], self.env.action_space(self.env_params).n, axis=0)

        reward_over_actions = jax.vmap(self._get_reward_noise, in_axes=(None, 0, 0))(ensrpr_state,
                                                                                     obs,
                                                                                     actions)
        # reward_over_actions = jnp.sum(reward_over_actions, axis=0)  # TODO removed the layer sum
        reward_over_actions = jnp.swapaxes(reward_over_actions[:, :, 0], 0, 1)

        return reward_over_actions

    @partial(jax.jit, static_argnums=(0,))
    def update_target_network(self, critic_state: TrainStateCritic) -> TrainStateCritic:
        critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params,
                                                                                   critic_state.target_params,
                                                                                   self.config.TAU)
                                            )

        return critic_state

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state):
        actor_state, critic_state, ensrpr_state, buffer_state, _, _, _, key = runner_state
        key, _key = jrandom.split(key)
        batch = self.per_buffer.sample(buffer_state, _key)

        # CRITIC training
        def critic_loss(actor_params, critic_params, critic_target_params, batch, key):
            obs = batch.experience.first.state
            action = batch.experience.first.action
            reward = batch.experience.first.reward
            done = batch.experience.first.done
            nobs = batch.experience.second.state

            _, next_state_log_pi, next_state_action_probs, key = self.act(actor_params, nobs, key)

            qf1_next_target = self.critic_network.apply(critic_target_params, nobs)
            # TODO ensure it uses the right params

            next_state_reward_noise = self._reward_noise_over_actions(ensrpr_state, nobs)
            state_action_reward_noise = self._get_reward_noise(ensrpr_state, obs, action)
            min_qf_next_target = next_state_action_probs * (qf1_next_target - (
                    next_state_reward_noise * next_state_action_probs * next_state_log_pi))

            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(axis=1)[:, jnp.newaxis]

            # VAPOR-LITE
            next_q_value = reward + state_action_reward_noise + (1 - done) * self.config.GAMMA * (min_qf_next_target)

            # use Q-values only for the taken actions
            qf1_values = self.critic_network.apply(critic_params, obs)
            qf1_a_values = jnp.take_along_axis(qf1_values, action, axis=1)

            td_error = qf1_a_values - next_q_value  # TODO ensure this okay as other stuff vmaps over time?

            # mse loss below
            qf_loss = jnp.mean(td_error ** 2)  # TODO check this is okay?

            # Get the importance weights.
            importance_weights = (1. / batch.priorities).astype(jnp.float32)
            importance_weights **= self.config.IMPORTANCE_SAMPLING_EXP  # TODO what is this val?
            importance_weights /= jnp.max(importance_weights)

            # reweight
            qf_loss = jnp.mean(importance_weights * qf_loss)
            new_priorities = jnp.abs(td_error) + 1e-7

            return qf_loss, new_priorities[:, 0]  # to remove the last dimensions of new_priorities

        (critic_loss, new_priorities), grads = jax.value_and_grad(critic_loss, argnums=1, has_aux=True)(
            actor_state.params,
            critic_state.params,
            critic_state.target_params,
            batch,
            key
        )

        # rb.update_priority(abs_td)
        buffer_state = self.per_buffer.set_priorities(buffer_state, batch.indices, new_priorities)

        critic_state = critic_state.apply_gradients(grads=grads)

        def actor_loss(actor_params, critic_params, batch, key):
            obs = batch.experience.first.state

            min_qf_values = self.critic_network.apply(critic_params, obs)  # TODO ensure it uses the right params

            state_reward_noise = self._reward_noise_over_actions(ensrpr_state, obs)

            _, log_pi, action_probs, _ = self.act(actor_params, obs, key)

            return jnp.mean(action_probs * ((state_reward_noise * action_probs * log_pi) - min_qf_values))

        actor_loss, grads = jax.value_and_grad(actor_loss, argnums=0)(actor_state.params,
                                                           critic_state.params,
                                                           batch,
                                                           key
                                                           )
        actor_state = actor_state.apply_gradients(grads=grads)

        def train_ensemble(indrpr_state, obs, actions, rewards):
            def reward_predictor_loss(rp_params, prior_params):
                rew_pred = indrpr_state.apply_fn(
                    {"params": {"static_prior": prior_params, "trainable": rp_params}}, (obs, actions))
                loss = jnp.mean(jnp.square(rew_pred - rewards))
                return loss, rew_pred

            (ensemble_loss, reward_preds), grads = jax.value_and_grad(reward_predictor_loss, argnums=0, has_aux=True)(
                indrpr_state.params, indrpr_state.static_prior_params)
            indrpr_state = indrpr_state.apply_gradients(grads=grads)

            return ensemble_loss, indrpr_state

        obs = batch.experience.first.state  # TODO bit messy so should probs clean up
        action = batch.experience.first.action
        jitter_reward = batch.experience.first.ensemble_reward

        ensemble_state = jnp.repeat(obs[jnp.newaxis, :], self.config.NUM_ENSEMBLE, axis=0)
        ensemble_action = jnp.repeat(action[jnp.newaxis, :], self.config.NUM_ENSEMBLE, axis=0)
        ensemble_reward = jnp.repeat(jitter_reward[jnp.newaxis, :], self.config.NUM_ENSEMBLE, axis=0)

        ensembled_loss, ensrpr_state = jax.vmap(train_ensemble)(ensrpr_state,
                                                                     ensemble_state,
                                                                     ensemble_action,
                                                                     ensemble_reward)

        return actor_state, critic_state, ensrpr_state, buffer_state, actor_loss, critic_loss, jnp.mean(ensembled_loss), key
