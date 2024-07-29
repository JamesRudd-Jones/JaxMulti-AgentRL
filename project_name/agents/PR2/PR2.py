import sys
import jax
import jax.numpy as jnp
from typing import Any
import jax.random as jrandom
from functools import partial
from project_name.agents.PR2.network import ActorCriticPR2  # TODO sort out this class import ting
import optax
from flax.training.train_state import TrainState
from project_name.utils import MemoryState
import flashbax as fbx


# class TrainStatePR2(TrainState):
#     train_state: TrainState
#     buffer_state: Any


class PR2Agent:
    def __init__(self,
                 env,
                 env_params,
                 key,
                 config):
        self.config = config
        self.env = env
        self.env_params = env_params
        self.network = ActorCriticPR2(action_dim=env.action_space(env_params).n)

        key, _key = jrandom.split(key)

        init_x = (jnp.zeros((1, config.NUM_ENVS, env.observation_space(env_params).n)),
                  jnp.zeros((1, config.NUM_ENVS)),
                  )

        self.network_params = self.network.init(_key, init_x)

        self.per_buffer = fbx.make_prioritised_flat_buffer(max_length=config.BUFFER_SIZE,
                                                           min_length=config.BATCH_SIZE,
                                                           sample_batch_size=config.BATCH_SIZE,
                                                           add_sequences=True,
                                                           add_batch_size=None,
                                                           priority_exponent=config.REPLAY_PRIORITY_EXP,
                                                           device=config.DEVICE)

        def linear_schedule(count):  # TODO put this somewhere better
            frac = (1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"])
            return config["LR"] * frac

        if config["ANNEAL_LR"]:
            self.tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                                  optax.adam(learning_rate=linear_schedule, eps=1e-5),
                                  )
        else:
            self.tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                                  optax.adam(config["LR"], eps=1e-5),
                                  )

    def create_train_state(self):
        return (TrainState.create(apply_fn=self.network.apply,
                                  params=self.network_params,
                                  tx=self.tx),
                self.per_buffer.init(
            TransitionNoInfo(state=jnp.zeros((*self.env.observation_space(self.env_params).shape, 1)),
                             action=jnp.zeros((1), dtype=jnp.int32),
                             reward=jnp.zeros((1)),
                             ensemble_reward=jnp.zeros((1)),
                             done=jnp.zeros((1), dtype=bool),
                             )))

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):  # TODO don't think should ever reset right?
        # mem_state = mem_state._replace(extras={
        #     "values": jnp.zeros((self.config.NUM_ENVS, 1)),
        #     "log_probs": jnp.zeros((self.config.NUM_ENVS, 1)),
        # },
        #     hstate=jnp.zeros((self.config.NUM_ENVS, 1)),
        # )
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def meta_policy(self, mem_state):
        return mem_state

    @partial(jax.jit, static_argnums=(0))
    def act(self, train_state: Any, mem_state: Any, ac_in: Any, key: Any):  # TODO better implement checks
        pi, value, action_logits = train_state.apply_fn(train_state.params, ac_in)
        key, _key = jrandom.split(key)
        action = pi.sample(seed=_key)
        log_prob = pi.log_prob(action)

        return mem_state, action, log_prob, value, key

    @partial(jax.jit, static_argnums=(0))
    def update(self, runner_state, traj_batch):
        # TODO convert traj_batch to buffer and then do sampling here etc, keeps on policy loop but works for our requirements

        train_state, mem_state, env_state, last_obs, last_done, key = runner_state
        key, _key = jrandom.split(key)
        batch = self.per_buffer.sample(mem_state, _key)

        # CRITIC training
        def _loss_fn(actor_params, critic_params, critic_target_params, batch, key):
            obs = batch.experience.first.state
            action = batch.experience.first.action
            reward = batch.experience.first.reward
            done = batch.experience.first.done
            nobs = batch.experience.second.state

            _, next_state_log_pi, next_state_action_probs, key = self.act(actor_params, nobs, key)

            qf_next_target = self.critic_network.apply(critic_target_params, nobs)

            qf_next_target = next_state_action_probs * (qf_next_target)  # - self.config.ALPHA * next_state_log_pi)
            # TODO above have removed entropy from policy evaluation

            # adapt Q-target for discrete Q-function
            qf_next_target = jnp.sum(qf_next_target, axis=1, keepdims=True)

            # VAPOR-LITE
            next_q_value = reward + (1 - done) * self.config.GAMMA * (qf_next_target)
            # TODO should use done or discount? I think these match up

            # use Q-values only for the taken actions
            qf1_values = self.critic_network.apply(critic_params, obs)
            qf1_a_values = jnp.take_along_axis(qf1_values, action, axis=1)

            td_error = qf1_a_values - next_q_value  # TODO ensure this okay as other stuff vmaps over time?

            # mse loss below
            qf_loss = 0.5 * jnp.mean(jnp.square(td_error))  # TODO check this is okay?

            # Get the importance weights.
            importance_weights = (1. / batch.priorities).astype(jnp.float32)
            importance_weights **= self.config.IMPORTANCE_SAMPLING_EXP  # TODO what is this val?
            importance_weights /= jnp.max(importance_weights)

            # reweight
            qf_loss = jnp.mean(importance_weights * qf_loss)
            new_priorities = jnp.abs(td_error) + 1e-7

            obs = batch.experience.first.state

            min_qf_values = self.critic_network.apply(critic_params, obs)  # TODO ensure it uses the right params

            _, log_pi, action_probs, _ = self.act(actor_params, obs, key)

            return jnp.mean(action_probs * ((self.config.ALPHA * log_pi) - min_qf_values))

            return qf_loss, new_priorities[:,0]

        (critic_loss, new_priorities), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            actor_state.params,
            critic_state.params,
            critic_state.target_params,
            batch,
            key
        )

        # rb.update_priority(abs_td)
        buffer_state = self.per_buffer.set_priorities(buffer_state, batch.indices, new_priorities)

        critic_state = critic_state.apply_gradients(grads=grads)


        actor_loss, grads = jax.value_and_grad(actor_loss)(actor_state.params,
                                                           critic_state.params,
                                                           batch,
                                                           key
                                                           )
        actor_state = actor_state.apply_gradients(grads=grads)

        return train_state, mem_state, env_state, last_obs, last_done, key

    @partial(jax.jit, static_argnums=(0,))
    def meta_update(self, runner_state, traj_batch):
        train_state, mem_state, env_state, last_obs, last_done, key = runner_state
        return train_state, mem_state, env_state, last_obs, last_done, key
