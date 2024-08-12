import sys
import jax
import jax.numpy as jnp
from typing import Any
import jax.random as jrandom
from functools import partial
from project_name.agents.PR2.network import ActorPR2, CriticPR2, OppNetworkPR2  # TODO sort out this class import ting
import optax
from flax.training.train_state import TrainState
from project_name.utils import MemoryState
import flashbax as fbx
from typing import NamedTuple
import flax
# import tensorflow as tf


class TrainStateExt(TrainState):
    target_params: flax.core.FrozenDict


class TrainStatePR2(NamedTuple):  # TODO is this correct tag?
    critic_state: TrainStateExt
    actor_state: TrainStateExt
    opp_state: TrainState


class TransitionPR2(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray


class PR2Agent:
    def __init__(self,
                 env,
                 env_params,
                 key,
                 config):
        self.config = config
        self.env = env
        self.env_params = env_params
        self.critic_network = CriticPR2(config=config)
        self.actor_network = ActorPR2(action_dim=env.action_space(env_params).n,
                                      config=config)
        self.opp_network = OppNetworkPR2(action_dim=config.NUM_AGENTS - 1, config=config)

        key, _key = jrandom.split(key)

        init_x = (jnp.zeros((1, config.NUM_ENVS, env.observation_space(env_params).n)),
                  jnp.zeros((1, config.NUM_ENVS)),
                  )

        self.critic_network_params = self.critic_network.init(_key, init_x,
                                                              jnp.zeros((1, config.NUM_ENVS, 1)),
                                                              jnp.zeros((1, config.NUM_ENVS, config.NUM_AGENTS - 1)))
        self.actor_network_params = self.actor_network.init(_key, init_x)
        self.opp_network_params = self.opp_network.init(_key, jnp.zeros(
            (1, config.NUM_ENVS, env.observation_space(env_params).n)),
                                                        jnp.zeros((1, config.NUM_ENVS, 1)))

        self.per_buffer = fbx.make_prioritised_flat_buffer(max_length=config.BUFFER_SIZE,
                                                           min_length=config.BATCH_SIZE,
                                                           sample_batch_size=config.BATCH_SIZE,
                                                           add_sequences=True,
                                                           add_batch_size=None,
                                                           priority_exponent=config.REPLAY_PRIORITY_EXP,
                                                           device=config.DEVICE)

        self.per_buffer = self.per_buffer.replace(init=jax.jit(self.per_buffer.init),
                                                  add=jax.jit(self.per_buffer.add, donate_argnums=0),
                                                  sample=jax.jit(self.per_buffer.sample),
                                                  can_sample=jax.jit(self.per_buffer.can_sample),
                                                  )

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
        return (TrainStatePR2(critic_state=TrainStateExt.create(apply_fn=self.critic_network.apply,
                                                                params=self.critic_network_params,
                                                                target_params=self.critic_network_params,
                                                                tx=self.tx),
                              actor_state=TrainStateExt.create(apply_fn=self.actor_network.apply,
                                                               params=self.actor_network_params,
                                                               target_params=self.actor_network_params,
                                                               tx=self.tx),
                              opp_state=TrainState.create(apply_fn=self.opp_network.apply,
                                                          params=self.opp_network_params,
                                                          tx=self.tx)),
                self.per_buffer.init(
                    TransitionPR2(done=jnp.zeros((self.config.NUM_ENVS), dtype=bool),
                                  action=jnp.zeros((self.config.NUM_AGENTS, self.config.NUM_ENVS), dtype=jnp.int32),
                                  reward=jnp.zeros((self.config.NUM_ENVS)),
                                  obs=jnp.zeros((self.config.NUM_ENVS, self.env.observation_space(self.env_params).n),
                                                dtype=jnp.int8),
                                  # TODO is it always an int for the obs?
                                  )))

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):  # TODO don't think should ever reset the buffer right?
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def meta_policy(self, mem_state):
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: Any, mem_state: Any, ac_in: Any, key: Any):  # TODO better implement checks
        pi, action_logits = train_state.actor_state.apply_fn(train_state.actor_state.params,
                                                             ac_in)  # TODO should this be target params or actual params?
        # value = train_state.critic_state.apply_fn(train_state.critic_state.params, ac_in)  # TODO same as above
        value = jnp.zeros((1))  # TODO don't need to track it for PR2 update but double check
        key, _key = jrandom.split(key)
        action = pi.sample(seed=_key)
        log_prob = pi.log_prob(action)

        return mem_state, action, log_prob, value, key

    @partial(jax.jit, static_argnums=(0, 2))
    def update(self, runner_state, agent, traj_batch):
        train_state, mem_state, env_state, ac_in, key = runner_state

        mem_state = self.per_buffer.add(mem_state, TransitionPR2(done=traj_batch.done[:, agent, :],
                                                                 action=traj_batch.action,
                                                                 reward=traj_batch.reward[:, agent, :],
                                                                 obs=traj_batch.obs[:, agent, :],
                                                                 ))

        key, _key = jrandom.split(key)
        batch = self.per_buffer.sample(mem_state, _key)

        # CRITIC training
        def _critic_loss(critic_target_params, critic_params, opp_params, batch, key):
            obs = batch.experience.first.obs
            action = batch.experience.first.action
            reward = batch.experience.first.reward
            done = batch.experience.first.done
            nobs = batch.experience.second.obs
            naction = batch.experience.second.action
            ndone = batch.experience.second.done

            def remove_element(arr, index):  # TODO can improve?
                if arr.shape[-1] == 1:
                    raise ValueError("Cannot remove element from an array of size 1")
                elif arr.shape[-1] == 2:
                    return jnp.expand_dims(arr[:, :, 1 - index], -1)
                else:
                    return jnp.concatenate([arr[:, :, :index], arr[:, :, index + 1:]])

            naction = jnp.swapaxes(naction, 1, 2)
            naction_ego = jnp.expand_dims(naction[:, :, agent], -1)
            naction_opp = train_state.opp_state.apply_fn(opp_params, obs, naction_ego)

            nvalue = train_state.critic_state.apply_fn(critic_target_params, (nobs, jnp.expand_dims(ndone, axis=-1)),
                                                       naction_ego,
                                                       naction_opp)

            next_q_value = reward + (1 - done) * self.config.GAMMA * (nvalue)  # TODO done or ndone?

            # use Q-values only for the taken actions
            action = jnp.swapaxes(action, 1, 2)
            action_ego = jnp.expand_dims(action[:, :, agent], -1)
            action_opp = remove_element(action, agent)

            value = train_state.critic_state.apply_fn(critic_params, (obs, jnp.expand_dims(done, axis=-1)),
                                                      action_ego,
                                                      action_opp)

            critic_loss = 0.5 * jnp.mean(jnp.square(value - next_q_value))  # TODO check this is okay?

            return critic_loss

        key, _key = jrandom.split(key)
        critic_loss, grads = jax.value_and_grad(_critic_loss, argnums=1)(  # train_state.actor_state.target_params,
            train_state.critic_state.target_params,
            train_state.critic_state.params,
            train_state.opp_state.params,
            batch,
            _key
        )

        train_state = train_state._replace(
            critic_state=train_state.critic_state.apply_gradients(grads=grads))  # TODO check this works

        def _actor_loss(actor_params, critic_params, opp_params, batch, key):
            obs = batch.experience.first.obs
            action = batch.experience.first.action
            reward = batch.experience.first.reward
            done = batch.experience.first.done
            nobs = batch.experience.second.obs
            naction = batch.experience.second.action
            ndone = batch.experience.second.done

            # actor part
            pi, action_logits = train_state.actor_state.apply_fn(actor_params, (
            obs, jnp.expand_dims(done, axis=-1)))  # TODO remove done part at some point as not needed
            action_ego = jnp.expand_dims(pi.sample(seed=key), -1)
            action_opp = train_state.opp_state.apply_fn(opp_params, obs, action_ego)

            q_target = train_state.critic_state.apply_fn(critic_params, (nobs, jnp.expand_dims(ndone, axis=-1)),
                                                         action_ego,
                                                         action_opp)

            return jnp.mean(q_target)

        actor_loss, grads = jax.value_and_grad(_actor_loss, argnums=0)(train_state.actor_state.params,
                                                                       train_state.critic_state.params,
                                                                       train_state.opp_state.params,
                                                                       batch,
                                                                       key
                                                                       )
        train_state = train_state._replace(actor_state=train_state.actor_state.apply_gradients(grads=grads))

        def _opp_policy_loss(critic_params, opp_params, batch, key):
            obs = batch.experience.first.obs
            action = batch.experience.first.action
            reward = batch.experience.first.reward
            done = batch.experience.first.done
            nobs = batch.experience.second.obs
            naction = batch.experience.second.action
            ndone = batch.experience.second.done

            action = jnp.swapaxes(action, 1, 2)
            action_ego = jnp.expand_dims(action[:, :, agent], -1)
            action_opp = train_state.opp_state.apply_fn(opp_params, obs, action_ego)

            n_updated_actions = int(self.config.NUM_ENVS * 2 * self.config.KERNEL_UPDATE_RATIO)
            n_fixed_actions = (self.config.NUM_ENVS * 2) - n_updated_actions

            combo_actions = jnp.split(action_opp, [n_fixed_actions, n_updated_actions], axis=1)
            fixed_actions = combo_actions[0]
            updated_actions = combo_actions[2]
            fixed_actions = jax.lax.stop_gradient(fixed_actions)

            svgd_q_target = train_state.critic_state.apply_fn(critic_params, (obs, jnp.expand_dims(done, axis=-1)),
                                                         action_ego,
                                                         fixed_actions)

            print(svgd_q_target)

            value = train_state.critic_state.apply_fn(critic_params, (obs, jnp.expand_dims(done, axis=-1)),
                                                      action_ego,
                                                      action_opp)

            # value -= ind_value  # TODO implement maybe idk

            squash_correction = jnp.sum(jnp.log(1 - fixed_actions ** 2), axis=-1)

            log_p = value + squash_correction

            def vgrad(f, x):
                y, vjp_fn = jax.vjp(f, x)
                return vjp_fn(jnp.ones(y.shape))[0]

            grad_log_p = vgrad(log_p, fixed_actions)[0]
            grad_log_p = jnp.expand_dims(grad_log_p, axis=2)
            grad_log_p = jax.lax.stop_gradient(grad_log_p)
            print(grad_log_p)

            sys.exit()

            print(fixed_actions)
            print(updated_actions)


            sys.exit()

            return

        opp_loss, grads = jax.value_and_grad(_opp_policy_loss, argnums=1)(train_state.critic_state.params,
                                                                          train_state.opp_state.params,
                                                                          batch,
                                                                          key)

        return train_state, mem_state, env_state, ac_in, key

    @partial(jax.jit, static_argnums=(0,2))
    def meta_update(self, runner_state, agent, traj_batch):
        train_state, mem_state, env_state, ac_in, key = runner_state
        return train_state, mem_state, env_state, ac_in, key

    @partial(jax.jit, static_argnums=(0, 3))
    def update_encoding(self, train_state, mem_state, agent, obs_batch, action, reward, done):
        return mem_state
