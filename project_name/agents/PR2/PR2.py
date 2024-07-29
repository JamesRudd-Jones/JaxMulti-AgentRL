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
from typing import NamedTuple


# class TrainStatePR2(TrainState):
#     train_state: TrainState
#     buffer_state: Any

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
                    TransitionPR2(done=jnp.zeros((1), dtype=bool),
                                  action=jnp.zeros((self.config.NUM_AGENTS)),
                                  reward=jnp.zeros((1)),
                                  obs=jnp.zeros((env.observation_space(env_params).n)),
                                  )))

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):  # TODO don't think should ever reset the buffer right?
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
    def update(self, runner_state, agent, traj_batch):
        train_state, mem_state, env_state, last_obs, last_done, key = runner_state

        # TODO convert traj_batch to buffer and then do sampling here etc, keeps on policy loop but works for our requirements
        print(traj_batch)
        sys.exit()

        mem_state = self.per_buffer.add(mem_state, TransitionNoInfo(done=trajectory_batch.done[:, agent],
                                                                    action=trajectory_batch.action,
                                                                    reward=trajectory_batch.reward[:, agent],
                                                                    obs=trajectory_batch.obs[:, agent],
                                                                    ))

        # traj_batch = jax.tree_map(lambda x: x[:, agent], traj_batch)

        key, _key = jrandom.split(key)
        batch = self.per_buffer.sample(mem_state, _key)

        # CRITIC training
        def _loss_fn(params, target_params, batch, key):
            obs = batch.experience.first.state
            action = batch.experience.first.action
            reward = batch.experience.first.reward
            done = batch.experience.first.done
            nobs = batch.experience.second.state
            ndone = batch.experience.second.done

            # critic part
            nobs_in = (nobs[jnp.newaxis, agent, :],  # TODO generalise this to cnn stuff
                       ndone[jnp.newaxis, agent],
                       )

            _, ego_action, _, _, key = self.act(train_state, mem_state, nobs_in, key)

            _, nvalue, _ = train_state.apply_fn(train_state.target_params, nobs_in)

            # _, _, next_state_log_pi, next_state_action_probs, key = self.act(train_state, nobs, key)
            #
            # qf_next_target = self.critic_network.apply(critic_target_params, nobs)

            qf_next_target = jnp.mean(nvalue, axis=1, keepdims=True)  # TODO over dimension m but idk what this is

            next_q_value = reward + (1 - done) * self.config.GAMMA * (qf_next_target)

            # use Q-values only for the taken actions
            obs_in = (nobs[jnp.newaxis, agent, :],  # TODO generalise this to cnn stuff
                       ndone[jnp.newaxis, agent],
                       )
            _, value, _ = train_state.apply_fn(train_state.params, nobs_in)

            # mse loss below
            critic_loss = 0.5 * jnp.mean(jnp.square(value - next_q_value ))  # TODO check this is okay?

            # actor part
            obs_in = (obs[jnp.newaxis, agent, :],  # TODO generalise this to cnn stuff
                       done[jnp.newaxis, agent],
                       )

            _, ego_action, _, _, key = self.act(train_state, mem_state, obs_in, key)

            _, log_pi, action_probs, _ = self.act(actor_params, obs, key)

            # return jnp.mean(action_probs * ((self.config.ALPHA * log_pi) - min_qf_values))

            total_loss = critic_loss + actor_loss  # TODO maybe better to separate

            return totaL_loss, (critic_loss, actor_loss)

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
    def meta_update(self, runner_state, agent, traj_batch):
        train_state, mem_state, env_state, last_obs, last_done, key = runner_state
        return train_state, mem_state, env_state, last_obs, last_done, key
