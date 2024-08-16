import sys
import jax
import jax.numpy as jnp
from typing import Any
import jax.random as jrandom
from functools import partial
from project_name.agents.ROMMEO.network import ActorROMMEO, JointCriticROMMEO, OppNetworkROMMEO, Opp
import optax
from flax.training.train_state import TrainState
from project_name.utils import MemoryStatem, TrainStateExt
import flashbax as fbx
from typing import NamedTuple
import flax
from project_name.agents import AgentBase
from project_name.utils import remove_element
from project_name.agents.PR2.kernel import adaptive_isotropic_gaussian_kernel


class TrainStateExtROMMEO(TrainState):
    target_params: flax.core.FrozenDict
    prior_params: flax.core.FrozenDict


class TrainStateROMMEO(NamedTuple):  # TODO is this correct tag?
    critic_state: TrainStateExt
    actor_state: TrainStateExt
    opp_state: TrainStateExtROMMEO


class TransitionROMMEO(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray


class ROMMEOAgent(AgentBase):
    def __init__(self,
                 env,
                 env_params,
                 key,
                 config):
        self.config = config
        self.env = env
        self.env_params = env_params
        self.critic_network = JointCriticROMMEO(config=config)
        self.actor_network = ActorROMMEO(action_dim=env.action_space(env_params).n,
                                      config=config)
        self.opp_network = OppNetworkROMMEO(action_dim=config.NUM_AGENTS - 1, config=config)  # TODO is num agents dim okay?
        self.opp_prior = OppPriorROMMEO

        key, _key = jrandom.split(key)

        init_x = (jnp.zeros((1, config.NUM_ENVS, env.observation_space(env_params).n)))

        self.critic_network_params = self.critic_network.init(_key, init_x,
                                                                          jnp.zeros((1, config.NUM_ENVS, 1)),
                                                                          jnp.zeros((1, config.NUM_ENVS,
                                                                                     config.NUM_AGENTS - 1)))
        self.actor_network_params = self.actor_network.init(_key, init_x)
        self.opp_network_params = self.opp_network.init(_key, jnp.zeros(
            (1, config.NUM_ENVS, 1, env.observation_space(env_params).n)),
                                                        jnp.zeros((1, config.NUM_ENVS, 1, 1)),
                                                        jnp.zeros((1, config.NUM_ENVS, 1,
                                                                   config.NUM_AGENTS - 1))
                                                        )

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
        return (TrainStateROMMEO(critic_state=TrainStateExt.create(apply_fn=self.critic_network.apply,
                                                                      params=self.critic_network_params,
                                                                      target_params=self.critic_network_params,
                                                                      tx=self.tx),
                              actor_state=TrainStateExt.create(apply_fn=self.actor_network.apply,
                                                               params=self.actor_network_params,
                                                               target_params=self.actor_network_params,
                                                               tx=self.tx),
                              opp_state=TrainStateExtROMMEO.create(apply_fn=self.opp_network.apply,
                                                          params=self.opp_network_params,
                                                          target_params=self.opp_network_params,
                                                                   prior_params=self.opp_network_params,
                                                          tx=self.tx)),
                self.per_buffer.init(
                    TransitionROMMEO(done=jnp.zeros((self.config.NUM_ENVS), dtype=bool),
                                  action=jnp.zeros((self.config.NUM_AGENTS, self.config.NUM_ENVS), dtype=jnp.int32),
                                  reward=jnp.zeros((self.config.NUM_ENVS)),
                                  obs=jnp.zeros((self.config.NUM_ENVS, self.env.observation_space(self.env_params).n),
                                                dtype=jnp.int8),
                                  # TODO is it always an int for the obs?
                                  )))

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self,
                     mem_state):  # TODO don't think should ever reset the buffer right? but should reset the rest?
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: Any, mem_state: Any, ac_in: Any, key: Any):  # TODO better implement checks
        # dist, _, _ = train_state.actor_state.apply_fn(train_state.actor_state.params,
        #                                                      ac_in)  # TODO should this be target params or actual params?
        value = jnp.zeros((1))  # TODO don't need to track it for PR2 update but double check
        key, _key = jrandom.split(key)
        # x_t = dist.sample(seed=_key)
        # if not self.config.REPARAMETRISE:
        #     x_t = jax.lax.stop_gradient(x_t)
        # log_prob = dist.log_prob(x_t)
        #
        # if self.config.SQUASH:
        #     actions = nn.tanh(x_t)
        # else:
        #     actions = x_t
        actions, log_prob, _ = self._opp_or_ego_model_act(train_state.actor_state, train_state.actor_state.params,
                                                          ac_in, key)

        return mem_state, actions, log_prob, value, key

    @partial(jax.jit, static_argnums=(0,))
    def _opp_or_ego_model_act(self, specific_train_state: Any, params: dict, ins , key):
        dist, mu, log_sig = specific_train_state.apply_fn(params, *ins)
        key, _key = jrandom.split(key)
        x_t = dist.sample(seed=_key)
        if not self.config.REPARAMETRISE:
            x_t = jax.lax.stop_gradient(x_t)
        log_prob = dist.log_prob(x_t)

        reg_loss = self.config.REGULARISER * 0.5 * jnp.mean(log_sig ** 2)  # TODO should this be within the loss section?
        reg_loss = reg_loss + self.config.REGULARISER * 0.5 * jnp.mean(log_mu ** 2)

        if self.config.SQUASH:
            actions = nn.tanh(x_t)
        else:
            actions = x_t

        return actions, log_prob, reg_loss

    @partial(jax.jit, static_argnums=(0,))
    def _get_opponent_prior(self, specific_train_state: Any, params: dict, obs: chex.Array, actions: chex.Array):
        dist, mu, log_sig = specific_train_state.apply_fn(params, obs)

        raw_actions = jnp.tanh(actions)

        log_prob = dist.log_prob(raw_actions)
        log_prob = log_prob - squash_correction(raw_actions)

        reg_loss = self.config.REGULARISER * 0.5 * jnp.mean(
            log_sig ** 2)  # TODO should this be within the loss section?
        reg_loss = reg_loss + self.config.REGULARISER * 0.5 * jnp.mean(log_mu ** 2)

        return  log_prob, reg_loss

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state, agent, traj_batch):
        train_state, mem_state, env_state, ac_in, key = runner_state

        mem_state = self.per_buffer.add(mem_state, TransitionROMMEO(done=traj_batch.done[:, agent, :],
                                                                 action=traj_batch.action,
                                                                 reward=traj_batch.reward[:, agent, :],
                                                                 obs=traj_batch.obs[:, agent, :],
                                                                 ))

        key, _key = jrandom.split(key)
        batch = self.per_buffer.sample(mem_state, _key)

        def _opp_prior_loss(prior_params, batch):
            opp_obs = batch.experience.first.obs  # TODO its recent_opp_obs??
            action = batch.experience.first.action

            action = jnp.swapaxes(action, 1, 2)
            action_opp = remove_element(action, agent)

            log_prob, reg_loss = self._get_opponent_prior(train_state.opp_state, prior_params, opp_obs, action_opp)

            loss = -jnp.mean(log_prob) + reg_loss

            return loss

        prior_loss, grads = jax.value_and_grad(_opp_prior_loss, argnums=0)(train_state.opp_state.prior_params,
                                                                          batch)
        train_state = train_state._replace(opp_state=train_state.opp_state.apply_gradients(grads=grads))  # TODO apply only to prior params not the whole dealio am unsure how

        def _opp_policy_loss(opp_params, prior_params, critic_params, actor_params, batch, key):
            obs = batch.experience.first.obs

            action_opp, action_opp_logprob, reg_loss = self._opp_or_ego_model_act(train_state.opp_state, opp_params, (obs,), key)

            prior_logprob, _ = self._get_opponent_prior(train_state.opp_state, prior_params, obs, action_opp)

            action_ego, action_ego_logprob, _ = self._opp_or_ego_model_act(train_state.actor_state, actor_params, (obs, action_opp), key)

            joint_q = train_state.joint_critic_state.apply_fn(critic_params, obs,
                                                              action_ego,
                                                              action_opp)

            opp_p_loss = jnp.mean(action_opp_logprob) - jnp.mean(prior_logprob) - jnp.mean(joint_q) + self.config.ANNEALING * agent_ego_logprob
            opp_p_loss += reg_loss

            return opp_p_loss

        key, _key = jrandom.split(key)  # TODO do I need this?
        opp_loss, grads = jax.value_and_grad(_opp_policy_loss, argnums=0)(train_state.opp_state.params,
                                                                          train_state.opp_state.prior_params,
                                                                        train_state.critic_state.params,
                                                                          train_state.actor_state.params,
                                                                          batch,
                                                                          _key)
        train_state = train_state._replace(opp_state=train_state.opp_state.apply_gradients(grads=grads))

        # CRITIC training
        def _critic_loss(opp_params, prior_params, critic_target_params, critic_params, actor_params, batch, key):
            obs = batch.experience.first.obs
            action = batch.experience.first.action
            reward = batch.experience.first.reward
            done = batch.experience.first.done
            nobs = batch.experience.second.obs

            naction_opp, naction_opp_logprob, _ = self._opp_or_ego_model_act(train_state.opp_state, opp_params, (nobs,), key)

            prior_logprob, _ = self._get_opponent_prior(train_state.opp_state, prior_params, nobs, naction_opp)

            naction_ego, naction_ego_logprob, _ = self._opp_or_ego_model_act(train_state.actor_state,
                                                                             actor_params,
                                                                             (nobs, naction_opp),
                                                                             key)


            # shape batch_size, num_envs, etc
            joint_target_q = train_state.critic_state.apply_fn(critic_target_params,
                                                                         nobs,
                                                                         naction_ego,
                                                                         naction_opp)

            joint_target_q = joint_target_q - self.config.ANNEALING * naction_ego_logprob - naction_opp_logprob + prior_logprob

            # use Q-values only for the taken actions
            action = jnp.swapaxes(action, 1, 2)
            action_ego = jnp.expand_dims(action[:, :, agent], -1)
            action_opp = remove_element(action, agent)
            joint_q = train_state.joint_critic_state.apply_fn(critic_params, obs,
                                                              action_ego,
                                                              action_opp)

            target_q = jax.lax.stop_gradient(reward + (1 - done) * self.config.GAMMA * (joint_target_q))

            critic_loss = 0.5 * jnp.mean(jnp.square(target_q - joint_q))

            return critic_loss

        key, _key = jrandom.split(key)
        critic_loss, grads = jax.value_and_grad(_critic_loss, argnums=2)(
            train_state.opp_state.params,
            train_state.opp_state.prior_params,
            train_state.critic_state.target_params,
            train_state.critic_state.params,
            train_state.actor_state.params,
            batch,
            _key
        )

        train_state = train_state._replace(critic_state=train_state.critic_state.apply_gradients(grads=grads))
        # TODO check this works

        def _actor_loss(actor_params, critic_params, opp_params, batch, key):
            obs = batch.experience.first.obs

            action_opp, action_opp_logprob, _ = self._opp_or_ego_model_act(train_state.opp_state, opp_params, (obs,), key)

            action_ego, action_ego_logprob, reg_loss = self._opp_or_ego_model_act(train_state.actor_state,
                                                                                  actor_params,
                                                                                  (obs, action_opp),
                                                                                  key)

            joint_q = train_state.joint_critic_state.apply_fn(critic_params, obs,
                                                              action_ego,
                                                              action_opp)

            pg_loss = self.config.ANNEALING * jnp.mean(action_ego_logprob) - jnp,mean(joint_q)
            pg_loss = pg_loss + reg_loss

            return pg_loss

        key, _key = jrandom.split(key)
        actor_loss, grads = jax.value_and_grad(_actor_loss, argnums=0)(train_state.actor_state.params,
                                                                       train_state.joint_critic_state.params,
                                                                       train_state.opp_state.params,
                                                                       batch,
                                                                       _key
                                                                       )
        train_state = train_state._replace(actor_state=train_state.actor_state.apply_gradients(grads=grads))

        return train_state, mem_state, env_state, ac_in, key

