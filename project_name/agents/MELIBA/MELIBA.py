import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
from project_name.agents.MELIBA.hierarchical_sequential_VAE import HierarchicalSequentialVAE, EncoderRNN, \
    DecoderRNN, Encoder  # TODO sort this out
from flax.training.train_state import TrainState
import optax
from project_name.agents.MELIBA.PPO import PPOAgent  # TODO sort this out
from functools import partial
import sys
import flax.linen as nn
from typing import Tuple, NamedTuple, Mapping, Any
import flashbax as fbx


class MemoryStateMELIBA(NamedTuple):
    """State consists of network extras (to be batched)"""
    ppo_hstate: jnp.ndarray
    encoder_hstate: jnp.ndarray
    decoder_hstate: jnp.ndarray
    vae_buffer_state: jnp.ndarray
    extras: Mapping[str, jnp.ndarray]


class TrainStateMELIBA(NamedTuple):
    vae_state: TrainState
    ppo_state: TrainState


class TransitionMELIBA(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray


class TransitionNoMemState(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    latent: jnp.ndarray


class MELIBAAgent:
    def __init__(self, env,
                 env_params,
                 key: chex.PRNGKey,
                 config):
        self.config = config
        self.env = env
        self.env_params = env_params

        if self.config.STATELESS:
            init_x = (jnp.zeros((1, config["NUM_ENVS"], 1)),
                      jnp.zeros((1, config["NUM_ENVS"], 1)),
                      # env.action_space(env_params).n)),     # actions
                      jnp.zeros((1, config["NUM_ENVS"], 1)),  # rewards
                      jnp.zeros((1, config["NUM_ENVS"])))  # dones
        else:
            init_x = (
                jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env_params).n)),  # states
                jnp.zeros((1, config["NUM_ENVS"], config.NUM_AGENTS)),
                # env.action_space(env_params).n)),     # actions
                jnp.zeros((1, config["NUM_ENVS"], 1)),  # rewards
                jnp.zeros((1, config["NUM_ENVS"])))  # dones

        key, _key = jrandom.split(key)
        self.encoder = Encoder(config)
        self.hsvae = HierarchicalSequentialVAE(_key, config)

        self.init_encoder_hstate = EncoderRNN.initialize_carry(config["NUM_ENVS"],
                                                               config["GRU_HIDDEN_DIM"])  # TODO individual configs pls
        self.init_decoder_hstate = DecoderRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

        self.hsvae_params = self.hsvae.init(_key, self.init_encoder_hstate, self.init_decoder_hstate, init_x)

        self.ppo = PPOAgent(env, env_params, key, config)

        self.vae_buffer = fbx.make_prioritised_flat_buffer(max_length=config.BUFFER_SIZE,
                                                           min_length=config.BATCH_SIZE,
                                                           sample_batch_size=config.BATCH_SIZE,
                                                           add_sequences=True,
                                                           add_batch_size=None,
                                                           priority_exponent=config.REPLAY_PRIORITY_EXP,
                                                           device=config.DEVICE)

        self.vae_buffer = self.vae_buffer.replace(init=jax.jit(self.vae_buffer.init),
                                                  add=jax.jit(self.vae_buffer.add, donate_argnums=0),
                                                  sample=jax.jit(self.vae_buffer.sample),
                                                  can_sample=jax.jit(self.vae_buffer.can_sample),
                                                  )

    def create_train_state(self):  # TODO seperate train state for vae and ppo?
        ppo_state = self.ppo.create_train_state()
        return (TrainStateMELIBA(vae_state=TrainState.create(apply_fn=self.hsvae.apply,
                                                             params=self.hsvae_params,
                                                             tx=optax.adam(self.config["LR"])),
                                 ppo_state=ppo_state[0]),
                MemoryStateMELIBA(ppo_hstate=ppo_state[1],
                                  encoder_hstate=self.init_encoder_hstate,
                                  decoder_hstate=self.init_decoder_hstate,
                                  vae_buffer_state=self.vae_buffer.init(
                                      TransitionMELIBA(done=jnp.zeros((self.config.NUM_ENVS), dtype=bool),
                                                       action=jnp.zeros((self.config.NUM_AGENTS, self.config.NUM_ENVS),
                                                                        dtype=jnp.int32),
                                                       reward=jnp.zeros((self.config.NUM_ENVS)),
                                                       obs=jnp.zeros((self.config.NUM_ENVS,
                                                                      self.env.observation_space(self.env_params).n),
                                                                     dtype=jnp.int8),
                                                       # TODO is it always an int for the obs?
                                                       )),
                                  extras={
                                      "values": jnp.zeros((self.config.NUM_ENVS, 1)),
                                      "log_probs": jnp.zeros((self.config.NUM_ENVS, 1)),
                                      "latent_sample": jnp.zeros((1, self.config.NUM_ENVS, self.config.LATENT_DIM)),
                                      "latent_mean": jnp.zeros((1, self.config.NUM_ENVS, self.config.LATENT_DIM)),
                                      "latent_logvar": jnp.zeros((1, self.config.NUM_ENVS, self.config.LATENT_DIM)),
                                  }, ),
                )

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):
        # TODO get batch of data from vae storage

        # TODO run encoder over it but needs hidden state to be reset here maybe?

        # TODO add the output latent to the mem_state

        mem_state = mem_state._replace(extras={
            "values": jnp.zeros((self.config.NUM_ENVS, 1)),
            "log_probs": jnp.zeros((self.config.NUM_ENVS, 1)),
            "latent_sample": jnp.zeros((1, self.config.NUM_ENVS, self.config.LATENT_DIM)),
            "latent_mean": jnp.zeros((1, self.config.NUM_ENVS, self.config.LATENT_DIM)),
            "latent_logvar": jnp.zeros((1, self.config.NUM_ENVS, self.config.LATENT_DIM)),
            # TODO should these reset to 0 or not idk?
        },
            ppo_hstate=jnp.zeros((self.config.NUM_ENVS, 1)),
            encoder_hstate=jnp.zeros((self.config.NUM_ENVS, self.config.GRU_HIDDEN_DIM)),
            decoder_hstate=jnp.zeros((self.config.NUM_ENVS, self.config.GRU_HIDDEN_DIM)),
        )
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def meta_policy(self, mem_state):
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: Any, mem_state: Any, ac_in: chex.Array, key: chex.PRNGKey):
        # obs, dones = ac_in
        # new_obs = jnp.concatenate((obs, belief), axis=-1)  # TODO should we embed the input or not? this would make it trickier if so
        # ac_in = new_obs, dones

        latent_sample = mem_state.extras["latent_sample"]
        latent_mean = mem_state.extras["latent_mean"]
        latent_logvar = mem_state.extras["latent_logvar"]

        # def _get_latent_for_policy(latent_sample=None, latent_mean=None, latent_logvar=None):
        #
        #     if (latent_sample is None) and (latent_mean is None) and (latent_logvar is None):
        #         return None
        #
        #     if args.add_nonlinearity_to_latent:
        #         latent_sample = F.relu(latent_sample)
        #         latent_mean = F.relu(latent_mean)
        #         latent_logvar = F.relu(latent_logvar)
        #
        #     if args.sample_embeddings:
        #         latent = latent_sample
        #     else:
        #         latent = torch.cat((latent_mean, latent_logvar), dim=-1)
        #
        #     if latent.shape[0] == 1:
        #         latent = latent.squeeze(0)
        #
        #     return latent

        # TODO get latent for policy
        # latent = _get_latent_for_policy(latent_sample=latent_sample, latent_mean=latent_mean,
        #                                latent_logvar=latent_logvar)
        latent = latent_sample

        # TODO add latent to something to feed into the policy

        # needs to act via ppo agent but also takes in posterior from encoder
        mem_state, action, log_prob, value, key = self.ppo.act(train_state.ppo_state, mem_state, ac_in, latent, key)

        # # TODO update encoding maybe
        # latent_sample, latent_mean, latent_logvar, encoder_hstate = self.hsvae.encoder()

        return mem_state, action, log_prob, value, key

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state: chex.Array, agent, traj_batch: chex.Array):
        train_state, mem_state, env_state, ac_in, key = runner_state

        new_buffer_state = self.vae_buffer.add(mem_state.vae_buffer_state,
                                               TransitionMELIBA(done=traj_batch.done[:, agent, :],
                                                                action=traj_batch.action,
                                                                reward=traj_batch.reward[:, agent, :],
                                                                obs=traj_batch.obs[:, agent, :],
                                                                ))
        mem_state = mem_state._replace(vae_buffer_state=new_buffer_state)
        key, _key = jrandom.split(key)
        batch = self.vae_buffer.sample(mem_state.vae_buffer_state, _key)

        # update ppo loss
        ppo_runner_state = train_state.ppo_state, mem_state, env_state, ac_in, key
        # TODO check it works with latent tracking below as unsure if gets at the right step basically
        ppo_traj_batch = TransitionNoMemState(traj_batch.global_done,
                                              traj_batch.done,
                                              traj_batch.action,
                                              traj_batch.value,
                                              traj_batch.reward,
                                              traj_batch.log_prob,
                                              traj_batch.obs,
                                              traj_batch.info,
                                              traj_batch.mem_state.extras["latent_sample"].squeeze(axis=1))
        ppo_train_state, mem_state, env_state, ac_in, key = self.ppo.update(ppo_runner_state, agent, ppo_traj_batch)
        train_state = train_state._replace(ppo_state=ppo_runner_state)  # TODO check this works

        # then also update vae but don't pass through each other
        vae_train_state, total_loss = self._update_vae(train_state, mem_state, batch)
        train_state = train_state._replace(vae_state=vae_train_state)  # TODO check this works

        # TODO update encoding my dude encode_running_trajectory

        return train_state, mem_state, env_state, ac_in, key

    @partial(jax.jit, static_argnums=(0,))
    def _update_vae(self, train_state, mem_state, batch):

        def calc_kl_loss(mean, log_var, elbo_indices):  # TODO unsure this is correct for now
            gaussian_dim = mean.shape[-1]
            means_inc_prior = jnp.concatenate((jnp.zeros((mean[0][jnp.newaxis, :].shape)), mean), axis=0)
            log_vars_inc_prior = jnp.concatenate((jnp.ones((log_var[0][jnp.newaxis, :].shape)), log_var), axis=0)
            # TODO is the above ones or zeros? varibad is ident matrix but in paper it says 1s?

            posterior_means = means_inc_prior[1:]
            prior_means = means_inc_prior[:-1]

            posterior_log_vars = log_vars_inc_prior[1:]
            prior_log_vars = log_vars_inc_prior[:-1]

            kl_divergences = 0.5 * (
                    jnp.sum(prior_log_vars, axis=-1) - jnp.sum(posterior_log_vars, axis=-1) - gaussian_dim +
                    jnp.sum(1 / jnp.exp(prior_log_vars) * jnp.exp(posterior_log_vars), axis=-1) +
                    jnp.sum(((prior_means - posterior_means) / jnp.exp(prior_log_vars) *
                             (prior_means - posterior_means)), axis=-1))

            if elbo_indices is not None:
                batchsize = kl_divergences.shape[-1]
                task_indices = jnp.arange(batchsize)  # .repeat(self.args.vae_subsample_elbos)
                kl_divergences = kl_divergences[
                    elbo_indices, task_indices]  # .reshape((self.args.vae_subsample_elbos, batchsize))

            return kl_divergences

        def compute_loss(params, hidden_encoder, hidden_decoder, batch):  # TODO do I need this hidden state idk?
            # TODO this kinda dodgy loss I think, as t may shift but H stays the same? maybe that is okay?

            obs = batch.experience.first.obs
            action = batch.experience.first.action
            reward = batch.experience.first.reward
            done = batch.experience.first.done
            nobs = batch.experience.second.obs
            naction = batch.experience.second.action
            ndone = batch.experience.second.done

            # pass through encoder to get the mean and stddev, include the prior so shape is H+1
            # TODO how to get prior from vae and use with the kl loss, they should both have the save dimensions
            future_action_logits, batch_mu, batch_log_sigma, hidden_encoder, hidden_decoder = train_state.vae_state.apply_fn(
                params,
                hidden_encoder,
                hidden_decoder,
                past_traj,
                full_traj=True)

            # train by elbo loss which will be dependent on t, aka an elbo for each t, includes prior in kl calc
            total_action_loss = []
            n_elbos = batch_mu.shape[0]
            for elbo in range(n_elbos):  # TODO make this jax later on
                # do a loss between predicted actions and future actions
                # this cuts window to elbo:H
                pred_actions_logits_window = future_action_logits[elbo:]
                true_actions_window = jnp.squeeze(past_traj[1][elbo:], axis=-1)
                # TODO maybe an index up or something? am unsure here also remove the need for the 1 index, also num_classes is action space

                # cross entropy loss as using discrete actions atm
                action_loss = jnp.prod(
                    optax.softmax_cross_entropy_with_integer_labels(pred_actions_logits_window, true_actions_window))
                # TODO is the above prod the right thing to do?

                total_action_loss.append(action_loss)

            # do the kl loss between distributions
            kl_loss = jnp.sum(calc_kl_loss(batch_mu, batch_log_sigma, None))  # TODO could be a mean or sum

            total_action_loss = sum(total_action_loss)

            # take the mean
            loss = jnp.mean(self.config["KL_WEIGHT"] * kl_loss + total_action_loss)  # TODO unsure if need this mean?

            future_actions = jnp.argmax(jax.nn.softmax(future_action_logits), axis=-1)

            return loss, (kl_loss, total_action_loss, future_actions)

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True, allow_int=True)  # TODO turn off int
        total_loss, grads = grad_fn(train_state.vae_state.params,
                                    mem_state.encoder_hstate,
                                    mem_state.decoder_hstate,
                                    batch)
        train_state = train_state.apply_gradients(grads=grads)

        return train_state, total_loss

    @partial(jax.jit, static_argnums=(0, 2))
    def meta_update(self, runner_state, agent, traj_batch):
        train_state, mem_state, env_state, ac_in, key = runner_state
        return train_state, mem_state, env_state, ac_in, key

    @partial(jax.jit, static_argnums=(0, 3))
    def update_encoding(self, train_state, mem_state, agent, obs_batch, action, reward, done):
        # TODO make sure this is fine lol as may mess up if get full traj length inputs
        obs_batch = jnp.expand_dims(obs_batch[agent], axis=0)
        action = jnp.expand_dims(jnp.swapaxes(action, 0, 1), axis=0)
        reward = jnp.expand_dims(reward[agent], axis=(0, 2))
        done = jnp.expand_dims(done[agent], axis=(0))

        encoder_hstate, latent_mean, latent_logvar = self.encoder.apply(
            {"params": train_state.vae_state.params["params"]["encoder"]},
            mem_state.encoder_hstate,
            (obs_batch, action, reward, done))

        latent_sample = self.hsvae._gaussian_sample(latent_mean, latent_logvar)

        mem_state = mem_state._replace(extras={
            "values": mem_state.extras["values"],
            "log_probs": mem_state.extras["log_probs"],
            "latent_sample": latent_sample,
            "latent_mean": latent_mean,
            "latent_logvar": latent_logvar,  # TODO should these reset to 0 or not idk?
        },
            encoder_hstate=encoder_hstate
        )

        return mem_state
