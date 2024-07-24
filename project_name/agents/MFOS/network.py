import sys

import flax.linen as nn
import functools
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
import distrax
import seaborn as sns
import matplotlib.pyplot as plt


class ScannedRNN(nn.Module):
    @functools.partial(nn.scan,
                       variable_broadcast="params",
                       in_axes=0,
                       out_axes=0,
                       split_rngs={"params": False},
                       )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x

        rnn_state = jnp.where(resets[:, jnp.newaxis],
                              self.initialize_carry(*rnn_state.shape),
                              rnn_state,
                              )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jrandom.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticMFOSRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        hidden_t, hidden_a, hidden_c = jnp.split(hidden, 3, axis=-1)  # TODO check this ting I guess

        # embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        # embedding = nn.relu(embedding)

        meta_emb = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        actor_emb = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        critic_emb = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        # TODO theres no non lineariites??

        hidden_a, embedding_a = ScannedRNN()(hidden_a, (actor_emb, dones))
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding_a)
        pi = distrax.Categorical(logits=actor_mean)

        hidden_c, embedding_c = ScannedRNN()(hidden_c, (critic_emb, dones))
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(embedding_c)

        hidden_t, embedding_t = ScannedRNN()(hidden_t, (meta_emb, dones))
        current_th = nn.sigmoid(nn.Dense(128)(embedding_t))

        hidden = jnp.concatenate([hidden_t, hidden_a, hidden_c], axis=-1)

        return hidden, pi, jnp.squeeze(critic, axis=-1), actor_mean, current_th
