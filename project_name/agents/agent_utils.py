import jax
import jax.numpy as jnp
import jax.random as jrandom

def make_linear_schedule(num_minibatches, update_epochs, num_updates, lr):
    def linear_schedule(count):
        frac = (1.0 - (count // (num_minibatches * update_epochs)) / num_updates)
        return lr * frac

    return linear_schedule