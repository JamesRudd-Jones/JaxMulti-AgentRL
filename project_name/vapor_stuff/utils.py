import chex
from typing import Optional
import jax.numpy as jnp
import chex


@chex.dataclass(frozen=True)  # TODO is in here and algo so should add both to utils
class TransitionNoInfo:
    state: chex.Array
    action: chex.Array
    reward: chex.Array
    ensemble_reward: chex.Array
    done: chex.Array
    logits: chex.Array


Array = chex.Array


def l2_loss(predictions: Array,
            targets: Optional[Array] = None) -> Array:
  """Caculates the L2 loss of predictions wrt targets.
  FROM RLAX BUT COPIED HERE

  If targets are not provided this function acts as an L2-regularizer for preds.

  Note: the 0.5 term is standard in "Pattern Recognition and Machine Learning"
  by Bishop, but not "The Elements of Statistical Learning" by Tibshirani.

  Args:
    predictions: a vector of arbitrary shape.
    targets: a vector of shape compatible with predictions.

  Returns:
    a vector of same shape of `predictions`.
  """
  if targets is None:
    targets = jnp.zeros_like(predictions)
  chex.assert_type([predictions, targets], float)
  return 0.5 * (predictions - targets)**2