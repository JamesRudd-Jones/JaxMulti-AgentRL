import chex


@chex.dataclass(frozen=True)  # TODO is in here and algo so should add both to utils
class TransitionNoInfo:
    state: chex.Array
    action: chex.Array
    reward: chex.Array
    ensemble_reward: chex.Array
    done: chex.Array
    logits: chex.Array