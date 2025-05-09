from absl import app
from project_name.baselines_run import run_train, run_eval
import wandb
from project_name.config import get_config  # TODO dodge need to know how to fix this
from ml_collections import config_dict
import jax
import jax.profiler
from .envs.KS_JAX import KS_JAX
from .envs.SailingEnv import SailingEnv
from .envs.env_wrappers import GymnaxToJaxMARL, NormalisedEnv, FlattenObservationWrapper
# from .deep_sea_wrapper import BsuiteToMARL
from .pax.envs.in_the_matrix import InTheMatrix, EnvParams as MatrixEnvParams
from .pax.envs.iterated_matrix_game import IteratedMatrixGame, EnvParams
from .pax.envs.coin_game import CoinGame
from .pax.envs.coin_game import EnvParams as CoinGameParams
from .agents import SingleAgent, MultiAgent
from .utils import Transition, EvalTransition, Utils_IMG, Utils_IMPITM, Utils_CG, Utils_DEEPSEA, Utils_KS
import jax.numpy as jnp


def main(_):
    config = get_config()
    config.NUM_AGENTS = len(config.AGENT_TYPE)

    # TODO need to change update output to be for model info rather than env_state

    # TODO make rnn an option for all rather than diff agents, be cool if can still call agent PPO_RNN for example

    config.DEVICE = jax.extend.backend.get_backend().platform

    # if config.CNN:
    ####################################################################################################################
    payoff = jnp.array([[[3, 0], [5, 1]], [[3, 5], [0, 1]]])
    env = InTheMatrix(num_inner_steps=config.NUM_INNER_STEPS, num_outer_steps=config.NUM_META_STEPS,
                      fixed_coin_location=False)
    env_params = MatrixEnvParams(payoff_matrix=payoff, freeze_penalty=5)
    # env = FlattenObservationWrapper(env)
    utils = Utils_IMPITM(config)

    ####################################################################################################################
    # env = GymnaxToJaxMARL("DeepSea-bsuite", {"size": config.NUM_INNER_STEPS,
    #                                          "sample_action_map": False})
    # # TODO need to adjust the gymnax file with the custom one
    # env_params = env.default_params
    # utils = Utils_DEEPSEA(config)

    ####################################################################################################################
    payoff = [[3, 3], [1, 4], [4, 1], [2, 2]]  # [[-1, -1], [-3, 0], [0, -3], [-2, -2]]  # payoff matrix for the IPD
    env = IteratedMatrixGame(num_inner_steps=config.NUM_INNER_STEPS, num_outer_steps=config.NUM_META_STEPS)
    env_params = EnvParams(payoff_matrix=payoff)
    utils = Utils_IMG(config)
    # TODO the above game has issues with when it ends? causing loss spikes it seems

    ####################################################################################################################
    env = CoinGame(num_inner_steps=config.NUM_INNER_STEPS, num_outer_steps=config.NUM_META_STEPS,
                   cnn=False, egocentric=False)
    env_params = CoinGameParams(payoff_matrix=[[1, 1, -2], [1, 1, -2]])
    utils = Utils_CG(config)

    ####################################################################################################################
    env = GymnaxToJaxMARL("KS_Equation", env=KS_JAX()) # TODO how to adjust default params for this step
    env_params = env.default_params
    utils = Utils_KS(config)

    ####################################################################################################################
    env = GymnaxToJaxMARL("SailingEnv", env=SailingEnv())  # TODO how to adjust default params for this step
    env_params = env.default_params
    utils = Utils_KS(config)

    if config.NORMALISE_ENV:
        env = NormalisedEnv(env, env_params)

    key = jax.random.PRNGKey(config.SEED)

    if config.NUM_AGENTS == 1:
        actor = SingleAgent(env=env, env_params=env_params, config=config, utils=utils, key=key)
        config.update(actor.agent.agent_config)
    else:
        actor = MultiAgent(env=env, env_params=env_params, config=config, utils=utils, key=key)
        for i, agent in enumerate(actor.agent_list):
            config[f'agent_config_{i + 1}'] = actor.agent_list[i].agent_config
            # TODO almost right but not quite there

    wandb.init(project="ProbInfMarl",
               entity=config.WANDB_ENTITY,
               config=config,
               group="sailing_env_tests",
               mode=config.WANDB
               )

    with jax.disable_jit(disable=config.DISABLE_JIT):
        train = jax.jit(run_train(config, actor, env, env_params, utils))
        # train = jax.jit(run_train(config))
        out = jax.block_until_ready(train())  # .block_until_ready()

        run_eval(config, actor, env, env_params, utils, out["runner_state"][0][0], out["runner_state"][0][1])

    print("FINITO")

if __name__ == '__main__':
    app.run(main)
