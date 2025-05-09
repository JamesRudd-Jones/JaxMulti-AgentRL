from ml_collections import config_dict


def get_config():
    # PPO and MFOS
    config = config_dict.ConfigDict()
    config.SEED = 42

    config.CNN = False
    # config.CNN = True

    # config.TOTAL_TIMESTEPS = 10000000
    config.NUM_INNER_STEPS = 64  # ep rollout length
    config.NUM_META_STEPS = 1000  # number of ep rollouts to run
    config.NUM_UPDATES = 1  # 2000  # 500  # number of meta rollouts, should be 1 for no meta training
    config.NUM_ENVS = 4
    config.NUM_EVAL_STEPS = 1000

    # config.WANDB = "disabled"
    config.WANDB = "online"

    config.DISABLE_JIT = False
    # config.DISABLE_JIT = True

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    config.AGENT_TYPE = ["DDPG"]  # , "PPO_RNN"]  # ["MFOS", "ERSAC"]  # ["ROMMEO", "ROMMEO"]
    # config.AGENT_TYPE = ["IDQN", "VLITE_MA"]  # , "PPO_RNN"]  # ["MFOS", "ERSAC"]  # ["ROMMEO", "ROMMEO"]

    config.CTDE = False
    # config.CTDE = True

    config.NORMALISE_ENV = False
    # config.NORMALISE_ENV = True

    return config


"""
M - Number of Meta Episodes
E - Number of Episodes
L - Episode Length
G - Number of Agents
N - Number of Envs
O - Observation Dim
A - Action Dim
"""
