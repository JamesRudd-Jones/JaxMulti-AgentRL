from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.SEED = 42

    config.CNN = False
    # config.CNN = True

    config.NUM_INNER_STEPS = 128  # ep rollout length
    config.NUM_UPDATES = 10000  # number of rollouts
    config.NUM_ENVS = 64
    config.NUM_EVAL_STEPS = 1000

    config.NUM_META_STEPS = 0

    config.WANDB = "disabled"
    # config.WANDB = "online"

    config.DISABLE_JIT = False
    # config.DISABLE_JIT = True

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    config.AGENT_TYPE = ["PPO"]
    # config.AGENT_TYPE = ["IDQN", "VLITE_MA"]  # , "PPO_RNN"]  # ["MFOS", "ERSAC"]  # ["ROMMEO", "ROMMEO"]

    config.CTDE = False
    # config.CTDE = True

    return config


"""
Suffixes
B - Batch size, probably when using replay buffer
M - Number of Meta Episodes
E - Number of Episodes
L - Episode Length/NUM_INNER_STEPS
S - Seq length if using trajectory buffer/Planning Horizon
G - Number of Agents
N - Number of Envs
O - Observation Dim
A - Action Dim
Z - More dimensions when in a list
U - Ensemble num
P - Plus
M - Minus
"""
