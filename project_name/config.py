from ml_collections import config_dict


def get_config():
    # PPO and MFOS
    config = config_dict.ConfigDict()
    config.SEED = 42

    config.CNN = False
    # config.CNN = True

    # config.TOTAL_TIMESTEPS = 10000000
    config.NUM_UPDATES = 25000  # 40000  # 10000
    config.NUM_INNER_STEPS = 100  # 30  # 128
    config.NUM_META_STEPS = 50  # 100  # 500
    config.NUM_ENVS = 1  # 8  # TODO should add an assert for this - MUST BE SAME SIZE OR BIGGER THAN NUM_MINIBATCHES
    config.NUM_DEVICES = 1

    # config.DEEP_SEA_MAP = 1  # 20

    # config.WANDB = "disabled"  # "online" if want it to work
    config.WANDB = "online"

    config.DISABLE_JIT = False
    # config.DISABLE_JIT = True

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    config.AGENT_TYPE = ["ERSAC", "MFOS"]  # ["ROMMEO", "ROMMEO"]  # ["PPO", "PPO"]
    config.NUM_AGENTS = 2  # TODO is this really the best way?

    return config
