from ml_collections import config_dict


def get_config():
    # PPO and MFOS
    config = config_dict.ConfigDict()
    config.SEED = 42

    config.CNN = False
    # config.CNN = True

    # config.TOTAL_TIMESTEPS = 10000000
    config.NUM_INNER_STEPS = 16  # 30  # 128  # ep rollout length
    config.NUM_META_STEPS = 600  # 100  # 500  # number of ep rollouts to run
    config.NUM_UPDATES = 50  # 500  # number of meta rollouts, should be 1 for no meta training
    config.NUM_ENVS = 256
    config.NUM_DEVICES = 1

    # config.DEEP_SEA_MAP = 1  # 20

    # config.WANDB = "disabled"  # "online" if want it to work
    config.WANDB = "online"

    config.DISABLE_JIT = False
    # config.DISABLE_JIT = True

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    config.AGENT_TYPE = ["MFOS", "ERSAC"]  # ["ROMMEO", "ROMMEO"]  # ["PPO", "PPO"]
    config.NUM_AGENTS = 2  # TODO is this really the best way?

    return config
