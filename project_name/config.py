from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.SEED = 42
    config.LR = 3e-4
    config.GAMMA = 0.96
    config.EPS = 1
    config.GRU_HIDDEN_DIM = 16*3  #  to make it divisible by 3 for MFOS, 256
    config.GAE_LAMBDA = 0.95
    config.UPDATE_EPOCHS = 2
    config.NUM_MINIBATCHS = 4
    config.CLIP_EPS = 0.2
    config.VF_COEF = 0.5
    config.ENT_COEF = 0.01

    config.ANNEAL_LR = False
    config.MAX_GRAD_NORM = 0.5

    # config.TOTAL_TIMESTEPS = 10000000
    config.NUM_UPDATES = 5000  # 10000
    config.NUM_INNER_STEPS = 100  # 128
    config.NUM_META_STEPS = 100
    config.NUM_ENVS = 4  # 8  MUST BE SAME SIZE OR BIGGER THAN NUM_MINIBATCHES
    config.NUM_DEVICES = 1

    config.BATCH_SIZE = 32

    # config.WANDB = "disabled"  # "online" if want it to work
    config.WANDB = "online"

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    config.AGENT_TYPE = ["PPO", "PPO"]  # ["PPO", "PPO"]
    config.NUM_AGENTS = 2  # TODO is this really the best way?

    return config