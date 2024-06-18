from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.SEED = 42
    config.LR = 3e-4
    config.GAMMA = 0.995
    config.EPS = 1
    config.TOTAL_TIMESTEPS = 10000000

    config.NUM_ENVS = 1

    config.NUM_ENSEMBLE = 10
    config.RP_NOISE = 0.1
    config.SIGMA_SCALE = 0.005

    config.TAU = 1.0

    config.AUTOTUNE = False
    config.TARGET_ENT_SCALE = 0.89
    config.ALPHA = 0.2

    config.BUFFER_SIZE = 100000
    config.LEARNING_STARTS = 80000  # 10
    config.TRAIN_FREQ = 4
    config.TARGET_NETWORK_FREQ = 1000
    config.NSTEP = 1

    config.BATCH_SIZE = 32

    config.START_EPS = 1
    config.END_EPS = 0.01
    config.EPS_DECAY = 0.10

    # config.WANDB = "disabled"  # "online" if want it to work
    config.WANDB = "online"

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    return config