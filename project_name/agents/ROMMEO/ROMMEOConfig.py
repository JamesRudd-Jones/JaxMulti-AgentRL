from ml_collections import config_dict


def get_ROMMEO_config():
    config = config_dict.ConfigDict()
    config.LR = 1e-3
    config.GAMMA = 0.95
    config.BUFFER_SIZE = 100000
    config.BATCH_SIZE = 32
    config.REPLAY_PRIORITY_EXP = 1.0
    config.REGULARISER = 0.001
    config.REPARAMETERISE = True
    config.SQUASH = True
    config.TARGET_UPDATE_INTERVAL = 1  # TODO check this
    config.TAU = 0.01
    config.DISCRETE = True
    config.ANNEALING = 0.5

    return config