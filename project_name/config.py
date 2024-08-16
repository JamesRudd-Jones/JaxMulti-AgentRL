from ml_collections import config_dict


def get_config():
    # PPO and MFOS
    config = config_dict.ConfigDict()
    config.SEED = 42
    config.LR = 0.025
    config.GAMMA = 0.96
    config.EPS = 1
    config.GRU_HIDDEN_DIM = 16*3  #  to make it divisible by 3 for MFOS, 256
    config.GAE_LAMBDA = 0.95
    config.UPDATE_EPOCHS = 4
    config.NUM_MINIBATCHES = 8  # TODO make it work for one env and one minibatch
    config.CLIP_EPS = 0.2
    config.VF_COEF = 0.5
    config.ENT_COEF = 0.01
    config.SCALE_CLIP_EPS = True  # False
    config.ANNEAL_LR = True
    config.MAX_GRAD_NORM = 0.5

    # PR2
    config.BUFFER_SIZE = 100000
    config.BATCH_SIZE = 32
    config.REPLAY_PRIORITY_EXP = 1.0
    config.VALUE_N_PARTICLES = 16
    config.KERNEL_N_PARTICLES = 32  # Think got to be double value_n particles
    config.KERNEL_UPDATE_RATIO = 0.5
    config.ANNEALING = 0.5  # TODO set this inside PR2 as it could be trained technically
    config.U_RANGE = 1.0

    # ROMMEO
    config.REGULARISER = 0.001

    # MELIBA
    config.LATENT_DIM = 2
    config.KL_WEIGHT = 0.01

    config.CNN = False
    # config.CNN = True

    # config.TOTAL_TIMESTEPS = 10000000
    config.NUM_UPDATES = 1000  # 40000  # 10000
    config.NUM_INNER_STEPS = 64  # 128
    config.NUM_META_STEPS = 1  # 500
    config.NUM_ENVS = 8  # MUST BE SAME SIZE OR BIGGER THAN NUM_MINIBATCHES
    config.NUM_DEVICES = 1

    config.WANDB = "disabled"  # "online" if want it to work
    # config.WANDB = "online"

    config.DISABLE_JIT = False
    # config.DISABLE_JIT = True

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    config.AGENT_TYPE = ["ROMMEO", "T4T"]  # ["PPO", "PPO"]
    config.NUM_AGENTS = 2  # TODO is this really the best way?

    return config