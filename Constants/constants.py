import torch

LEARNING_RATE = 1e-3 * 3

NUM_SESSIONS_CUDA = 4
NUM_SESSIONS_CPU = 16
FRAMES_FOR_UPDATE_TARGET = 5000
FRAMES_FOR_UPDATE_NETWORK = 200
BATCH_SIZE = 32
FRAME_SKIP = 4
NUM_FRAMES = 30_000

NUM_CHANNELS = 4
W = 84
H = 84

GAMMA = 0.99
SIZE = 50000


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")