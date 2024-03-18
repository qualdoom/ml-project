import torch

LEARNING_RATE = 1e-3 * 1

NUM_SESSIONS_CUDA = 4
NUM_SESSIONS_CPU = 16
FRAMES_FOR_UPDATE_TARGET = 5000
BATCH_SIZE = 32
FRAME_SKIP = 1
NUM_FRAMES = 100_000

NUM_CHANNELS = 4
W = 64
H = 64

GAMMA = 0.99
SIZE = 5000


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")