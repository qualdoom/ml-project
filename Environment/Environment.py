from IPython.display import clear_output
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchrl.envs import *
from torchrl.envs.libs.gym import *
from joblib import Parallel, delayed
from Constants.constants import *

def get_environment(name, frame_skip=FRAME_SKIP, width=W, height=H):
    return TransformedEnv(
        GymEnv(name, from_pixels=True, frame_skip=4),
        Compose(
            ToTensorImage(in_keys=["pixels"], out_keys=["pixels_trsf"]),
            Resize(in_keys=["pixels_trsf"], w=width, h=height),
            GrayScale(in_keys=["pixels_trsf"])
        )
    ).to(get_device())

