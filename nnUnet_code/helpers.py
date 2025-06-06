import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .softmorph.dilation import Dilation
from .softmorph.erosion import Erosion
from .softmorph.closing import Closing
from .softmorph.opening import Opening
# import logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MORPH_OPS = {
    "dilation3d": Dilation(), 
    "erosion3d": Erosion(),
    "closing": Closing(),
    "opening": Opening(),
}

def get_morph_from_env():
    op_name = os.environ.get("SOFTMORPH_OP", "closing").lower()
    # breakpoint()
    # logger.info("######### Using %s as morphological operation #########", op_name)
    morph = MORPH_OPS.get(op_name, MORPH_OPS["closing"])
    return morph

def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    morph = get_morph_from_env()
    # breakpoint()
    x = torch.softmax(x, 0)
    if x.dim() == 4:
        x = morph(x.unsqueeze(0))
    else:
        x = morph(x.unsqueeze(0))
    return x[0]

def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    morph = get_morph_from_env()
    x = torch.softmax(x, 1)
    # breakpoint()
    if x.dim() == 5:
        x = morph(x)
    else:
        x = morph(x)
    return x



def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
