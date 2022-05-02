import time
import torch
import numpy as np

def torch_setup():
    """Sets the device we work on."""
    
    print(f'Torch version: {torch.__version__}, CUDA: {torch.version.cuda}')
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print('WARNING: You may want to change the runtime to GPU!')
        DEVICE = 'cpu'
    else:
        DEVICE = 'cuda:0'
    return DEVICE

def set_seed(seed=None):
    """Sets the seeds of random number generators."""
    
    if seed is None:
        seed = time.time() # Take a random seed

    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return seed