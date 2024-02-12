# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:01:25 2023

@author: m.kiakojouri
"""

import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Print the name of the GPU
    print(torch.cuda.get_device_name(0))
else:
    print("GPU not available.")
