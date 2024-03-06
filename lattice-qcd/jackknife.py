import numpy as np

def jackknife(data, block_size, estimator=np.mean):
    """
    Perform a jackknife resampling on the data.

    Parameters:
    - data: The dataset (list or numpy array) to resample.
    - block_size: The size of each block.
    - estimator: The statistical estimator to use. Default is numpy.mean.

    Returns:
    - jackknife_ave: The average estimate from the jackknife resampling.
    - jackknife_error: The standard error of the estimate.
    """
    n_blocks = len(data) // block_size
    
    if n_blocks <= 0:
        print("Error: Not enough data points")
        return None, None
    
    # Truncate the data to a multiple of block_size and split into blocks
    data = np.array(data[:n_blocks * block_size])
    blocks = np.split(data, n_blocks)
    
    estimator_blocks = np.empty(n_blocks)
    for i, block in enumerate(blocks):
        try:
            estimator_blocks[i] = estimator(block)
        except Exception as e:
            print(f"Error processing block {i}: {e}")
            return None, None
  
    jackknife_ave = np.mean(estimator_blocks)
    
    # Calculate variance and standard error
    variance = np.sum((estimator_blocks - jackknife_ave)**2) * ((n_blocks - 1) / n_blocks)
    jackknife_error = np.sqrt(variance)
    
    return jackknife_ave, jackknife_error
