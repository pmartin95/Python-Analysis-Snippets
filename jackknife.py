import numpy as np

def jackknife(data, block_size, estimator=np.mean):
    n_blocks = len(data) // block_size
    
    if n_blocks <= 0:
        print("Error: Not enough data points")
        return None, None
    
    data = np.array(data[:n_blocks * block_size])
    blocks = np.split(data, n_blocks)
    
    estimator_blocks = []
    for block in blocks:
        try:
            local_estimate = estimator(block)
        except Exception as e:
            print(f"Error: {e}")
            return None, None
        estimator_blocks.append(local_estimate)
  
    jackknife_ave = np.mean(estimator_blocks)
    
    variance = np.sum((estimator_blocks - jackknife_ave)**2) * ((n_blocks - 1) / n_blocks)
    return jackknife_ave, np.sqrt(variance)
import numpy as np

def jackknife(data, block_size, estimator=np.mean):
    n_blocks = len(data) // block_size
    
    if n_blocks <= 0:
        print("Error: Not enough data points")
        return None, None
    
    data = np.array(data[:n_blocks * block_size])
    blocks = np.split(data, n_blocks)
    
    estimator_blocks = []
    for block in blocks:
        try:
            local_estimate = estimator(block)
        except Exception as e:
            print(f"Error: {e}")
            return None, None
        estimator_blocks.append(local_estimate)
  
    jackknife_ave = np.mean(estimator_blocks)
    
    variance = np.sum((estimator_blocks - jackknife_ave)**2) * ((n_blocks - 1) / n_blocks)
    return jackknife_ave, np.sqrt(variance)
