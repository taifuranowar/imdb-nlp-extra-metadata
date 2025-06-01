import torch
import gc
import os
import subprocess

def reset_gpu():
    print("Resetting GPU environment...")
    
    # Clear Python memory
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Kill Python processes (optional - be careful!)
    # subprocess.run(['pkill', '-f', 'python'], check=False)
    
    print("GPU reset complete!")

if __name__ == "__main__":
    reset_gpu()