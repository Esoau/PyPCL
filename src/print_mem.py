import psutil
import torch 
import os

def print_memory_usage(step_name=""):
    """Prints current CPU and GPU memory usage."""
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 * 1024)  # in MB
    print(f"\n--- Memory Usage after: {step_name} ---")
    print(f"CPU Memory Usage: {cpu_mem:.2f} MB")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 * 1024)  # in MB
        gpu_mem_cached = torch.cuda.memory_reserved() / (1024 * 1024)  # in MB
        print(f"GPU Memory Allocated: {gpu_mem_alloc:.2f} MB")
        print(f"GPU Memory Reserved: {gpu_mem_cached:.2f} MB")
    print("-------------------------------------------------")