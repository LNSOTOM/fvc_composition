
# Release Cache
import gc

def report_gpu():
    print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()
    
#%%
def check_memory_status():
    """Check memory status and perform garbage collection."""
    import gc
    import torch
    import psutil
    
    # Get initial memory stats
    if torch.cuda.is_available():
        initial_gpu = torch.cuda.memory_allocated() / (1024 ** 3)
    initial_ram = psutil.Process().memory_info().rss / (1024 ** 3)
    
    # Run garbage collection and get result
    collected = gc.collect()
    
    # Empty CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        final_gpu = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_diff = initial_gpu - final_gpu
        print(f"GPU Memory: Before {initial_gpu:.2f}GB, After {final_gpu:.2f}GB, Freed {gpu_diff:.2f}GB")
    
    # Check RAM after collection
    final_ram = psutil.Process().memory_info().rss / (1024 ** 3)
    ram_diff = initial_ram - final_ram
    
    print(f"RAM Memory: Before {initial_ram:.2f}GB, After {final_ram:.2f}GB, Freed {ram_diff:.2f}GB")
    print(f"Garbage collector removed {collected} objects")
    
    return collected

print(check_memory_status())

#%%
import psutil
import gc

def report_ram():
    mem = psutil.virtual_memory()
    used_gb = (mem.total - mem.available) / (1024 ** 3)
    print(f"RAM used: {used_gb:.2f} GB of {mem.total / (1024**3):.2f} GB")

# Usage
report_ram()
gc.collect()
report_ram()

#%%
def reset_torch_environment(verbose=True):
    import gc
    import torch
    import psutil

    if verbose: print("🔁 Cleaning up environment...")

    # Step 1: Selectively delete user variables (skip core ones)
    safe_names = {"gc", "torch", "psutil", "reset_torch_environment"}
    for name in list(globals()):
        if not name.startswith('_') and name not in safe_names:
            try:
                del globals()[name]
            except Exception as e:
                if verbose:
                    print(f"⚠️ Could not delete {name}: {e}")

    # Step 2: Garbage collection (clears CPU memory)
    gc.collect()
    if verbose: print("✅ Garbage collection done.")

    # Step 3: Empty CUDA cache (clears GPU memory)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if verbose:
            print("✅ CUDA cache cleared.")
            print(f"📉 Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"📉 Memory Reserved : {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Step 4: RAM usage info
    if verbose:
        vm = psutil.virtual_memory()
        print(f"💾 RAM Usage: {vm.used / 1024**3:.2f} GB / {vm.total / 1024**3:.2f} GB")

    if verbose: print("✅ Torch environment successfully reset.\n")


#%%
reset_torch_environment()

