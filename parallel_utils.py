#!/usr/bin/env python3
"""
Shared Parallel Processing Utilities for Aldarion Chess Engine

This module provides unified infrastructure for both self-play training data generation
and model evaluation, reducing code duplication and ensuring consistent behavior.
"""

import torch
import multiprocessing as mp
import time
from typing import List, Tuple, Dict, Any, Union


def detect_available_gpus():
    """
    Detect available CUDA GPUs with detailed information
    """
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return [{'device': 'cpu', 'name': 'CPU', 'memory_gb': 0, 'max_processes': mp.cpu_count()}]
    
    gpu_count = torch.cuda.device_count()
    gpu_info = []
    
    print(f"\nDetected {gpu_count} GPU(s):")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        estimated_max_processes = max(1, int(gpu_memory * 0.8 / 0.5)) # <-- allows for only 80 percent of gpu memory and assumes each process uses 500MB of gpu memory
        
        gpu_info.append({
            'device': f'cuda:{i}',
            'name': gpu_name,
            'memory_gb': gpu_memory,
            'max_processes': min(estimated_max_processes, mp.cpu_count() // len(range(gpu_count)))
        })
        
        print(f"  cuda:{i}: {gpu_name} ({gpu_memory:.1f}GB, est. max processes: {gpu_info[-1]['max_processes']})")
        
    return gpu_info


def calculate_optimal_processes_per_gpu(gpu_info: List[Dict], cpu_utilization: float = 0.90, 
                                      max_processes_per_gpu: int = None) -> int:
    """
    Calculate optimal number of processes per GPU based on hardware
    """
    cpu_cores = mp.cpu_count()
    num_gpus = len(gpu_info)
    
    if max_processes_per_gpu is not None:
        return min(max_processes_per_gpu, cpu_cores // num_gpus)
    
    cpu_based_processes = max(1, int((cpu_cores * cpu_utilization) // num_gpus))
    
    if gpu_info[0]['device'] != 'cpu':
        memory_based_processes = min(gpu['max_processes'] for gpu in gpu_info)
    else:
        memory_based_processes = cpu_cores
    
    optimal_processes = min(cpu_based_processes, memory_based_processes)
    
    print(f"\nProcess calculation:")
    print(f"  CPU cores: {cpu_cores}")
    print(f"  Target CPU utilization: {cpu_utilization * 100:.0f}%")
    print(f"  CPU-based processes per GPU: {cpu_based_processes}")
    print(f"  Memory-based processes per GPU: {memory_based_processes}")
    print(f"  Optimal processes per GPU: {optimal_processes}")
    
    return optimal_processes


def calculate_workload_distribution(total_tasks: int, gpu_info: List[Dict], processes_per_gpu: int) -> Dict[str, List[int]]:
    """
    Distribute tasks across GPUs and processes with balanced GPU utilization
    """
    num_gpus = len(gpu_info)
    
    distribution = {}
    for gpu in gpu_info:
        distribution[gpu['device']] = [0] * processes_per_gpu
    
    for task_idx in range(total_tasks):

        gpu_idx = task_idx % num_gpus
        tasks_assigned_to_gpu = task_idx // num_gpus
        process_within_gpu = tasks_assigned_to_gpu % processes_per_gpu
        
        gpu_device = gpu_info[gpu_idx]['device']
        distribution[gpu_device][process_within_gpu] += 1
    
    return distribution


def cleanup_gpu_memory(device: torch.device, process_id: int = None, models: List = None):
    """
    Perform explicit GPU memory cleanup
    """
    try:
        if models:
            for model in models:
                if model is not None:
                    del model
        
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if process_id is not None:
                print(f"Process {process_id}: GPU memory cleared")
            else:
                print("GPU memory cleared")

    except Exception as cleanup_error:
        if process_id is not None:
            print(f"Process {process_id}: Warning - GPU cleanup error: {cleanup_error}")
        else:
            print(f"Warning - GPU cleanup error: {cleanup_error}")


def final_gpu_cleanup():
    """
    Perform final GPU memory cleanup across all devices
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            print("Final GPU memory cleanup completed")
    except Exception as cleanup_error:
        print(f"Warning - Final GPU cleanup error: {cleanup_error}")


def create_process_statistics(process_id: int, gpu_device: str, start_time: float, 
                            tasks_completed: int, tasks_requested: int, **kwargs) -> Dict[str, Any]:
    """
    Create standardized process statistics dictionary
    """
    end_time = time.time()
    total_time = end_time - start_time
    
    base_stats = {
        'process_id': process_id,
        'gpu_device': gpu_device,
        'tasks_requested': tasks_requested,
        'tasks_completed': tasks_completed,
        'total_time_seconds': total_time,
        'tasks_per_minute': (tasks_completed / total_time) * 60 if total_time > 0 else 0,
    }
    
    base_stats.update(kwargs)
    
    return base_stats


def run_parallel_task_execution(task_config: Dict[str, Any], 
                               worker_function,
                               cpu_utilization: float = 0.90,
                               max_processes_per_gpu: int = None) -> Tuple[List, List]:

    print('\n')
    print("="*60)
    print("PARALLEL TASK EXECUTION")
    print("="*60)

    gpu_info = detect_available_gpus()
    processes_per_gpu = calculate_optimal_processes_per_gpu(
        gpu_info, cpu_utilization, max_processes_per_gpu
    )
    
    total_processes = len(gpu_info) * processes_per_gpu
    total_tasks = task_config.get('total_tasks', 0)
    
    print(f"\nConfiguration:")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Expected CPU utilization: {(total_processes / mp.cpu_count()) * 100:.1f}%")
    
    workload = calculate_workload_distribution(total_tasks, gpu_info, processes_per_gpu)
    
    print(f"\nWorkload distribution:")
    for gpu_device, tasks_list in workload.items():
        print(f"  {gpu_device}: {tasks_list} (total: {sum(tasks_list)} tasks)")

    print('\n')
    print("="*60)
    print("BEGIN EXECUTION")
    print("="*60)
    
    print(f"\nStarting parallel execution...")
    start_time = time.time()
    with mp.Pool(processes=total_processes) as pool:
        process_args = []
        process_id = 0
        
        for gpu_device, tasks_list in workload.items():
            for num_tasks in tasks_list:
                if num_tasks > 0:  # Only create processes with work to do
                    args = (gpu_device, num_tasks, task_config, process_id)
                    process_args.append(args)
                    process_id += 1
        
        print(f"Launching {len(process_args)} worker processes...")
        results = pool.starmap(worker_function, process_args)
    
    all_task_results = []
    process_statistics = []
    
    for task_results, stats in results:
        all_task_results.extend(task_results)
        process_statistics.append(stats)
    
    end_time = time.time()
    execution_time = end_time - start_time
    final_gpu_cleanup()
    print(f"\nExecution completed in {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)")
    
    return all_task_results, process_statistics