#!/usr/bin/env python3
"""
Parallel Training Data Generation for Chess MCTS

This script implements multi-GPU and multi-CPU parallelization for training data generation.
It uses the existing MTCS.run_game() function unchanged, adding parallelization layers on top.

Architecture:
- Multi-GPU: Distributes games across available GPUs
- Multi-CPU: Each GPU runs multiple processes in parallel
- One model per process: Simple, no shared state
- Result aggregation: Combines all training data efficiently

Performance Target: 10-20x speedup over sequential generation
"""

import os
import sys
import torch
import multiprocessing as mp
import argparse
import pickle
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any
import traceback

# Import existing modules
import MTCS as mt
import model as md


def detect_available_gpus() -> List[Dict[str, Any]]:
    """
    Detect available CUDA GPUs with detailed information
    
    Returns:
        List of GPU info dictionaries with device, name, memory, etc.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return [{'device': 'cpu', 'name': 'CPU', 'memory_gb': 0, 'max_processes': mp.cpu_count()}]
    
    gpu_count = torch.cuda.device_count()
    gpu_info = []
    
    print(f"Detected {gpu_count} GPU(s):")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        
        # Estimate max processes based on GPU memory (assuming ~500MB per model + inference)
        estimated_max_processes = max(1, int(gpu_memory * 0.8 / 0.5))  # 80% memory usage, 500MB per process
        
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
    
    Args:
        gpu_info: List of GPU information dictionaries
        cpu_utilization: Target CPU utilization (0.0 to 1.0)
        max_processes_per_gpu: Manual override for max processes
    
    Returns:
        Optimal processes per GPU
    """
    cpu_cores = mp.cpu_count()
    num_gpus = len(gpu_info)
    
    if max_processes_per_gpu is not None:
        return min(max_processes_per_gpu, cpu_cores // num_gpus)
    
    # Calculate based on CPU cores and target utilization
    cpu_based_processes = max(1, int((cpu_cores * cpu_utilization) // num_gpus))
    
    # Calculate based on GPU memory constraints
    if gpu_info[0]['device'] != 'cpu':
        memory_based_processes = min(gpu['max_processes'] for gpu in gpu_info)
    else:
        memory_based_processes = cpu_cores
    
    # Use the minimum of both constraints
    optimal_processes = min(cpu_based_processes, memory_based_processes)
    
    print(f"Process calculation:")
    print(f"  CPU cores: {cpu_cores}")
    print(f"  Target CPU utilization: {cpu_utilization * 100:.0f}%")
    print(f"  CPU-based processes per GPU: {cpu_based_processes}")
    print(f"  Memory-based processes per GPU: {memory_based_processes}")
    print(f"  Optimal processes per GPU: {optimal_processes}")
    
    return optimal_processes


def calculate_workload_distribution(total_games: int, gpu_info: List[Dict], processes_per_gpu: int) -> Dict[str, List[int]]:
    """
    Distribute games across GPUs and processes with balanced GPU utilization
    
    Args:
        total_games: Total number of games to generate
        gpu_info: List of GPU information dictionaries
        processes_per_gpu: Number of CPU processes per GPU
    
    Returns:
        Dictionary mapping GPU device -> list of games per process
        Example: {'cuda:0': [5, 5, 4, 4], 'cuda:1': [5, 5, 4, 4]}
    """
    num_gpus = len(gpu_info)
    total_processes = num_gpus * processes_per_gpu
    
    # Initialize distribution
    distribution = {}
    for gpu in gpu_info:
        distribution[gpu['device']] = [0] * processes_per_gpu
    
    # Distribute games by alternating between GPUs first, then round-robin within each GPU
    # This ensures balanced distribution across GPUs regardless of total process count
    for game_idx in range(total_games):
        # Alternate between GPUs first for better balance
        gpu_idx = game_idx % num_gpus
        # Then round-robin within processes on that GPU
        games_assigned_to_gpu = game_idx // num_gpus
        process_within_gpu = games_assigned_to_gpu % processes_per_gpu
        
        gpu_device = gpu_info[gpu_idx]['device']
        distribution[gpu_device][process_within_gpu] += 1
    
    # Print distribution for debugging
    print(f"Balanced workload distribution:")
    for gpu_device, games_list in distribution.items():
        active_processes = sum(1 for games in games_list if games > 0)
        total_games_gpu = sum(games_list)
        print(f"  {gpu_device}: {games_list}")
        print(f"    Active processes: {active_processes}/{processes_per_gpu}, Total games: {total_games_gpu}")
    
    return distribution


def worker_process(gpu_device: str, num_games: int, num_simulations: int, 
                  temperature: float, model_path: str, process_id: int) -> Tuple[List, Dict]:
    """
    Worker process that generates training data for assigned games
    
    Args:
        gpu_device: GPU device to use (e.g., 'cuda:0')
        num_games: Number of games for this process to generate
        num_simulations: MCTS simulations per move
        temperature: Temperature for move selection
        model_path: Path to model weights
        process_id: Unique process identifier
    
    Returns:
        Tuple of (training_data, process_stats)
    """
    start_time = time.time()
    
    try:
        print(f"Process {process_id}: Starting on {gpu_device} with {num_games} games")
        
        # Setup device
        device = torch.device(gpu_device)
        
        # Load model (each process gets its own copy)
        model_load_start = time.time()
        model = md.ChessNet()
        model.to(device)
        
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            print(f"Process {process_id}: Model loaded successfully on {gpu_device}")
        else:
            print(f"Process {process_id}: Warning - Model file {model_path} not found. Using random weights.")
        
        model_load_time = time.time() - model_load_start
        
        # Generate training data using existing function
        all_training_data = []
        games_completed = 0
        game_lengths = []
        game_outcomes = []
        total_training_examples = 0
        
        for game_num in range(num_games):
            try:
                game_start_time = time.time()
                print(f"Process {process_id}: Game {game_num + 1}/{num_games}")
                
                # Use existing run_game function unchanged
                training_data = mt.run_game(model, temperature, num_simulations, device)
                all_training_data.extend(training_data)
                games_completed += 1
                
                # Collect game statistics
                game_length = len(training_data)
                game_lengths.append(game_length)
                total_training_examples += game_length
                
                # Extract game outcome from last training example
                if training_data:
                    outcome = training_data[-1][3]  # (board_fen, history_fens, move_probs, outcome)
                    game_outcomes.append(outcome)
                
                game_time = time.time() - game_start_time
                print(f"Process {process_id}: Game {game_num + 1} completed in {game_time:.1f}s, {len(training_data)} examples")
                
            except Exception as e:
                print(f"Process {process_id}: Error in game {game_num + 1}: {e}")
                continue
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate detailed statistics
        white_wins = sum(1 for outcome in game_outcomes if outcome > 0)
        black_wins = sum(1 for outcome in game_outcomes if outcome < 0)
        draws = sum(1 for outcome in game_outcomes if outcome == 0)
        
        avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
        min_game_length = min(game_lengths) if game_lengths else 0
        max_game_length = max(game_lengths) if game_lengths else 0
        
        # Enhanced process statistics
        process_stats = {
            'process_id': process_id,
            'gpu_device': gpu_device,
            'games_requested': num_games,
            'games_completed': games_completed,
            'training_examples': len(all_training_data),
            'total_time_seconds': total_time,
            'model_load_time_seconds': model_load_time,
            'games_per_minute': (games_completed / total_time) * 60 if total_time > 0 else 0,
            'examples_per_minute': (len(all_training_data) / total_time) * 60 if total_time > 0 else 0,
            'game_outcomes': {
                'white_wins': white_wins,
                'black_wins': black_wins,
                'draws': draws
            },
            'game_length_stats': {
                'average': avg_game_length,
                'minimum': min_game_length,
                'maximum': max_game_length
            },
            'simulations_per_move': num_simulations,
            'temperature': temperature
        }
        
        print(f"Process {process_id}: Completed {games_completed}/{num_games} games in {total_time:.1f}s")
        print(f"Process {process_id}: {len(all_training_data)} examples, {games_completed/total_time*60:.1f} games/min")
        return all_training_data, process_stats
        
    except Exception as e:
        print(f"Process {process_id}: Fatal error: {e}")
        traceback.print_exc()
        return [], {
            'process_id': process_id, 
            'gpu_device': gpu_device,
            'error': str(e),
            'total_time_seconds': time.time() - start_time
        }


def run_parallel_training_generation(total_games: int, num_simulations: int, 
                                    temperature: float, model_path: str, 
                                    cpu_utilization: float = 0.90,
                                    max_processes_per_gpu: int = None) -> Tuple[List, List]:
    """
    Coordinate parallel training data generation across GPUs and CPU processes
    
    Args:
        total_games: Total number of games to generate
        num_simulations: MCTS simulations per move
        temperature: Temperature for move selection
        model_path: Path to model weights
        cpu_utilization: Target CPU utilization (0.0 to 1.0)
        max_processes_per_gpu: Manual override for max processes per GPU
    
    Returns:
        Tuple of (all_training_data, process_statistics)
    """
    print("="*60)
    print("PARALLEL TRAINING DATA GENERATION")
    print("="*60)
    
    # Detect hardware
    gpu_info = detect_available_gpus()
    
    # Calculate optimal processes per GPU
    processes_per_gpu = calculate_optimal_processes_per_gpu(
        gpu_info, cpu_utilization, max_processes_per_gpu
    )
    
    total_processes = len(gpu_info) * processes_per_gpu
    
    print(f"\nConfiguration:")
    print(f"  Total games: {total_games}")
    print(f"  GPUs: {len(gpu_info)}")
    print(f"  Processes per GPU: {processes_per_gpu}")
    print(f"  Total processes: {total_processes}")
    print(f"  Expected CPU utilization: {(total_processes / mp.cpu_count()) * 100:.1f}%")
    
    # Calculate workload distribution
    workload = calculate_workload_distribution(total_games, gpu_info, processes_per_gpu)
    
    print(f"\nWorkload distribution:")
    for gpu_device, games_list in workload.items():
        print(f"  {gpu_device}: {games_list} (total: {sum(games_list)} games)")
    
    # Start parallel processes
    print(f"\nStarting parallel execution...")
    start_time = time.time()
    
    with mp.Pool(processes=total_processes) as pool:
        # Create process arguments
        process_args = []
        process_id = 0
        
        for gpu_device, games_list in workload.items():
            for num_games in games_list:
                if num_games > 0:  # Only create processes with work to do
                    args = (gpu_device, num_games, num_simulations, temperature, model_path, process_id)
                    process_args.append(args)
                    process_id += 1
        
        # Execute processes in parallel
        print(f"Launching {len(process_args)} worker processes...")
        results = pool.starmap(worker_process, process_args)
    
    # Aggregate results
    all_training_data = []
    process_statistics = []
    
    for training_data, stats in results:
        all_training_data.extend(training_data)
        process_statistics.append(stats)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Calculate aggregate statistics
    total_games_completed = sum(stats.get('games_completed', 0) for stats in process_statistics)
    total_white_wins = sum(stats.get('game_outcomes', {}).get('white_wins', 0) for stats in process_statistics)
    total_black_wins = sum(stats.get('game_outcomes', {}).get('black_wins', 0) for stats in process_statistics)
    total_draws = sum(stats.get('game_outcomes', {}).get('draws', 0) for stats in process_statistics)
    
    avg_games_per_min = sum(stats.get('games_per_minute', 0) for stats in process_statistics if 'games_per_minute' in stats)
    avg_examples_per_min = sum(stats.get('examples_per_minute', 0) for stats in process_statistics if 'examples_per_minute' in stats)
    
    # Print detailed summary
    print("="*60)
    print("EXECUTION COMPLETE")
    print("="*60)
    print(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)")
    print(f"Games completed: {total_games_completed}/{total_games}")
    print(f"Total training examples: {len(all_training_data)}")
    print(f"Average game length: {len(all_training_data) / total_games_completed:.1f} moves" if total_games_completed > 0 else "")
    
    print(f"\nPerformance metrics:")
    print(f"  Games per minute (aggregate): {avg_games_per_min:.1f}")
    print(f"  Examples per minute (aggregate): {avg_examples_per_min:.1f}")
    print(f"  Examples per second: {len(all_training_data) / execution_time:.2f}")
    
    print(f"\nGame outcomes:")
    print(f"  White wins: {total_white_wins} ({total_white_wins/total_games_completed*100:.1f}%)" if total_games_completed > 0 else "")
    print(f"  Black wins: {total_black_wins} ({total_black_wins/total_games_completed*100:.1f}%)" if total_games_completed > 0 else "")
    print(f"  Draws: {total_draws} ({total_draws/total_games_completed*100:.1f}%)" if total_games_completed > 0 else "")
    
    return all_training_data, process_statistics


def save_training_data(training_data: List, process_stats: List, output_filename: str = None) -> str:
    """
    Save training data and process statistics
    
    Args:
        training_data: List of training examples
        process_stats: List of process statistics
        output_filename: Optional output filename
    
    Returns:
        Filename where data was saved
    """
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"parallel_training_data_{timestamp}.pkl"
    
    # Create directories if they don't exist
    import os
    os.makedirs("training_data", exist_ok=True)
    os.makedirs("training_data_stats", exist_ok=True)
    
    # Generate full paths with proper organization
    main_data_path = os.path.join("training_data", output_filename)
    base_filename = os.path.splitext(output_filename)[0]
    stats_filename = f"{base_filename}_stats.pkl"
    stats_path = os.path.join("training_data_stats", stats_filename)
    
    # Save training data
    with open(main_data_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    # Save statistics
    with open(stats_path, 'wb') as f:
        pickle.dump(process_stats, f)
    
    print(f"Training data saved to: {main_data_path}")
    print(f"Process statistics saved to: {stats_path}")
    
    return main_data_path


def create_stats_summary(process_stats: List[Dict]) -> str:
    """Create a human-readable summary of the statistics"""
    if not process_stats:
        return "No statistics available"
    
    summary = []
    summary.append("="*60)
    summary.append("DETAILED STATISTICS SUMMARY")
    summary.append("="*60)
    
    # Overall statistics
    total_games = sum(stats.get('games_completed', 0) for stats in process_stats)
    total_examples = sum(stats.get('training_examples', 0) for stats in process_stats)
    total_time = max(stats.get('total_time_seconds', 0) for stats in process_stats)
    
    summary.append(f"Overall Performance:")
    summary.append(f"  Total games: {total_games}")
    summary.append(f"  Total training examples: {total_examples}")
    summary.append(f"  Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    summary.append(f"  Average examples per game: {total_examples/total_games:.1f}" if total_games > 0 else "")
    
    # Per-process breakdown
    summary.append(f"\nPer-Process Breakdown:")
    for stats in process_stats:
        if 'error' in stats:
            summary.append(f"  Process {stats['process_id']} ({stats.get('gpu_device', 'unknown')}): ERROR - {stats['error']}")
        else:
            gpu = stats.get('gpu_device', 'unknown')
            games = stats.get('games_completed', 0)
            examples = stats.get('training_examples', 0)
            time_taken = stats.get('total_time_seconds', 0)
            games_per_min = stats.get('games_per_minute', 0)
            
            summary.append(f"  Process {stats['process_id']} ({gpu}):")
            summary.append(f"    Games: {games}, Examples: {examples}")
            summary.append(f"    Time: {time_taken:.1f}s, Rate: {games_per_min:.1f} games/min")
            
            # Game outcomes for this process
            outcomes = stats.get('game_outcomes', {})
            if outcomes:
                w, b, d = outcomes.get('white_wins', 0), outcomes.get('black_wins', 0), outcomes.get('draws', 0)
                summary.append(f"    Outcomes: W:{w} B:{b} D:{d}")
    
    # Hardware utilization summary
    gpu_usage = {}
    for stats in process_stats:
        gpu = stats.get('gpu_device', 'unknown')
        if gpu not in gpu_usage:
            gpu_usage[gpu] = {'processes': 0, 'games': 0, 'examples': 0}
        gpu_usage[gpu]['processes'] += 1
        gpu_usage[gpu]['games'] += stats.get('games_completed', 0)
        gpu_usage[gpu]['examples'] += stats.get('training_examples', 0)
    
    summary.append(f"\nHardware Utilization:")
    for gpu, usage in gpu_usage.items():
        summary.append(f"  {gpu}: {usage['processes']} processes, {usage['games']} games, {usage['examples']} examples")
    
    return "\n".join(summary)


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Parallel MCTS Training Data Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect optimal configuration
  python3 parallel_training_data.py --total_games 100
  
  # Use 90% CPU utilization
  python3 parallel_training_data.py --total_games 200 --cpu_utilization 0.90
  
  # Manual override: max 8 processes per GPU
  python3 parallel_training_data.py --total_games 500 --max_processes_per_gpu 8
  
  # High-intensity run with detailed statistics
  python3 parallel_training_data.py --total_games 1000 --num_simulations 200 --cpu_utilization 0.95
        """
    )
    
    parser.add_argument('--total_games', type=int, default=100,
                        help='Total number of games to generate (default: 100)')
    parser.add_argument('--num_simulations', type=int, default=100,
                        help='MCTS simulations per move (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature for move selection (default: 0.8)')
    parser.add_argument('--model_path', type=str, default='model_weights/model_weights.pth',
                        help='Path to model weights (default: model_weights/model_weights.pth)')
    parser.add_argument('--cpu_utilization', type=float, default=0.90,
                        help='Target CPU utilization 0.0-1.0 (default: 0.90)')
    parser.add_argument('--max_processes_per_gpu', type=int, default=None,
                        help='Manual override for max processes per GPU (auto-detect if not specified)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename for training data')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.total_games <= 0:
        print("Error: total_games must be positive")
        sys.exit(1)
    
    if not (0.0 <= args.cpu_utilization <= 1.0):
        print("Error: cpu_utilization must be between 0.0 and 1.0")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Warning: Model file {args.model_path} not found. Will use random weights.")
    
    # Run parallel training generation
    try:
        training_data, process_stats = run_parallel_training_generation(
            total_games=args.total_games,
            num_simulations=args.num_simulations,
            temperature=args.temperature,
            model_path=args.model_path,
            cpu_utilization=args.cpu_utilization,
            max_processes_per_gpu=args.max_processes_per_gpu
        )
        
        # Save results
        if len(training_data) > 0:
            filename = save_training_data(training_data, process_stats, args.output)
            
            # Print final statistics
            print(f"\n{'='*60}")
            print("FINAL RESULTS")
            print(f"{'='*60}")
            print(f"Training examples generated: {len(training_data)}")
            print(f"Saved to: {filename}")
            
            # Create and print detailed statistics summary
            stats_summary = create_stats_summary(process_stats)
            print(f"\n{stats_summary}")
            
            # Save statistics summary as text file
            base_name = os.path.basename(filename)
            summary_filename = base_name.replace('.pkl', '_summary.txt')
            summary_path = os.path.join("training_data_stats", summary_filename)
            with open(summary_path, 'w') as f:
                f.write(stats_summary)
            print(f"\nDetailed statistics saved to: {summary_path}")
            
        else:
            print("No training data generated!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    main()