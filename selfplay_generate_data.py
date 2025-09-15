#!/usr/bin/env python3
"""
Unified Self-Play Training Data Generation for Aldarion Chess Engine

This script uses the unified parallel processing infrastructure to generate
training data through self-play games.
"""

import os
import sys
import argparse
import time
import pickle
from datetime import datetime
from typing import List, Dict

# Import unified modules
from parallel_utils import run_parallel_task_execution
from parallel_workers import selfplay_worker_process


def generate_selfplay_data(total_games: int, num_simulations: int, 
                          temperature: float, model_path: str,
                          cpu_utilization: float = 0.90,
                          max_processes_per_gpu: int = None,
                          output_filename: str = None) -> str:
    """
    Generate self-play training data using unified parallel processing
    
    Args:
        total_games: Total number of games to generate
        num_simulations: MCTS simulations per move
        temperature: Temperature for move selection
        model_path: Path to model weights
        cpu_utilization: Target CPU utilization (0.0 to 1.0)
        max_processes_per_gpu: Manual override for max processes per GPU
        output_filename: Optional output filename
    
    Returns:
        Path to saved training data file
    """
    print("="*60)
    print("UNIFIED SELF-PLAY DATA GENERATION")
    print("="*60)
    print(f"Total games: {total_games}")
    print(f"Simulations per move: {num_simulations}")
    print(f"Temperature: {temperature}")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"CPU utilization: {cpu_utilization*100:.0f}%")
    
    # Create task configuration
    task_config = {
        'total_tasks': total_games,
        'num_simulations': num_simulations,
        'temperature': temperature,
        'model_path': model_path
    }
    
    # Execute parallel self-play
    start_time = time.time()
    training_data, process_statistics = run_parallel_task_execution(
        task_config=task_config,
        worker_function=selfplay_worker_process,
        cpu_utilization=cpu_utilization,
        max_processes_per_gpu=max_processes_per_gpu
    )
    execution_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("SELF-PLAY GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Training examples generated: {len(training_data)}")
    print(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)")
    
    if len(training_data) == 0:
        print("❌ No training data generated!")
        return None
    
    # Save training data
    saved_file = save_training_data(training_data, process_statistics, output_filename)
    return saved_file


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
        output_filename = f"unified_selfplay_data_{timestamp}.pkl"
    
    # Create directories if they don't exist
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


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Unified Self-Play Training Data Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect optimal configuration
  python3 selfplay_unified.py --total_games 100
  
  # Use 90% CPU utilization
  python3 selfplay_unified.py --total_games 200 --cpu_utilization 0.90
  
  # Manual override: max 8 processes per GPU
  python3 selfplay_unified.py --total_games 500 --max_processes_per_gpu 8
  
  # High-intensity run with detailed statistics
  python3 selfplay_unified.py --total_games 1000 --num_simulations 200 --cpu_utilization 0.95
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
    
    # Create necessary directories
    os.makedirs("training_data", exist_ok=True)
    os.makedirs("training_data_stats", exist_ok=True)
    
    # Run self-play generation
    try:
        saved_file = generate_selfplay_data(
            total_games=args.total_games,
            num_simulations=args.num_simulations,
            temperature=args.temperature,
            model_path=args.model_path,
            cpu_utilization=args.cpu_utilization,
            max_processes_per_gpu=args.max_processes_per_gpu,
            output_filename=args.output
        )
        
        if saved_file:
            print(f"\n✅ Self-play data generation successful!")
            print(f"📁 Data saved to: {saved_file}")
            sys.exit(0)
        else:
            print(f"\n❌ Self-play data generation failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()