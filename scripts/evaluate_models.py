#!/usr/bin/env python3
"""
Model Evaluation Script

This script uses unified efficient process management for model evaluation
to evaluate two chess models against each other. Each process plays multiple games
sequentially.
"""

import os
import sys
import argparse
import time
from datetime import datetime
import torch
import numpy as np
import random

# Import unified modules
from src.training.parallel_utils import run_parallel_task_execution
from src.training.parallel_workers import evaluation_worker_process


def set_seed(seed=42):
    """
    Set seeds for reproducibility across all random number generators
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For deterministic behavior in PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Set all random seeds to {seed} for reproducible evaluation")


def evaluate_models(old_model_path, new_model_path, num_games, num_simulations, cpu_utilization = 0.9, seed=42):
    """
    Evaluate two models against each other using unified parallel processing
    """
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print('\n')
    print(f"Old model: {os.path.basename(old_model_path)}")
    print(f"New model: {os.path.basename(new_model_path)}")
    print(f"Games: {num_games}")
    print(f"Simulations per move: {num_simulations}")
    print(f"CPU utilization: {cpu_utilization*100:.0f}%")
    
    task_config = {
        'total_tasks': num_games,
        'num_simulations': num_simulations,
        'old_model_path': old_model_path,
        'new_model_path': new_model_path,
        'seed': seed
    }
    
    start_time = time.time()
    game_results, process_statistics = run_parallel_task_execution(
        task_config=task_config,
        worker_function=evaluation_worker_process,
        cpu_utilization=cpu_utilization
    )
    execution_time = time.time() - start_time
    
    successful_games = [r for r in game_results if 'error' not in r]
    failed_games = [r for r in game_results if 'error' in r]
    
    # Count results from new model's perspective
    new_model_wins = 0
    old_model_wins = 0
    draws = 0
    
    for game in successful_games:
        result = game['result']
        white_is_new = game['white_is_new']
        
        if result == 1.0:  # White wins
            if white_is_new:
                new_model_wins += 1
            else:
                old_model_wins += 1
        elif result == -1.0:  # Black wins
            if not white_is_new:  # Black is new model
                new_model_wins += 1
            else:
                old_model_wins += 1
        else:  # Draw
            draws += 1
    
    total_games_played = len(successful_games)
    
    # Calculate score-based win rate (AlphaZero style: win=1.0, draw=0.5, loss=0.0)
    new_model_score = new_model_wins + 0.5 * draws
    new_model_score_rate = new_model_score / total_games_played * 100
    
    # Also calculate traditional win rate (only decisive games) for comparison
    decisive_games = new_model_wins + old_model_wins
    traditional_win_rate = new_model_wins / decisive_games * 100 if decisive_games > 0 else 0
    
    return {
        'new_model_wins': new_model_wins,
        'old_model_wins': old_model_wins,
        'draws': draws,
        'total_games': total_games_played,
        'decisive_games': decisive_games,
        'score_rate': new_model_score_rate,  # PRIMARY: Score-based rate (wins + 0.5*draws)
        'traditional_win_rate': traditional_win_rate,  # Traditional win rate (decisive only)
        'new_model_score': new_model_score,
        'execution_time': execution_time,
        'successful_games': successful_games,
        'failed_games': failed_games,
        'process_statistics': process_statistics
    }


def save_evaluation_results(results, old_model_path, new_model_path, output_dir = None):
    """
    Save evaluation results and statistics
    """

    if output_dir is None:
        output_dir = "evaluation_results"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    old_name = os.path.splitext(os.path.basename(old_model_path))[0]
    new_name = os.path.splitext(os.path.basename(new_model_path))[0]
    stats_filename = f"evaluation_{old_name}_vs_{new_name}_{timestamp}_stats.txt"
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, stats_filename)
    
    with open(stats_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Old model: {os.path.basename(old_model_path)}\n")
        f.write(f"New model: {os.path.basename(new_model_path)}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        
        f.write(f"Games played: {results['total_games']}\n")
        f.write(f"Simulations per move: {results.get('num_simulations', 'N/A')}\n")
        f.write(f"Execution time: {results['execution_time']:.2f} seconds ({results['execution_time']/60:.1f} minutes)\n\n")
        
        f.write("="*60 + "\n")
        f.write("GAME OUTCOMES\n")
        f.write("="*60 + "\n\n")
        
        total = results['total_games']
        f.write(f"New model wins: {results['new_model_wins']} ({results['new_model_wins']/total*100:.1f}%)\n")
        f.write(f"Old model wins: {results['old_model_wins']} ({results['old_model_wins']/total*100:.1f}%)\n")
        f.write(f"Draws: {results['draws']} ({results['draws']/total*100:.1f}%)\n")
        f.write(f"Decisive games: {results['decisive_games']}/{total} ({results['decisive_games']/total*100:.1f}%)\n\n")
        
        f.write("="*60 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Score-based rate: {results['score_rate']:.1f}% (wins + 0.5×draws) ← PRIMARY METRIC\n")
        f.write(f"Traditional win rate: {results['traditional_win_rate']:.1f}% (wins among decisive games)\n")
        f.write(f"New model score: {results['new_model_score']:.1f}/{total} points\n\n")
        
        if results.get('process_statistics'):
            f.write("="*60 + "\n")
            f.write("PROCESS STATISTICS\n")
            f.write("="*60 + "\n\n")
            
            for i, stats in enumerate(results['process_statistics']):
                if 'error' in stats:
                    continue
                f.write(f"Process {i}: {stats.get('gpu_device', 'N/A')}\n")
                f.write(f"  Games completed: {stats.get('tasks_completed', 0)}\n")
                f.write(f"  New model wins: {stats.get('new_model_wins', 0)}\n")
                f.write(f"  Old model wins: {stats.get('old_model_wins', 0)}\n")
                f.write(f"  Draws: {stats.get('draws', 0)}\n\n")
    
    print(f"Evaluation results saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Model Evaluation for Aldarion Chess Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--old_model', type=str, required=True)
    parser.add_argument('--new_model', type=str, required=True)
    parser.add_argument('--num_games', type=int, default=50)
    parser.add_argument('--num_simulations', type=int, default=200)
    parser.add_argument('--win_threshold', type=float, default=55.0)
    parser.add_argument('--cpu_utilization', type=float, default=0.9)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    os.makedirs("evaluation_results", exist_ok=True)
    try:
        
        results = evaluate_models(
            old_model_path=args.old_model,
            new_model_path=args.new_model,
            num_games=args.num_games,
            num_simulations=args.num_simulations,
            cpu_utilization=args.cpu_utilization,
            seed=args.seed
        )
        
        save_evaluation_results(results, args.old_model, args.new_model, args.output)
        
        score_rate = results['score_rate']
        if score_rate > args.win_threshold:
            print(f"\nACCEPT NEW MODEL!")
            print(f"New model score rate ({score_rate:.1f}%) exceeds threshold ({args.win_threshold}%)")
            sys.exit(0)
        else:
            print(f"\nREJECT NEW MODEL!")
            print(f"New model score rate ({score_rate:.1f}%) below threshold ({args.win_threshold}%)")
            sys.exit(1)
            
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()