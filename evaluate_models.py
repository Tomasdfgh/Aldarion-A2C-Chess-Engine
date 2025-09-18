#!/usr/bin/env python3
"""
Unified Model Evaluation Script for Aldarion Chess Engine

This script uses unified efficient process management for model evaluation
to evaluate two chess models against each other. Each process plays multiple games
sequentially instead of creating one process per game.
"""

import os
import sys
import argparse
import time
import pickle
from datetime import datetime
from typing import Dict, List

# Import unified modules
from parallel_utils import run_parallel_task_execution, final_gpu_cleanup
from parallel_workers import evaluation_worker_process


def evaluate_models(old_model_path: str, new_model_path: str, 
                           num_games: int, num_simulations: int,
                           cpu_utilization: float = 0.8) -> Dict:
    """
    Evaluate two models against each other using unified parallel processing
    
    Args:
        old_model_path: Path to current best model
        new_model_path: Path to newly trained model
        num_games: Total number of games to play
        num_simulations: MCTS simulations per move
        cpu_utilization: Target CPU utilization
    
    Returns:
        Dictionary with evaluation results
    """
    print("="*60)
    print("UNIFIED MODEL EVALUATION")
    print("="*60)
    print(f"Old model: {os.path.basename(old_model_path)}")
    print(f"New model: {os.path.basename(new_model_path)}")
    print(f"Games: {num_games}")
    print(f"Simulations per move: {num_simulations}")
    print(f"CPU utilization: {cpu_utilization*100:.0f}%")
    
    # Create task configuration
    task_config = {
        'total_tasks': num_games,
        'num_simulations': num_simulations,
        'old_model_path': old_model_path,
        'new_model_path': new_model_path,
        'starting_game_id': 0
    }
    
    # Execute parallel evaluation
    start_time = time.time()
    game_results, process_statistics = run_parallel_task_execution(
        task_config=task_config,
        worker_function=evaluation_worker_process,
        cpu_utilization=cpu_utilization
    )
    execution_time = time.time() - start_time
    
    # Analyze results
    successful_games = [r for r in game_results if 'error' not in r]
    failed_games = [r for r in game_results if 'error' in r]
    
    if len(successful_games) == 0:
        print("No games completed successfully!")
        return {'error': 'No successful games'}
    
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
    
    # Calculate aggregate statistics from process stats
    total_processes = len([s for s in process_statistics if 'error' not in s])
    avg_games_per_process = total_games_played / total_processes if total_processes > 0 else 0
    
    # Evaluation summary
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)")
    print(f"Games played: {total_games_played}/{num_games}")
    print(f"Failed games: {len(failed_games)}")
    print(f"Processes used: {total_processes}")
    print(f"Average games per process: {avg_games_per_process:.1f}")
    
    print(f"\nResults (from new model's perspective):")
    print(f"  New model wins: {new_model_wins} ({new_model_wins/total_games_played*100:.1f}%)")
    print(f"  Old model wins: {old_model_wins} ({old_model_wins/total_games_played*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/total_games_played*100:.1f}%)")
    print(f"  Decisive games: {decisive_games}/{total_games_played} ({decisive_games/total_games_played*100:.1f}%)")
    
    print(f"\nEvaluation metrics:")
    print(f"  Score-based rate: {new_model_score_rate:.1f}% (wins + 0.5×draws) ← PRIMARY METRIC")
    print(f"  Traditional win rate: {traditional_win_rate:.1f}% (wins among decisive games)")
    print(f"  New model score: {new_model_score:.1f}/{total_games_played} points")
    
    if successful_games:
        avg_game_time = sum(g['game_time_seconds'] for g in successful_games) / len(successful_games)
        avg_moves = sum(g['move_count'] for g in successful_games) / len(successful_games)
        print(f"\nGame statistics:")
        print(f"  Average game time: {avg_game_time:.1f} seconds")
        print(f"  Average moves per game: {avg_moves:.1f}")
        print(f"  Games per minute: {total_games_played / (execution_time/60):.1f}")
    
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


def save_evaluation_results(results: Dict, old_model_path: str, new_model_path: str, 
                          output_filename: str = None) -> str:
    """
    Save evaluation results and statistics
    
    Args:
        results: Evaluation results dictionary
        old_model_path: Path to old model
        new_model_path: Path to new model
        output_filename: Optional output filename
    
    Returns:
        Filename where data was saved
    """
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        old_name = os.path.splitext(os.path.basename(old_model_path))[0]
        new_name = os.path.splitext(os.path.basename(new_model_path))[0]
        output_filename = f"evaluation_{old_name}_vs_{new_name}_{timestamp}.pkl"
    
    # Create directories if they don't exist
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Generate full path
    results_path = os.path.join("evaluation_results", output_filename)
    
    # Add metadata to results
    results['metadata'] = {
        'old_model_path': old_model_path,
        'new_model_path': new_model_path,
        'timestamp': datetime.now().isoformat(),
        'evaluation_type': 'unified_parallel'
    }
    
    # Save results
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Evaluation results saved to: {results_path}")
    return results_path


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Unified Model Evaluation for Aldarion Chess Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python3 evaluate_models.py --old_model model_v1.pth --new_model model_v2.pth --num_games 30 --num_simulations 200
  
  # High-quality evaluation
  python3 evaluate_models.py --old_model model_weights.pth --new_model model_weights_v5.pth --num_games 100 --num_simulations 400
  
  # Fast evaluation with reduced CPU usage
  python3 evaluate_models.py --old_model current_best.pth --new_model candidate.pth --num_games 200 --num_simulations 800 --cpu_utilization 0.5

Notes:
- Score-based win rate is the primary metric (wins + 0.5*draws)
- New model win rate >55% typically means the new model is significantly stronger
- Each process plays multiple games sequentially for better efficiency
        """
    )
    
    parser.add_argument('--old_model', type=str, required=True,
                        help='Path to the current best model')
    parser.add_argument('--new_model', type=str, required=True,
                        help='Path to the newly trained model to evaluate')
    parser.add_argument('--num_games', type=int, default=50,
                        help='Number of games to play (default: 50)')
    parser.add_argument('--num_simulations', type=int, default=200,
                        help='MCTS simulations per move (default: 200)')
    parser.add_argument('--win_threshold', type=float, default=55.0,
                        help='Win rate threshold for accepting new model (default: 55.0%)')
    parser.add_argument('--cpu_utilization', type=float, default=0.8,
                        help='CPU utilization for parallel processing (default: 0.8)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename for evaluation results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.old_model):
        print(f"Error: Old model file not found: {args.old_model}")
        sys.exit(1)
    
    if not os.path.exists(args.new_model):
        print(f"Error: New model file not found: {args.new_model}")
        sys.exit(1)
    
    if args.num_games <= 0:
        print("Error: num_games must be positive")
        sys.exit(1)
    
    if not (0.0 <= args.cpu_utilization <= 1.0):
        print("Error: cpu_utilization must be between 0.0 and 1.0")
        sys.exit(1)
    
    if not (0.0 <= args.win_threshold <= 100.0):
        print("Error: win_threshold must be between 0.0 and 100.0")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Run evaluation
    try:
        print(f"  Old model: {args.old_model}")
        print(f"  New model: {args.new_model}")
        print(f"  Target: >{args.win_threshold}% score rate for acceptance")
        
        results = evaluate_models(
            old_model_path=args.old_model,
            new_model_path=args.new_model,
            num_games=args.num_games,
            num_simulations=args.num_simulations,
            cpu_utilization=args.cpu_utilization
        )
        
        if 'error' in results:
            print(f"Evaluation failed: {results['error']}")
            sys.exit(1)
        
        # Save results
        results_file = save_evaluation_results(results, args.old_model, args.new_model, args.output)
        
        # Determine acceptance/rejection
        score_rate = results['score_rate']
        if score_rate > args.win_threshold:
            print(f"\nACCEPT NEW MODEL!")
            print(f"New model score rate ({score_rate:.1f}%) exceeds threshold ({args.win_threshold}%)")
            sys.exit(0)  # Success code for acceptance
        else:
            print(f"\nREJECT NEW MODEL!")
            print(f"New model score rate ({score_rate:.1f}%) below threshold ({args.win_threshold}%)")
            sys.exit(1)  # Failure code for rejection
            
    except KeyboardInterrupt:
        print(f"\nEvaluation interrupted by user")
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