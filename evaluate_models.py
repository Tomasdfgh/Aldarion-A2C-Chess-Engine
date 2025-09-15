#!/usr/bin/env python3
"""
Parallel Model Evaluation Script for Aldarion Chess Engine

This script pits two chess models against each other using parallel processing
to determine which is stronger. Used in the training loop to decide whether 
to accept a new model or keep the previous best.

Architecture similar to parallel_training_data.py but for competitive games.
"""

import os
import sys
import torch
import chess
import argparse
import time
import multiprocessing as mp
from datetime import datetime
from typing import Tuple, Dict, List
import traceback

# Import existing modules
import model as md
import MTCS as mt


def play_competitive_game(white_model_path: str, black_model_path: str, 
                         num_simulations: int, device_str: str, game_id: int) -> Dict:
    """
    Play one competitive game between two models
    
    Args:
        white_model_path: Path to model playing white
        black_model_path: Path to model playing black  
        num_simulations: MCTS simulations per move
        device_str: Device string ('cuda:0', 'cpu', etc.)
        game_id: Unique game identifier
    
    Returns:
        Dictionary with game results and statistics
    """
    start_time = time.time()
    
    try:
        device = torch.device(device_str)
        
        # Load both models
        white_model = md.ChessNet()
        black_model = md.ChessNet()
        
        white_model.to(device)
        black_model.to(device)
        
        # Load weights
        white_state = torch.load(white_model_path, map_location=device, weights_only=True)
        black_state = torch.load(black_model_path, map_location=device, weights_only=True)
        
        white_model.load_state_dict(white_state)
        black_model.load_state_dict(black_state)
        
        white_model.eval()
        black_model.eval()
        
        # Play the game
        board = chess.Board()
        game_history = []
        move_count = 0
        
        while not board.is_game_over() and move_count < 300:  # Early stopping
            # Select model based on whose turn it is
            current_model = white_model if board.turn else black_model
            current_player = "White" if board.turn else "Black"
            
            try:
                # Get best move from model (deterministic for evaluation)
                best_move, _ = mt.get_best_move(
                    model=current_model,
                    board_fen=board.fen(),
                    num_simulations=num_simulations,
                    device=device,
                    game_history=game_history,
                    temperature=0.0  # Deterministic play for evaluation
                )
                
                if best_move is None:
                    break
                
                # Apply move
                move_obj = chess.Move.from_uci(best_move)
                board.push(move_obj)
                game_history.append(board.copy())
                move_count += 1
                
            except Exception as e:
                print(f"Game {game_id}: Error during {current_player} move: {e}")
                break
        
        # Determine game result
        if board.is_checkmate():
            # Winner gets +1, loser gets -1
            result = -1.0 if board.turn else 1.0  # If White's turn -> White checkmated -> Black wins
            result_str = "Black wins" if result == -1.0 else "White wins"
        elif board.is_stalemate() or board.is_insufficient_material() or \
             board.is_seventyfive_moves() or board.is_fivefold_repetition():
            result = 0.0
            result_str = "Draw"
        elif move_count >= 300:
            result = 0.0
            result_str = "Draw (300-ply limit)"
        else:
            result = 0.0
            result_str = "Draw (unexpected end)"
        
        game_time = time.time() - start_time
        
        game_result = {
            'game_id': game_id,
            'result': result,  # From White's perspective: +1=White wins, 0=Draw, -1=Black wins
            'result_str': result_str,
            'move_count': move_count,
            'game_time_seconds': game_time,
            'white_model': os.path.basename(white_model_path),
            'black_model': os.path.basename(black_model_path)
        }
        
        print(f"Game {game_id}: {result_str} in {move_count} moves ({game_time:.1f}s)")
        return game_result
        
    except Exception as e:
        print(f"Game {game_id}: Fatal error: {e}")
        traceback.print_exc()
        return {
            'game_id': game_id,
            'error': str(e),
            'game_time_seconds': time.time() - start_time
        }


def evaluate_models_parallel(old_model_path: str, new_model_path: str, 
                            num_games: int, num_simulations: int,
                            cpu_utilization: float = 0.8) -> Dict:
    """
    Evaluate two models against each other using parallel processing
    
    Args:
        old_model_path: Path to current best model
        new_model_path: Path to newly trained model
        num_games: Total number of games to play
        num_simulations: MCTS simulations per move
        cpu_utilization: Target CPU utilization
    
    Returns:
        Dictionary with evaluation results and statistics
    """
    print("=" * 60)
    print("PARALLEL MODEL EVALUATION")
    print("=" * 60)
    
    # Hardware detection (similar to parallel_training_data.py)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPU(s)")
        devices = [f'cuda:{i}' for i in range(num_gpus)]
    else:
        print("Using CPU")
        devices = ['cpu']
    
    # Calculate processes
    cpu_cores = mp.cpu_count()
    total_processes = max(1, int(cpu_cores * cpu_utilization))
    processes_per_device = max(1, total_processes // len(devices))
    
    print(f"Configuration:")
    print(f"  Total games: {num_games}")
    print(f"  CPU cores: {cpu_cores}")
    print(f"  Target CPU utilization: {cpu_utilization * 100:.0f}%")
    print(f"  Total processes: {total_processes}")
    print(f"  Processes per device: {processes_per_device}")
    
    # Distribute games across processes
    # Play equal games as White and Black (swap roles)
    games_per_process = num_games // total_processes
    remaining_games = num_games % total_processes
    
    process_args = []
    game_id = 0
    
    for device in devices:
        for process_idx in range(processes_per_device):
            if game_id >= num_games:
                break
            
            # Calculate games for this process
            games_for_process = games_per_process
            if remaining_games > 0:
                games_for_process += 1
                remaining_games -= 1
            
            if games_for_process == 0:
                break
            
            # Alternate who plays white/black for fairness
            for local_game in range(games_for_process):
                if (game_id + local_game) % 2 == 0:
                    # New model plays White
                    white_path = new_model_path
                    black_path = old_model_path
                else:
                    # Old model plays White  
                    white_path = old_model_path
                    black_path = new_model_path
                
                args = (white_path, black_path, num_simulations, device, game_id + local_game)
                process_args.append(args)
            
            game_id += games_for_process
    
    print(f"Launching {len(process_args)} games across {total_processes} processes...")
    print(f"New model plays White in ~50% of games for fair evaluation")
    
    # Execute games in parallel
    start_time = time.time()
    
    with mp.Pool(processes=total_processes) as pool:
        results = pool.starmap(play_competitive_game, process_args)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Analyze results
    successful_games = [r for r in results if 'error' not in r]
    failed_games = [r for r in results if 'error' in r]
    
    if len(successful_games) == 0:
        print("❌ No games completed successfully!")
        return {'error': 'No successful games'}
    
    # Count results from new model's perspective
    new_model_wins = 0
    old_model_wins = 0
    draws = 0
    
    for game in successful_games:
        result = game['result']
        white_is_new = game['white_model'] == os.path.basename(new_model_path)
        
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
    
    # Evaluation summary
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)")
    print(f"Games played: {total_games_played}/{num_games}")
    print(f"Failed games: {len(failed_games)}")
    
    print(f"\nResults (from new model's perspective):")
    print(f"  New model wins: {new_model_wins} ({new_model_wins/total_games_played*100:.1f}%)")
    print(f"  Old model wins: {old_model_wins} ({old_model_wins/total_games_played*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/total_games_played*100:.1f}%)")
    print(f"  Decisive games: {decisive_games}/{total_games_played} ({decisive_games/total_games_played*100:.1f}%)")
    
    print(f"\nEvaluation metrics:")
    print(f"  Score-based rate: {new_model_score_rate:.1f}% (wins + 0.5×draws) ← PRIMARY METRIC")
    print(f"  Traditional win rate: {traditional_win_rate:.1f}% (wins among decisive games)")
    print(f"  New model score: {new_model_score:.1f}/{total_games_played} points")
    
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
        'failed_games': failed_games
    }


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Parallel model evaluation for Aldarion Chess Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick evaluation (30 games, 200 sims per move)
  python3 evaluate_models.py --old_model model_v1.pth --new_model model_v2.pth --num_games 30 --num_simulations 200
  
  # Standard evaluation (100 games, 400 sims per move)  
  python3 evaluate_models.py --old_model model_weights.pth --new_model model_weights_v5.pth --num_games 100 --num_simulations 400
  
  # High-quality evaluation (200 games, 800 sims per move)
  python3 evaluate_models.py --old_model current_best.pth --new_model candidate.pth --num_games 200 --num_simulations 800 --cpu_utilization 0.9

Notes:
  - New model win rate >55% typically means the new model is significantly stronger
  - Games alternate who plays White/Black for fair evaluation
  - Uses deterministic play (temperature=0) for consistent evaluation
        """
    )
    
    parser.add_argument('--old_model', type=str, required=True,
                        help='Path to current best model')
    parser.add_argument('--new_model', type=str, required=True,
                        help='Path to newly trained model to evaluate')
    parser.add_argument('--num_games', type=int, default=100,
                        help='Total number of games to play (default: 100)')
    parser.add_argument('--num_simulations', type=int, default=200,
                        help='MCTS simulations per move (default: 200)')
    parser.add_argument('--cpu_utilization', type=float, default=0.8,
                        help='Target CPU utilization 0.0-1.0 (default: 0.8)')
    parser.add_argument('--win_threshold', type=float, default=55.0,
                        help='Win rate threshold for accepting new model (default: 55.0%)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.old_model):
        print(f"Error: Old model file not found: {args.old_model}")
        sys.exit(1)
        
    if not os.path.exists(args.new_model):
        print(f"Error: New model file not found: {args.new_model}")
        sys.exit(1)
    
    if not (0.0 <= args.cpu_utilization <= 1.0):
        print("Error: cpu_utilization must be between 0.0 and 1.0")
        sys.exit(1)
    
    print(f"Evaluating models:")
    print(f"  Old model: {args.old_model}")
    print(f"  New model: {args.new_model}")
    print(f"  Games: {args.num_games}")
    print(f"  Simulations per move: {args.num_simulations}")
    print(f"  Win threshold: {args.win_threshold}%")
    
    try:
        # Run evaluation
        results = evaluate_models_parallel(
            old_model_path=args.old_model,
            new_model_path=args.new_model,
            num_games=args.num_games,
            num_simulations=args.num_simulations,
            cpu_utilization=args.cpu_utilization
        )
        
        if 'error' in results:
            print("❌ Evaluation failed!")
            sys.exit(1)
        
        # Make decision based on score rate
        score_rate = results['score_rate']
        if score_rate >= args.win_threshold:
            print(f"\n🏆 ACCEPT NEW MODEL!")
            print(f"New model score rate ({score_rate:.1f}%) exceeds threshold ({args.win_threshold}%)")
            sys.exit(0)  # Success - accept new model
        else:
            print(f"\n💀 REJECT NEW MODEL!")
            print(f"New model score rate ({score_rate:.1f}%) below threshold ({args.win_threshold}%)")
            sys.exit(1)  # Failure - reject new model
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    main()