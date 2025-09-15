#!/usr/bin/env python3
"""
Model Evaluation Script for Aldarion Chess Engine

This script pits two chess models against each other to determine which is stronger.
Used in the training loop to decide whether to accept a new model or keep the previous best.
"""

import os
import sys
import torch
import chess
import argparse
import time
from datetime import datetime
from typing import Tuple, Dict

# Import existing modules
import model as md
import MTCS as mt


def play_game(white_model, black_model, num_simulations, device):
    """
    Play one game between two models
    
    Args:
        white_model: Model playing white pieces
        black_model: Model playing black pieces
        num_simulations: MCTS simulations per move
        device: PyTorch device
    
    Returns:
        tuple: (game_result, move_count, game_pgn)
        game_result: 1.0 (white wins), 0.0 (draw), -1.0 (black wins)
    """
    board = chess.Board()
    game_history = []
    move_count = 0
    
    while not board.is_game_over():
        # Select model based on whose turn it is
        current_model = white_model if board.turn else black_model
        
        try:
            # Get best move from model
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
            if move_obj in board.legal_moves:
                board.push(move_obj)
                game_history.append(board.copy())
                move_count += 1
            else:
                print(f"Illegal move attempted: {best_move}")
                break
                
        except Exception as e:
            print(f"Error during move generation: {e}")
            break
    
    # Determine game result
    if board.is_checkmate():
        result = -1.0 if board.turn else 1.0  # If White's turn -> White checkmated -> Black wins (-1), vice versa
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        result = 0.0  # Draw
    else:
        result = 0.0  # Unexpected end condition - draw
    
    return result, move_count, str(board)


def evaluate_models(old_model_path, new_model_path, num_games=100, num_simulations=200, device=None):
    """
    Evaluate two models by playing them against each other
    
    Args:
        old_model_path: Path to the current best model
        new_model_path: Path to the new candidate model
        num_games: Number of games to play (should be even)
        num_simulations: MCTS simulations per move
        device: PyTorch device
    
    Returns:
        dict: Evaluation results with win rates and statistics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading models...")
    print(f"Old model: {old_model_path}")
    print(f"New model: {new_model_path}")
    
    # Load models
    old_model = md.ChessNet()
    new_model = md.ChessNet()
    
    old_model.to(device)
    new_model.to(device)
    
    if os.path.exists(old_model_path):
        old_state = torch.load(old_model_path, map_location=device, weights_only=True)
        old_model.load_state_dict(old_state)
    else:
        print(f"Warning: Old model {old_model_path} not found. Using random weights.")
    
    if os.path.exists(new_model_path):
        new_state = torch.load(new_model_path, map_location=device, weights_only=True)
        new_model.load_state_dict(new_state)
    else:
        print(f"Error: New model {new_model_path} not found!")
        return None
    
    old_model.eval()
    new_model.eval()
    
    print(f"Starting evaluation: {num_games} games, {num_simulations} simulations per move")
    print(f"Device: {device}")
    
    # Results tracking
    results = {
        'new_as_white_wins': 0,
        'new_as_white_draws': 0,
        'new_as_white_losses': 0,
        'new_as_black_wins': 0,
        'new_as_black_draws': 0,
        'new_as_black_losses': 0,
        'total_games': 0,
        'game_details': []
    }
    
    games_per_side = num_games // 2
    
    # Play games with new model as white
    print(f"\nPlaying {games_per_side} games with new model as White...")
    for game_num in range(games_per_side):
        print(f"Game {game_num + 1}/{games_per_side} (New as White)", end=" ")
        
        start_time = time.time()
        game_result, move_count, final_pos = play_game(
            white_model=new_model,
            black_model=old_model, 
            num_simulations=num_simulations,
            device=device
        )
        game_time = time.time() - start_time
        
        if game_result > 0:  # New model (white) wins
            results['new_as_white_wins'] += 1
            print(f"- NEW WINS in {move_count} moves ({game_time:.1f}s)")
        elif game_result < 0:  # Old model (black) wins
            results['new_as_white_losses'] += 1
            print(f"- old wins in {move_count} moves ({game_time:.1f}s)")
        else:  # Draw
            results['new_as_white_draws'] += 1
            print(f"- draw in {move_count} moves ({game_time:.1f}s)")
        
        results['game_details'].append({
            'game_num': game_num + 1,
            'new_color': 'white',
            'result': game_result,
            'moves': move_count,
            'time_seconds': game_time
        })
        results['total_games'] += 1
    
    # Play games with new model as black
    print(f"\nPlaying {games_per_side} games with new model as Black...")
    for game_num in range(games_per_side):
        print(f"Game {game_num + 1}/{games_per_side} (New as Black)", end=" ")
        
        start_time = time.time()
        game_result, move_count, final_pos = play_game(
            white_model=old_model,
            black_model=new_model,
            num_simulations=num_simulations,
            device=device
        )
        game_time = time.time() - start_time
        
        if game_result < 0:  # New model (black) wins
            results['new_as_black_wins'] += 1
            print(f"- NEW WINS in {move_count} moves ({game_time:.1f}s)")
        elif game_result > 0:  # Old model (white) wins  
            results['new_as_black_losses'] += 1
            print(f"- old wins in {move_count} moves ({game_time:.1f}s)")
        else:  # Draw
            results['new_as_black_draws'] += 1
            print(f"- draw in {move_count} moves ({game_time:.1f}s)")
        
        results['game_details'].append({
            'game_num': game_num + games_per_side + 1,
            'new_color': 'black', 
            'result': -game_result,  # Flip perspective for black
            'moves': move_count,
            'time_seconds': game_time
        })
        results['total_games'] += 1
    
    # Calculate final statistics
    total_new_wins = results['new_as_white_wins'] + results['new_as_black_wins']
    total_draws = results['new_as_white_draws'] + results['new_as_black_draws']
    total_new_losses = results['new_as_white_losses'] + results['new_as_black_losses']
    
    new_win_rate = (total_new_wins / results['total_games']) * 100
    draw_rate = (total_draws / results['total_games']) * 100
    new_score = (total_new_wins + 0.5 * total_draws) / results['total_games'] * 100
    
    results.update({
        'total_new_wins': total_new_wins,
        'total_draws': total_draws,
        'total_new_losses': total_new_losses,
        'new_win_rate': new_win_rate,
        'draw_rate': draw_rate,
        'new_score': new_score,
        'accept_new_model': new_score > 55.0  # Standard threshold
    })
    
    return results


def print_evaluation_results(results):
    """Print detailed evaluation results"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"Total games played: {results['total_games']}")
    print(f"New model wins: {results['total_new_wins']}")
    print(f"Draws: {results['total_draws']}")
    print(f"New model losses: {results['total_new_losses']}")
    
    print(f"\nDetailed breakdown:")
    print(f"As White: {results['new_as_white_wins']}W-{results['new_as_white_draws']}D-{results['new_as_white_losses']}L")
    print(f"As Black: {results['new_as_black_wins']}W-{results['new_as_black_draws']}D-{results['new_as_black_losses']}L")
    
    print(f"\nPerformance metrics:")
    print(f"Win rate: {results['new_win_rate']:.1f}%")
    print(f"Draw rate: {results['draw_rate']:.1f}%") 
    print(f"Score (W + 0.5*D): {results['new_score']:.1f}%")
    
    print(f"\nDecision: {'✅ ACCEPT NEW MODEL' if results['accept_new_model'] else '❌ REJECT NEW MODEL'}")
    print(f"Threshold: 55.0% (new model scored {results['new_score']:.1f}%)")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description='Evaluate two chess models by playing them against each other',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python3 evaluate_models.py --old_model model_v1.pth --new_model model_v2.pth
  
  # Quick evaluation with fewer games
  python3 evaluate_models.py --old_model old.pth --new_model new.pth --num_games 20
  
  # High-quality evaluation
  python3 evaluate_models.py --old_model best.pth --new_model candidate.pth --num_games 200 --num_simulations 400
        """
    )
    
    parser.add_argument('--old_model', type=str, required=True,
                        help='Path to current best model')
    parser.add_argument('--new_model', type=str, required=True,
                        help='Path to new candidate model')
    parser.add_argument('--num_games', type=int, default=50,
                        help='Number of games to play (default: 50, should be even)')
    parser.add_argument('--num_simulations', type=int, default=200,
                        help='MCTS simulations per move (default: 200)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to file')
    
    args = parser.parse_args()
    
    # Ensure even number of games
    if args.num_games % 2 != 0:
        args.num_games += 1
        print(f"Adjusted to {args.num_games} games for equal white/black distribution")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run evaluation
    try:
        start_time = time.time()
        results = evaluate_models(
            old_model_path=args.old_model,
            new_model_path=args.new_model, 
            num_games=args.num_games,
            num_simulations=args.num_simulations,
            device=device
        )
        
        if results is None:
            print("Evaluation failed!")
            sys.exit(1)
        
        evaluation_time = time.time() - start_time
        results['evaluation_time_minutes'] = evaluation_time / 60
        
        # Print results
        print_evaluation_results(results)
        print(f"\nTotal evaluation time: {evaluation_time/60:.1f} minutes")
        
        # Save results if requested
        if args.output:
            import pickle
            with open(args.output, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved to {args.output}")
        
        # Return appropriate exit code
        sys.exit(0 if results['accept_new_model'] else 1)
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()