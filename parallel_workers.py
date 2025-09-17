#!/usr/bin/env python3
"""
Unified Worker Functions for Parallel Chess Engine Tasks

This module provides worker functions that can handle both self-play training
and model evaluation using the same underlying infrastructure.
"""

import os
import torch
import time
import traceback
from typing import List, Tuple, Dict, Any

# Import existing modules
import MTCS as mt
import model as md
from parallel_utils import cleanup_gpu_memory, create_process_statistics


def selfplay_worker_process(gpu_device: str, num_games: int, task_config: Dict[str, Any], 
                           process_id: int) -> Tuple[List, Dict]:
    """
    Worker process for self-play training data generation
    
    Args:
        gpu_device: GPU device to use (e.g., 'cuda:0')
        num_games: Number of games for this process to generate
        task_config: Configuration dictionary with task parameters
        process_id: Unique process identifier
    
    Returns:
        Tuple of (training_data, process_stats)
    """
    start_time = time.time()
    
    try:
        num_simulations = task_config['num_simulations']
        temperature = task_config['temperature']
        c_puct = task_config.get('c_puct', 4.0)  # Default to 4.0 if not specified
        model_path = task_config['model_path']
        
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
        game_ending_reasons = []
        
        for game_num in range(num_games):
            try:
                game_start_time = time.time()
                print(f"Process {process_id}: Game {game_num + 1}/{num_games}")
                
                # Use existing run_game function with game tracking info
                training_data = mt.run_game(model, temperature, num_simulations, device, 
                                          c_puct=c_puct, current_game=game_num + 1, total_games=num_games, process_id=process_id)
                all_training_data.extend(training_data)
                games_completed += 1
                
                # Collect game statistics
                game_length = len(training_data)
                game_lengths.append(game_length)
                
                # Extract game outcome from last training example
                if training_data:
                    outcome = training_data[-1][3]  # (board_fen, history_fens, move_probs, outcome)
                    game_outcomes.append(outcome)
                
                # Get the ending reason from the last completed game
                ending_reason = mt.get_last_game_ending_reason()
                if ending_reason:
                    game_ending_reasons.append(ending_reason)
                else:
                    game_ending_reasons.append("Unknown ending reason")
                
                game_time = time.time() - game_start_time
                print(f"Process {process_id}: Game {game_num + 1} completed in {game_time:.1f}s, {len(training_data)} examples")
                
            except Exception as e:
                print(f"Process {process_id}: Error in game {game_num + 1}: {e}")
                # Add error as ending reason
                game_ending_reasons.append(f"Game error: {str(e)}")
                continue
        
        # Calculate detailed statistics
        white_wins = sum(1 for outcome in game_outcomes if outcome > 0)
        black_wins = sum(1 for outcome in game_outcomes if outcome < 0)
        draws = sum(1 for outcome in game_outcomes if outcome == 0)
        
        avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
        min_game_length = min(game_lengths) if game_lengths else 0
        max_game_length = max(game_lengths) if game_lengths else 0
        
        # Count ending reasons
        ending_reason_counts = {}
        for reason in game_ending_reasons:
            ending_reason_counts[reason] = ending_reason_counts.get(reason, 0) + 1
        
        # Create process statistics
        process_stats = create_process_statistics(
            process_id=process_id,
            gpu_device=gpu_device,
            start_time=start_time,
            tasks_completed=games_completed,
            tasks_requested=num_games,
            training_examples=len(all_training_data),
            model_load_time_seconds=model_load_time,
            examples_per_minute=(len(all_training_data) / (time.time() - start_time)) * 60 if time.time() - start_time > 0 else 0,
            game_outcomes={
                'white_wins': white_wins,
                'black_wins': black_wins,
                'draws': draws
            },
            game_length_stats={
                'average': avg_game_length,
                'minimum': min_game_length,
                'maximum': max_game_length
            },
            game_ending_reasons=ending_reason_counts,
            simulations_per_move=num_simulations,
            temperature=temperature
        )
        
        print(f"Process {process_id}: Completed {games_completed}/{num_games} games")
        print(f"Process {process_id}: {len(all_training_data)} examples, {games_completed/(time.time()-start_time)*60:.1f} games/min")
        
        # Explicit GPU memory cleanup
        cleanup_gpu_memory(device, process_id, [model])
        
        return all_training_data, process_stats
        
    except Exception as e:
        print(f"Process {process_id}: Fatal error: {e}")
        traceback.print_exc()
        
        # Cleanup on error
        if 'model' in locals():
            cleanup_gpu_memory(locals().get('device', torch.device('cpu')), process_id, [locals()['model']])
        
        return [], create_process_statistics(
            process_id=process_id, 
            gpu_device=gpu_device,
            start_time=start_time,
            tasks_completed=0,
            tasks_requested=num_games,
            error=str(e)
        )


def evaluation_worker_process(gpu_device: str, num_games: int, task_config: Dict[str, Any], 
                            process_id: int) -> Tuple[List, Dict]:
    """
    Worker process for model evaluation games
    
    Args:
        gpu_device: GPU device to use (e.g., 'cuda:0')
        num_games: Number of games for this process to play
        task_config: Configuration dictionary with task parameters
        process_id: Unique process identifier
    
    Returns:
        Tuple of (game_results, process_stats)
    """
    start_time = time.time()
    
    try:
        num_simulations = task_config['num_simulations']
        old_model_path = task_config['old_model_path']
        new_model_path = task_config['new_model_path']
        starting_game_id = task_config.get('starting_game_id', process_id * num_games)
        
        print(f"Process {process_id}: Starting on {gpu_device} with {num_games} evaluation games")
        
        # Setup device
        device = torch.device(gpu_device)
        
        # Load both models (each process gets its own copies)
        model_load_start = time.time()
        old_model = md.ChessNet()
        new_model = md.ChessNet()
        
        old_model.to(device)
        new_model.to(device)
        
        # Load weights
        old_state = torch.load(old_model_path, map_location=device, weights_only=True)
        new_state = torch.load(new_model_path, map_location=device, weights_only=True)
        
        old_model.load_state_dict(old_state)
        new_model.load_state_dict(new_state)
        
        old_model.eval()
        new_model.eval()
        
        model_load_time = time.time() - model_load_start
        print(f"Process {process_id}: Both models loaded successfully on {gpu_device}")
        
        # Play evaluation games
        game_results = []
        games_completed = 0
        game_times = []
        move_counts = []
        new_model_wins = 0
        old_model_wins = 0
        draws = 0
        game_ending_reasons = []
        
        for game_num in range(num_games):
            try:
                game_id = starting_game_id + game_num
                game_start_time = time.time()
                
                print(f"Process {process_id}: Evaluation game {game_num + 1}/{num_games} (ID: {game_id})")
                
                # Alternate who plays white/black for fairness
                if game_id % 2 == 0:
                    # New model plays White
                    white_model = new_model
                    black_model = old_model
                    white_is_new = True
                else:
                    # Old model plays White  
                    white_model = old_model
                    black_model = new_model
                    white_is_new = False
                
                # Play the competitive game
                game_result = play_single_evaluation_game(
                    white_model=white_model,
                    black_model=black_model,
                    num_simulations=num_simulations,
                    device=device,
                    game_id=game_id,
                    white_is_new=white_is_new,
                    old_model_path=old_model_path,
                    new_model_path=new_model_path
                )
                
                game_results.append(game_result)
                games_completed += 1
                
                # Collect statistics
                if 'error' not in game_result:
                    game_times.append(game_result['game_time_seconds'])
                    move_counts.append(game_result['move_count'])
                    
                    # Collect ending reason
                    if 'result_str' in game_result:
                        game_ending_reasons.append(game_result['result_str'])
                    else:
                        game_ending_reasons.append("Unknown ending reason")
                    
                    # Count wins from new model's perspective
                    result = game_result['result']
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
                
                game_time = time.time() - game_start_time
                print(f"Process {process_id}: Game {game_num + 1} completed in {game_time:.1f}s")
                
            except Exception as e:
                print(f"Process {process_id}: Error in evaluation game {game_num + 1}: {e}")
                # Add error as ending reason
                game_ending_reasons.append(f"Game error: {str(e)}")
                # Add error result
                game_results.append({
                    'game_id': starting_game_id + game_num,
                    'error': str(e),
                    'game_time_seconds': time.time() - game_start_time
                })
                continue
        
        # Create process statistics
        avg_game_time = sum(game_times) / len(game_times) if game_times else 0
        avg_moves = sum(move_counts) / len(move_counts) if move_counts else 0
        
        # Count ending reasons
        ending_reason_counts = {}
        for reason in game_ending_reasons:
            ending_reason_counts[reason] = ending_reason_counts.get(reason, 0) + 1
        
        process_stats = create_process_statistics(
            process_id=process_id,
            gpu_device=gpu_device,
            start_time=start_time,
            tasks_completed=games_completed,
            tasks_requested=num_games,
            model_load_time_seconds=model_load_time,
            new_model_wins=new_model_wins,
            old_model_wins=old_model_wins,
            draws=draws,
            successful_games=len([r for r in game_results if 'error' not in r]),
            failed_games=len([r for r in game_results if 'error' in r]),
            avg_game_time_seconds=avg_game_time,
            avg_moves_per_game=avg_moves,
            game_ending_reasons=ending_reason_counts,
            simulations_per_move=num_simulations
        )
        
        print(f"Process {process_id}: Completed {games_completed}/{num_games} evaluation games")
        print(f"Process {process_id}: Results - New: {new_model_wins}, Old: {old_model_wins}, Draws: {draws}")
        
        # Explicit GPU memory cleanup
        cleanup_gpu_memory(device, process_id, [old_model, new_model])
        
        return game_results, process_stats
        
    except Exception as e:
        print(f"Process {process_id}: Fatal error: {e}")
        traceback.print_exc()
        
        # Cleanup on error
        models_to_cleanup = []
        if 'old_model' in locals():
            models_to_cleanup.append(locals()['old_model'])
        if 'new_model' in locals():
            models_to_cleanup.append(locals()['new_model'])
        
        if models_to_cleanup:
            cleanup_gpu_memory(locals().get('device', torch.device('cpu')), process_id, models_to_cleanup)
        
        return [], create_process_statistics(
            process_id=process_id, 
            gpu_device=gpu_device,
            start_time=start_time,
            tasks_completed=0,
            tasks_requested=num_games,
            error=str(e)
        )


def play_single_evaluation_game(white_model, black_model, num_simulations: int, device: torch.device,
                               game_id: int, white_is_new: bool, old_model_path: str, new_model_path: str) -> Dict:
    """
    Play a single competitive game between two models
    
    Args:
        white_model: Model playing white
        black_model: Model playing black
        num_simulations: MCTS simulations per move
        device: PyTorch device
        game_id: Unique game identifier
        white_is_new: Whether white model is the new model
        old_model_path: Path to old model (for result tracking)
        new_model_path: Path to new model (for result tracking)
    
    Returns:
        Dictionary with game results
    """
    import chess
    
    start_time = time.time()
    
    try:
        # Play the game
        board = chess.Board()
        game_history = []
        move_count = 0
        
        while not board.is_game_over() and move_count < 800:  # Early stopping
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
        elif move_count >= 800:
            result = 0.0
            result_str = "Draw (800-ply limit)"
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
            'white_model': os.path.basename(new_model_path) if white_is_new else os.path.basename(old_model_path),
            'black_model': os.path.basename(old_model_path) if white_is_new else os.path.basename(new_model_path),
            'white_is_new': white_is_new
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