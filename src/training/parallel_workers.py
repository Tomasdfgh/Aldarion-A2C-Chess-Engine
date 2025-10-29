#!/usr/bin/env python3
"""
Worker Functions for Parallel Chess Engine Tasks

This module provides worker functions that can handle both self-play training
and model evaluation using the same underlying infrastructure.
"""

import os
import torch
import time
import random
import numpy as np

# Import existing modules
from src.agent import mcts as mt
from src.agent import model as md
from src.training.parallel_utils import cleanup_gpu_memory, create_process_statistics


def set_worker_seed(process_id, base_seed=42):
    """
    Set seeds for reproducibility in worker processes
    Each process gets a unique seed based on process_id
    """
    worker_seed = base_seed + process_id
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    # For deterministic behavior in PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return worker_seed


def selfplay_worker_process(gpu_device, num_games, task_config, process_id):
    """
    Worker process for self-play training data generation
    """
    start_time = time.time()
    
    try:
        # Set deterministic seed for this worker process
        base_seed = task_config.get('seed', 42)
        worker_seed = set_worker_seed(process_id, base_seed)
        
        num_simulations = task_config['num_simulations']
        temperature = task_config.get('temperature', 1.0)
        c_puct = task_config.get('c_puct', 2.0)
        max_game_length = task_config.get('max_game_length', 800)
        model_path = task_config['model_path']
        
        print(f"Process {process_id}: Starting on {gpu_device} with {num_games} games (seed: {worker_seed})")
        device = torch.device(gpu_device)
        
        # Load model (each process gets its own copy)
        model = md.ChessNet()
        model.to(device)
        
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            print(f"Process {process_id}: Model loaded successfully on {gpu_device}")
        else:
            print(f"Process {process_id}: Warning - Model file {model_path} not found. Using random weights.")
        

        all_training_data = []
        games_completed = 0
        game_lengths = []
        game_outcomes = []
        game_ending_reasons = []
        
        for game_num in range(num_games):
            try:
                game_start_time = time.time()
                print(f"Process {process_id}: Game {game_num + 1}/{num_games}")

                training_data, ending_reason = mt.run_game(model, num_simulations, device, temperature=temperature, c_puct=c_puct, max_game_length=max_game_length, current_game=game_num + 1, total_games=num_games, process_id=process_id)
                all_training_data.extend(training_data)
                
                game_length = len(training_data)
                game_lengths.append(game_length)
                if training_data:
                    outcome = training_data[-1][3]
                    game_outcomes.append(outcome)
                game_ending_reasons.append(ending_reason)
                games_completed += 1
                
                game_time = time.time() - game_start_time
                print(f"Process {process_id}: Game {game_num + 1} completed in {game_time:.1f}s, {len(training_data)} examples")
                
            except Exception as e:
                print(f"Process {process_id}: Error in game {game_num + 1}: {e}")
                game_ending_reasons.append(f"Game error: {str(e)}")
                continue
        
        white_wins = sum(1 for outcome in game_outcomes if outcome < 0)
        black_wins = sum(1 for outcome in game_outcomes if outcome > 0)
        draws = sum(1 for outcome in game_outcomes if outcome == 0)
        
        avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
        min_game_length = min(game_lengths) if game_lengths else 0
        max_game_length = max(game_lengths) if game_lengths else 0
        
        ending_reason_counts = {}
        for reason in game_ending_reasons:
            ending_reason_counts[reason] = ending_reason_counts.get(reason, 0) + 1

        process_stats = create_process_statistics(
            process_id=process_id,
            gpu_device=gpu_device,
            start_time=start_time,
            tasks_completed=games_completed,
            tasks_requested=num_games,
            training_examples=len(all_training_data),
            examples_per_minute=(len(all_training_data) / (time.time() - start_time)) * 60 if time.time() - start_time > 0 else 0,
            game_outcomes={'white_wins': white_wins, 'black_wins': black_wins, 'draws': draws},
            game_length_stats={'average': avg_game_length, 'minimum': min_game_length, 'maximum': max_game_length },
            game_ending_reasons=ending_reason_counts,
            simulations_per_move=num_simulations,
            temperature=temperature
        )
        
        print(f"Process {process_id}: Completed {games_completed}/{num_games} games")
        print(f"Process {process_id}: {len(all_training_data)} examples, {games_completed/(time.time()-start_time)*60:.1f} games/min")
        cleanup_gpu_memory(device, process_id, [model])
        
        return all_training_data, process_stats
        
    except Exception as e:
        print(f"Process {process_id}: Fatal error: {e}")
        
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

def evaluation_worker_process(gpu_device: str, num_games: int, task_config, 
                            process_id: int):
    """
    Worker process for model evaluation games
    """
    start_time = time.time()
    
    try:
        # Set deterministic seed for this worker process
        base_seed = task_config.get('seed', 42)
        worker_seed = set_worker_seed(process_id, base_seed)
        
        num_simulations = task_config['num_simulations']
        max_game_length = task_config.get('max_game_length', 800)
        old_model_path = task_config['old_model_path']
        new_model_path = task_config['new_model_path']
        starting_game_id = process_id * num_games
        
        print(f"Process {process_id}: Starting on {gpu_device} with {num_games} evaluation games (seed: {worker_seed})")
        
        device = torch.device(gpu_device)
        old_model = md.ChessNet()
        new_model = md.ChessNet()
        old_model.to(device)
        new_model.to(device)
        
        old_model.load_state_dict(torch.load(old_model_path, map_location=device, weights_only=True))
        old_model.eval()

        new_model.load_state_dict(torch.load(new_model_path, map_location=device, weights_only=True))
        new_model.eval()
        
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
                game_result = mt.play_single_evaluation_game(
                    white_model=white_model,
                    black_model=black_model,
                    num_simulations=num_simulations,
                    device=device,
                    game_id=game_id,
                    white_is_new=white_is_new,
                    old_model_path=old_model_path,
                    new_model_path=new_model_path,
                    process_id=process_id,
                    game_num=game_num + 1,
                    total_games=num_games,
                    max_game_length=max_game_length
                )
                
                game_results.append(game_result)
                games_completed += 1
                
                # Collect statistics
                if 'error' not in game_result:
                    game_times.append(game_result['game_time_seconds'])
                    move_counts.append(game_result['move_count'])
                    
                    if 'result_str' in game_result:
                        game_ending_reasons.append(game_result['result_str'])
                    else:
                        game_ending_reasons.append("Unknown ending reason")
                    
                    result = game_result['result']
                    if result == 1.0:  # White wins
                        if white_is_new:
                            new_model_wins += 1
                        else:
                            old_model_wins += 1
                    elif result == -1.0:  # Black wins
                        if not white_is_new:
                            new_model_wins += 1
                        else:
                            old_model_wins += 1
                    else:
                        draws += 1
                
            except Exception as e:
                print(f"Process {process_id}: Error in evaluation game {game_num + 1}: {e}")
                game_ending_reasons.append(f"Game error: {str(e)}")
                game_results.append({
                    'game_id': starting_game_id + game_num,
                    'error': str(e),
                    'game_time_seconds': time.time() - game_start_time
                })
                continue
        
        avg_game_time = sum(game_times) / len(game_times) if game_times else 0
        avg_moves = sum(move_counts) / len(move_counts) if move_counts else 0
        ending_reason_counts = {}
        for reason in game_ending_reasons:
            ending_reason_counts[reason] = ending_reason_counts.get(reason, 0) + 1
        
        process_stats = create_process_statistics(
            process_id=process_id,
            gpu_device=gpu_device,
            start_time=start_time,
            tasks_completed=games_completed,
            tasks_requested=num_games,
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
        cleanup_gpu_memory(device, process_id, [old_model, new_model])
        
        return game_results, process_stats
        
    except Exception as e:
        print(f"Process {process_id}: Fatal error: {e}")
        
        return [], create_process_statistics(
            process_id=process_id, 
            gpu_device=gpu_device,
            start_time=start_time,
            tasks_completed=0,
            tasks_requested=num_games,
            error=str(e)
        )