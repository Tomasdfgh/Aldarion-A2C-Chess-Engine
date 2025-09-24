#!/usr/bin/env python3
"""
Worker Functions for Parallel Chess Engine Tasks

This module provides worker functions that can handle both self-play training
and model evaluation using the same underlying infrastructure.
"""

import os
import torch
import time
import chess
import random

# Import existing modules
import MTCS as mt
import model as md
from parallel_utils import cleanup_gpu_memory, create_process_statistics


def selfplay_worker_process(gpu_device, num_games, task_config, process_id):
    """
    Worker process for self-play training data generation
    """
    start_time = time.time()
    
    try:
        num_simulations = task_config['num_simulations']
        temperature = task_config.get('temperature', 1.0)
        c_puct = task_config.get('c_puct', 2.0)
        model_path = task_config['model_path']
        
        print(f"Process {process_id}: Starting on {gpu_device} with {num_games} games")
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
                

                training_data, ending_reason = mt.run_game(model, num_simulations, device, temperature=temperature, 
                                          c_puct=c_puct, current_game=game_num + 1, total_games=num_games, process_id=process_id)
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
        num_simulations = task_config['num_simulations']
        old_model_path = task_config['old_model_path']
        new_model_path = task_config['new_model_path']
        starting_game_id = process_id * num_games
        
        print(f"Process {process_id}: Starting on {gpu_device} with {num_games} evaluation games")
        
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
                game_result = play_single_evaluation_game(
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
                    total_games=num_games
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
                
                game_time = time.time() - game_start_time
                print(f"Process {process_id}: Game {game_num + 1} completed in {game_time:.1f}s")
                
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


def play_single_evaluation_game(white_model, black_model, num_simulations, device, game_id, white_is_new,
                                old_model_path, new_model_path, process_id, game_num, total_games):
    """
    Play a single competitive game between two models with private MCTS trees
    """
    start_time = time.time()
    
    try:
        board = chess.Board.from_chess960_pos(random.randint(0, 959))
        game_history = []
        move_count = 0
        
        print(f"Process {process_id}: Game {game_num}/{total_games}: Starting FEN: {board.fen()}")
        print('\n')
        
        white_tree = None
        black_tree = None
        
        while not board.is_game_over() and move_count < 800:
            current_model = white_model if board.turn else black_model
            current_tree = white_tree if board.turn else black_tree
            current_player = "White" if board.turn else "Black"
            print(f"Move {move_count + 1}, {current_player} to move")
            
            try:
                # Get best move using private tree (with subtree reuse)
                best_move, selected_child = return_move_and_child(
                    model=current_model,
                    board_fen=board.fen(),
                    num_simulations=num_simulations,
                    device=device,
                    game_history=game_history,
                    existing_tree=current_tree,
                    temperature=0.0  # Deterministic play for evaluation
                )
                
                if best_move is None:
                    print(f"Process {process_id}: Game {game_num}/{total_games}: No legal moves available")
                    break
                
                model_info = "New" if (board.turn and white_is_new) or (not board.turn and not white_is_new) else "Old"
                print(f"Selected move: {best_move} ({model_info} model), [Process {process_id}] - Game {game_num}/{total_games}")
                print()
                
                # Update the tree for the current player
                if board.turn:
                    white_tree = selected_child
                else:
                    black_tree = selected_child

                move_obj = chess.Move.from_uci(best_move)
                board.push(move_obj)
                game_history.append(board.copy())
                move_count += 1
                
            except Exception as e:
                print(f"Process {process_id}: Game {game_num}/{total_games}: Error during {current_player} move: {e}")
                break
        
        # Determine game result
        if board.is_checkmate():
            result = -1.0 if board.turn else 1.0
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
            'result': result,
            'result_str': result_str,
            'move_count': move_count,
            'game_time_seconds': game_time,
            'white_model': os.path.basename(new_model_path) if white_is_new else os.path.basename(old_model_path),
            'black_model': os.path.basename(old_model_path) if white_is_new else os.path.basename(new_model_path),
            'white_is_new': white_is_new
        }
        
        print(f"Process {process_id}: Game {game_num}/{total_games}: {result_str} in {move_count} moves ({game_time:.1f}s)")
        return game_result
        
    except Exception as e:
        print(f"Process {process_id}: Game {game_num}/{total_games}: Fatal error: {e}")
        return {
            'game_id': game_id,
            'error': str(e),
            'game_time_seconds': time.time() - start_time
        }
    
def return_move_and_child(model, board_fen, num_simulations, device,
                                 game_history=None, existing_tree=None, temperature=0.0, c_puct=2.0):
    """
    Get the best move for a given position using MCTS with optional tree reuse
    """
    
    board = chess.Board(board_fen)
    if existing_tree is not None and existing_tree.state == board_fen:
        root = existing_tree
        print(f"Reusing MCTS tree (N={root.N}, children={len(root.children)})")
    else:
        root = None
        
        if existing_tree is not None:
            for child in existing_tree.children:
                if child.state == board_fen:
                    child.parent = None
                    root = child
                    print(f"Promoting child to root (N={root.N}, children={len(root.children)})")
                    break
        
        if root is None:
            root = mt.MTCSNode(
                team=board.turn,
                state=board_fen,
                action=None,
                n=0,
                w=0.0,
                q=0.0,
                p=1.0
            )
            print("Created fresh MCTS root")
    
    root = mt.mcts_search(root, model, num_simulations, device, game_history, add_root_noise=False, c_puct=c_puct)

    if temperature == 0.0:
        best_child = max(root.children, key=lambda x: x.N)
        best_move = best_child.action
        selected_child = best_child
    else:
        #Should never be this btw, eval games should be deterministic
        best_move, selected_child = mt.select_move(root, temperature)
    
    if selected_child is not None:
        selected_child.parent = None
    
    return best_move, selected_child

