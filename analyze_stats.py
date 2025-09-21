#!/usr/bin/env python3
import pickle
import sys
import os

def analyze_stats_file(filepath):
    """Analyze a training stats or evaluation results pickle file and print summary statistics."""
    
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found")
        return
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return
    
    # Detect file type and handle accordingly
    if is_evaluation_file(data):
        analyze_evaluation_results(data, filepath)
        return
    
    # Handle training data format with command_info
    process_data_list = data
    command_info = None
    
    if isinstance(data, dict) and 'process_stats' in data:
        process_data_list = data['process_stats']
        command_info = data.get('command_info')
    
    if not isinstance(process_data_list, list) or len(process_data_list) == 0:
        print("Error: Expected a non-empty list of process data")
        return
    
    total_white_wins = 0
    total_black_wins = 0
    total_draws = 0
    total_games = 0
    total_tasks_completed = 0
    total_training_examples = 0
    total_ending_reasons = {}
    
    print(f"Analyzing: {os.path.basename(filepath)}")
    
    # Display command information if available
    if command_info:
        print(f"\nCommand used:")
        print(f"  {command_info.get('command_line', 'N/A')}")
        print(f"  Timestamp: {command_info.get('timestamp', 'N/A')}")
        if 'arguments' in command_info:
            args = command_info['arguments']
            print(f"  Parameters: games={args.get('total_games')}, sims={args.get('num_simulations')}, temp={args.get('temperature')}, c_puct={args.get('c_puct')}, cpu={args.get('cpu_utilization')}")
    
    print(f"Number of processes: {len(process_data_list)}")
    print("\nPer-process breakdown:")
    
    for process_data in process_data_list:
        try:
            process_id = process_data['process_id']
            tasks_completed = process_data['tasks_completed']
            training_examples = process_data['training_examples']
            game_outcomes = process_data['game_outcomes']
            
            white_wins = game_outcomes['white_wins']
            black_wins = game_outcomes['black_wins'] 
            draws = game_outcomes['draws']
            
            process_total_games = white_wins + black_wins + draws
            
            print(f"Process {process_id:2d}: {tasks_completed} tasks, {process_total_games} games, {training_examples:,} examples (W:{white_wins}, B:{black_wins}, D:{draws})")
            
            # Display game ending reasons for this process
            if 'game_ending_reasons' in process_data:
                ending_reasons = process_data['game_ending_reasons']
                if ending_reasons:
                    print(f"             Game ending reasons:")
                    for reason, count in sorted(ending_reasons.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / process_total_games * 100) if process_total_games > 0 else 0
                        print(f"               {reason}: {count} ({percentage:.1f}%)")
                    
                    # Add to totals
                    for reason, count in ending_reasons.items():
                        total_ending_reasons[reason] = total_ending_reasons.get(reason, 0) + count
            
            total_white_wins += white_wins
            total_black_wins += black_wins
            total_draws += draws
            total_games += process_total_games
            total_tasks_completed += tasks_completed
            total_training_examples += training_examples
            
        except KeyError as e:
            print(f"Warning: Missing key {e} in process data")
            continue
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"{'='*60}")
    print(f"Tasks completed:      {total_tasks_completed:,}")
    print(f"Total games:          {total_games:,}")
    print(f"Training examples:    {total_training_examples:,}")
    print(f"")
    print(f"Game outcomes:")
    print(f"  White wins:         {total_white_wins:,} ({total_white_wins/total_games*100:.1f}%)")
    print(f"  Black wins:         {total_black_wins:,} ({total_black_wins/total_games*100:.1f}%)")
    print(f"  Draws:              {total_draws:,} ({total_draws/total_games*100:.1f}%)")
    print(f"")
    print(f"Ratios:")
    print(f"  Games per task:     {total_games/total_tasks_completed if total_tasks_completed > 0 else 0:.2f}")
    print(f"  Examples per game:  {total_training_examples/total_games if total_games > 0 else 0:.1f}")
    print(f"  Examples per task:  {total_training_examples/total_tasks_completed if total_tasks_completed > 0 else 0:.1f}")
    
    # Display total game ending reasons
    if total_ending_reasons:
        print(f"")
        print(f"Game ending reasons (across all processes):")
        for reason, count in sorted(total_ending_reasons.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_games * 100) if total_games > 0 else 0
            print(f"  {reason}: {count:,} ({percentage:.1f}%)")

def is_evaluation_file(data):
    """Detect if the pickle file contains evaluation results vs training stats."""
    if not isinstance(data, dict):
        return False
    
    # Check for evaluation-specific keys
    evaluation_keys = ['new_model_wins', 'old_model_wins', 'score_rate', 'successful_games']
    return any(key in data for key in evaluation_keys)

def analyze_evaluation_results(data, filepath):
    """Analyze evaluation results pickle file."""
    print(f"Analyzing EVALUATION RESULTS: {os.path.basename(filepath)}")
    print("="*60)
    
    # Show recommendation first
    score_rate = data.get('score_rate', 0)
    threshold = 55.0  # Standard threshold for model acceptance
    if score_rate > threshold:
        print(f"RECOMMENDATION: ACCEPT NEW MODEL!")
        print(f"New model score rate ({score_rate:.1f}%) exceeds threshold ({threshold}%)")
    else:
        print(f"RECOMMENDATION: REJECT NEW MODEL!")
        print(f"New model score rate ({score_rate:.1f}%) below threshold ({threshold}%)")
    print("="*60)
    print()
    
    # Metadata
    if 'metadata' in data:
        meta = data['metadata']
        print(f"Timestamp: {meta.get('timestamp', 'Unknown')}")
        print(f"Old model: {os.path.basename(meta.get('old_model_path', 'Unknown'))}")
        print(f"New model: {os.path.basename(meta.get('new_model_path', 'Unknown'))}")
        print()
    
    # Main results
    total_games = data.get('total_games', 0)
    new_wins = data.get('new_model_wins', 0) 
    old_wins = data.get('old_model_wins', 0)
    draws = data.get('draws', 0)
    decisive_games = data.get('decisive_games', 0)
    score_rate = data.get('score_rate', 0)
    win_rate = data.get('traditional_win_rate', 0)
    execution_time = data.get('execution_time', 0)
    
    print("EVALUATION SUMMARY:")
    print(f"  Total games played:    {total_games}")
    print(f"  New model wins:        {new_wins} ({new_wins/total_games*100:.1f}%)")
    print(f"  Old model wins:        {old_wins} ({old_wins/total_games*100:.1f}%)")
    print(f"  Draws:                 {draws} ({draws/total_games*100:.1f}%)")
    print(f"  Decisive games:        {decisive_games}/{total_games} ({decisive_games/total_games*100:.1f}%)")
    print()
    print(f"EVALUATION METRICS:")
    print(f"  Score-based rate:      {score_rate:.1f}% (wins + 0.5×draws) ← PRIMARY METRIC")
    print(f"  Traditional win rate:  {win_rate:.1f}% (wins among decisive games)")
    print(f"  New model score:       {data.get('new_model_score', 0):.1f}/{total_games} points")
    print()
    print(f"PERFORMANCE:")
    print(f"  Execution time:        {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
    if total_games > 0:
        print(f"  Games per minute:      {total_games/(execution_time/60):.1f}")
    
    
    # Individual games summary
    if 'successful_games' in data:
        games = data['successful_games']
        if games:
            print()
            print(f"GAME DETAILS ({len(games)} games):")
            print("-" * 50)
            
            # Game length statistics
            move_counts = [g.get('move_count', 0) for g in games]
            game_times = [g.get('game_time_seconds', 0) for g in games]
            
            if move_counts:
                avg_moves = sum(move_counts) / len(move_counts)
                min_moves = min(move_counts)
                max_moves = max(move_counts)
                print(f"  Average moves per game: {avg_moves:.1f}")
                print(f"  Move range:             {min_moves} - {max_moves}")
            
            if game_times:
                avg_time = sum(game_times) / len(game_times)
                min_time = min(game_times)
                max_time = max(game_times)
                print(f"  Average game time:      {avg_time:.1f} seconds")
                print(f"  Time range:             {min_time:.1f} - {max_time:.1f} seconds")
            
            # Show first few games as examples
            print()
            print("Sample games:")
            for i, game in enumerate(games[:10]):
                game_id = game.get('game_id', i)
                result = game.get('result', 0)
                white_is_new = game.get('white_is_new', False)
                move_count = game.get('move_count', 0)
                game_time = game.get('game_time_seconds', 0)
                result_str = game.get('result_str', 'Unknown')
                
                # Determine winner from new model's perspective
                if result == 1.0:  # White wins
                    winner = "New" if white_is_new else "Old"
                elif result == -1.0:  # Black wins
                    winner = "New" if not white_is_new else "Old"
                else:  # Draw
                    winner = "Draw"
                
                white_model = "New" if white_is_new else "Old"
                black_model = "Old" if white_is_new else "New"
                
                print(f"  Game {game_id:2d}: {white_model} vs {black_model} → {winner:4s} ({move_count:3d} moves, {game_time:.1f}s)")
            
            if len(games) > 10:
                print(f"  ... and {len(games) - 10} more games")
    
    # Process statistics
    if 'process_statistics' in data:
        stats = data['process_statistics']
        print()
        print(f"PROCESS STATISTICS ({len(stats)} processes):")
        print("-" * 50)
        
        total_proc_games = 0
        total_proc_time = 0
        
        for i, stat in enumerate(stats):
            if 'error' in stat:
                continue
                
            pid = stat.get('process_id', i)
            device = stat.get('gpu_device', 'Unknown')
            completed = stat.get('tasks_completed', 0) 
            requested = stat.get('tasks_requested', 0)
            time_taken = stat.get('total_time_seconds', 0)
            rate = stat.get('tasks_per_minute', 0)
            
            total_proc_games += completed
            total_proc_time += time_taken
            
            print(f"  Process {pid:2d}: {device:8s} - {completed}/{requested} games ({rate:.1f} games/min)")
        
        if total_proc_games > 0:
            print(f"  Overall rate: {total_proc_games/(total_proc_time/60):.1f} games/min across all processes")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_stats.py <stats_file.pkl>")
        print("Examples:")
        print("  python analyze_stats.py training_data_stats/unified_selfplay_data_20250916_132051_stats.pkl")
        print("  python analyze_stats.py evaluation_results/evaluation_model_weights_vs_model_weights_final_20250920_185927.pkl")
        sys.exit(1)
    
    stats_file = sys.argv[1]
    analyze_stats_file(stats_file)

if __name__ == "__main__":
    main()