#!/usr/bin/env python3
import pickle
import sys
import os

def analyze_stats_file(filepath):
    """Analyze a training stats pickle file and print summary statistics."""
    
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found")
        return
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return
    
    # Handle new format with command_info
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
            print(f"  Parameters: games={args.get('total_games')}, sims={args.get('num_simulations')}, temp={args.get('temperature')}, cpu={args.get('cpu_utilization')}")
    
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

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_stats.py <stats_file.pkl>")
        print("Example: python analyze_stats.py training_data_stats/unified_selfplay_data_20250916_132051_stats.pkl")
        sys.exit(1)
    
    stats_file = sys.argv[1]
    analyze_stats_file(stats_file)

if __name__ == "__main__":
    main()