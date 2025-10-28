#!/usr/bin/env python3
import os
import sys
import argparse
import time
import pickle
from datetime import datetime
import multiprocessing as mp

# Import unified modules
from src.training.parallel_utils import run_parallel_task_execution
from src.training.parallel_workers import selfplay_worker_process


def generate_selfplay_data(total_games, num_simulations, temperature, model_path,c_puct = 2.0, cpu_utilization = 0.90,
                          max_processes_per_gpu= None, output_dir = None, command_info = None, seed = 42):
    """
    Generate self-play training data using parallel processing
    """
    print("="*60)
    print("SELF-PLAY DATA GENERATION")
    print("="*60)
    print(f"Total games: {total_games}")
    print(f"Simulations per move: {num_simulations}")
    print(f"Temperature: {temperature}")
    print(f"C_PUCT: {c_puct}")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"CPU utilization: {cpu_utilization*100:.0f}%")
    
    task_config = {
        'total_tasks': total_games,
        'num_simulations': num_simulations,
        'temperature': temperature,
        'c_puct': c_puct,
        'model_path': model_path,
        'seed': seed
    }
    
    start_time = time.time()
    training_data, process_statistics = run_parallel_task_execution(
        task_config=            task_config,
        worker_function=        selfplay_worker_process,
        cpu_utilization=        cpu_utilization,
        max_processes_per_gpu=  max_processes_per_gpu
    )
    execution_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("SELF-PLAY GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Training examples generated: {len(training_data)}")
    print(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)")
    
    # Save training data
    saved_file = save_training_data(training_data, process_statistics, output_dir, command_info)
    return saved_file


def save_training_data(training_data, process_stats, output_dir = None, command_info = None):
    """
    Save training data and process statistics
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"selfplay_data_{timestamp}.pkl"
    
    if output_dir is None:
        main_data_dir = "training_data"
        stats_data_dir = "training_data_stats"
    else:
        main_data_dir = output_dir
        stats_data_dir = output_dir
    
    os.makedirs(main_data_dir, exist_ok=True)
    os.makedirs(stats_data_dir, exist_ok=True)
    main_data_path = os.path.join(main_data_dir, output_filename)
    base_filename = os.path.splitext(output_filename)[0]
    stats_filename = f"{base_filename}_stats.txt"
    stats_path = os.path.join(stats_data_dir, stats_filename)
    
    # -- Saving Training Data -- #
    with open(main_data_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    # -- Saving Stats information -- #
    with open(stats_path, 'w') as f:
        # Command used section
        if command_info and 'command_line' in command_info:
            f.write("Command used:\n")
            f.write(f"  {command_info['command_line']}\n")
            if 'timestamp' in command_info:
                f.write(f"  Timestamp: {command_info['timestamp']}\n")
            if 'arguments' in command_info:
                args = command_info['arguments']
                f.write(f"  Parameters: games={args.get('total_games')}, sims={args.get('num_simulations')}, ")
                f.write(f"temp={args.get('temperature')}, c_puct={args.get('c_puct')}, cpu={args.get('cpu_utilization')}\n")
        
        f.write(f"Number of processes: {len(process_stats)}\n\n")
        
        # Per-process breakdown
        f.write("Per-process breakdown:\n")
        for i, stats in enumerate(process_stats):
            tasks = stats.get('tasks_completed', 0)
            examples = stats.get('training_examples', 0)
            outcomes = stats.get('game_outcomes', {})
            w = outcomes.get('white_wins', 0)
            b = outcomes.get('black_wins', 0)
            d = outcomes.get('draws', 0)
            
            f.write(f"Process {i:2d}: {tasks} tasks, {tasks} games, {examples:,} examples (W:{w}, B:{b}, D:{d})\n")
            
            # Game ending reasons
            reasons = stats.get('game_ending_reasons', {})
            if reasons:
                f.write(f"             Game ending reasons:\n")
                total_games = sum(reasons.values())
                for reason, count in reasons.items():
                    pct = (count / total_games * 100) if total_games > 0 else 0
                    f.write(f"               {reason}: {count} ({pct:.1f}%)\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("SUMMARY:\n")
        f.write("="*60 + "\n")
        
        total_tasks = sum(s.get('tasks_completed', 0) for s in process_stats)
        total_examples = sum(s.get('training_examples', 0) for s in process_stats)
        
        total_white_wins = 0
        total_black_wins = 0
        total_draws = 0
        all_ending_reasons = {}
        
        for stats in process_stats:
            outcomes = stats.get('game_outcomes', {})
            total_white_wins += outcomes.get('white_wins', 0)
            total_black_wins += outcomes.get('black_wins', 0)
            total_draws += outcomes.get('draws', 0)
            
            reasons = stats.get('game_ending_reasons', {})
            for reason, count in reasons.items():
                all_ending_reasons[reason] = all_ending_reasons.get(reason, 0) + count
        
        total_games = total_white_wins + total_black_wins + total_draws
        
        f.write(f"Tasks completed:      {total_tasks}\n")
        f.write(f"Total games:          {total_games}\n")
        f.write(f"Training examples:    {total_examples:,}\n\n")
        
        f.write("Game outcomes:\n")
        if total_games > 0:
            f.write(f"  White wins:         {total_white_wins} ({total_white_wins/total_games*100:.1f}%)\n")
            f.write(f"  Black wins:         {total_black_wins} ({total_black_wins/total_games*100:.1f}%)\n")
            f.write(f"  Draws:              {total_draws} ({total_draws/total_games*100:.1f}%)\n")
        
        f.write(f"\nRatios:\n")
        if total_tasks > 0:
            f.write(f"  Games per task:     {total_games/total_tasks:.2f}\n")
        if total_games > 0:
            f.write(f"  Examples per game:  {total_examples/total_games:.1f}\n")
        if total_tasks > 0:
            f.write(f"  Examples per task:  {total_examples/total_tasks:.1f}\n")
        
        if all_ending_reasons:
            f.write(f"\nGame ending reasons (across all processes):\n")
            for reason, count in sorted(all_ending_reasons.items(), key=lambda x: -x[1]):
                pct = (count / total_games * 100) if total_games > 0 else 0
                f.write(f"  {reason}: {count} ({pct:.1f}%)\n")
    
    print(f"Training data saved to: {main_data_path}")
    print(f"Process statistics saved to: {stats_path}")
    
    return main_data_path

def main():
    parser = argparse.ArgumentParser(
        description='Self-Play Training Data Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--total_games', type=int, default=100)
    parser.add_argument('--num_simulations', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--c_puct', type=float, default=2.0)
    parser.add_argument('--model_path', type=str, default='model_weights/model_weights.pth')
    parser.add_argument('--cpu_utilization', type=float, default=0.90)
    parser.add_argument('--max_processes_per_gpu', type=int, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Warning: Model file {args.model_path} not found. Will use random weights.")
    
    command_info = {
        'command_line': ' '.join(sys.argv),
        'timestamp': datetime.now().isoformat(),
        'arguments': {
            'total_games': args.total_games,
            'num_simulations': args.num_simulations,
            'temperature': args.temperature,
            'c_puct': args.c_puct,
            'model_path': args.model_path,
            'cpu_utilization': args.cpu_utilization,
            'max_processes_per_gpu': args.max_processes_per_gpu,
            'output': args.output
        }
    }
    
    try:
        saved_file = generate_selfplay_data(
            total_games=args.total_games,
            num_simulations=args.num_simulations,
            temperature=args.temperature,
            c_puct=args.c_puct,
            model_path=args.model_path,
            cpu_utilization=args.cpu_utilization,
            max_processes_per_gpu=args.max_processes_per_gpu,
            output_dir=args.output,
            command_info=command_info,
            seed=args.seed
        )
        
        if saved_file:
            print(f"\nSelf-play data generation successful!")
            print(f"Data saved to: {saved_file}")
            sys.exit(0)
        else:
            print(f"\nSelf-play data generation failed!")
            sys.exit(1)

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()