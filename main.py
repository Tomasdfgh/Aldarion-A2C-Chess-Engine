#!/usr/bin/env python3
"""
Main Training Pipeline for Aldarion Chess Engine

This script runs a complete training iteration:
1. Analyzes previous iteration results
2. Sets up new iteration folder
3. Runs self-play data generation
4. Trains new model on the data  
5. Evaluates new model vs current best

Each iteration builds on the previous one's results.
"""

import os
import sys
import shutil
import re
import argparse
from pathlib import Path

# Import the main functions from existing scripts
from selfplay_generate_data import generate_selfplay_data
from evaluate_models import evaluate_models
import train_model


def find_latest_iteration():
    """
    Find the highest numbered iteration folder
    
    Returns:
        int: Latest iteration number, or raises error if none found
    """
    print("="*60)
    print("SCANNING FOR PREVIOUS ITERATIONS")
    print("="*60)
    
    iterations_dir = Path("Iterations")
    if not iterations_dir.exists():
        print("Error: Iterations/ directory not found!")
        sys.exit(1)
    
    # Find all iteration folders
    iteration_folders = []
    for folder in iterations_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("Iteration_"):
            try:
                num = int(folder.name.split("_")[1])
                iteration_folders.append(num)
                print(f"Found: {folder.name}")
            except (IndexError, ValueError):
                print(f"Skipping invalid folder: {folder.name}")
    
    if not iteration_folders:
        print("Error: No previous iteration folders found!")
        print("Expected at least one folder like 'Iteration_0'")
        sys.exit(1)
    
    latest_iteration = max(iteration_folders)
    print(f"Latest iteration: Iteration_{latest_iteration}")
    return latest_iteration


def analyze_previous_evaluation(prev_iteration_num):
    """
    Analyze the evaluation results from the previous iteration
    
    Args:
        prev_iteration_num: Previous iteration number
        
    Returns:
        bool: True if new model should be accepted, False otherwise
    """
    print("\n" + "="*60) 
    print("ANALYZING PREVIOUS ITERATION RESULTS")
    print("="*60)
    
    prev_folder = Path(f"Iterations/Iteration_{prev_iteration_num}")
    print(f"Checking folder: {prev_folder}")
    
    # Find evaluation file
    eval_files = list(prev_folder.glob("evaluation_*.pkl"))
    if not eval_files:
        print("Error: No evaluation file found in previous iteration!")
        sys.exit(1)
    
    if len(eval_files) > 1:
        print("Warning: Multiple evaluation files found. Using the first one.")
    
    eval_file = eval_files[0]
    print(f"Found evaluation file: {eval_file.name}")
    
    # Run analyze_stats.py to get the recommendation
    print("Running analysis...")
    import subprocess
    result = subprocess.run([
        "python3", "analyze_stats.py", str(eval_file)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running analyze_stats.py:")
        print(result.stderr)
        sys.exit(1)
    
    # Parse the output for recommendation
    output = result.stdout
    print("Analysis output:")
    print("-" * 40)
    # Print first 10 lines to show the recommendation
    lines = output.split('\n')
    for line in lines[:15]:
        print(line)
    print("-" * 40)
    
    # Check for acceptance
    if "ACCEPT NEW MODEL!" in output:
        print("RESULT: New model was ACCEPTED in previous iteration")
        return True
    elif "REJECT NEW MODEL!" in output:
        print("RESULT: New model was REJECTED in previous iteration") 
        return False
    else:
        print("Error: Could not determine evaluation result!")
        print("Expected to find 'ACCEPT NEW MODEL!' or 'REJECT NEW MODEL!'")
        sys.exit(1)


def setup_new_iteration(prev_iteration_num, accept_new_model):
    """
    Create new iteration folder and copy appropriate model
    
    Args:
        prev_iteration_num: Previous iteration number
        accept_new_model: Whether to use new model as base
        
    Returns:
        int: New iteration number
    """
    new_iteration_num = prev_iteration_num + 1
    
    print("\n" + "="*60)
    print("SETTING UP NEW ITERATION")
    print("="*60)
    
    prev_folder = Path(f"Iterations/Iteration_{prev_iteration_num}")
    new_folder = Path(f"Iterations/Iteration_{new_iteration_num}")
    
    # Create new iteration folder
    print(f"Creating new folder: {new_folder}")
    new_folder.mkdir(exist_ok=True)
    
    # Determine which model to copy as the base
    if accept_new_model:
        source_model = prev_folder / f"new_model_{prev_iteration_num}.pth"
        print(f"Copying NEW model as base: {source_model.name}")
    else:
        source_model = prev_folder / "model_weights.pth" 
        print(f"Keeping OLD model as base: {source_model.name}")
    
    # Copy the selected model as the new base
    target_model = new_folder / "model_weights.pth"
    
    if not source_model.exists():
        print(f"Error: Source model not found: {source_model}")
        sys.exit(1)
    
    print(f"Copying: {source_model} → {target_model}")
    shutil.copy2(source_model, target_model)
    
    # Verify the copy
    if target_model.exists():
        source_size = source_model.stat().st_size
        target_size = target_model.stat().st_size
        print(f"Copy successful! Size: {target_size:,} bytes (original: {source_size:,})")
    else:
        print("Copy failed!")
        sys.exit(1)
    
    print(f"New iteration {new_iteration_num} ready with base model: model_weights.pth")
    return new_iteration_num


def run_selfplay_phase(iteration_folder, total_games, num_simulations):
    """
    Run self-play data generation phase
    
    Args:
        iteration_folder: Path to current iteration folder
        total_games: Number of games to generate
        num_simulations: MCTS simulations per move
    
    Returns:
        str: Path to generated training data file
    """
    print("\n" + "="*60)
    print("PHASE 1: SELF-PLAY DATA GENERATION")
    print("="*60)
    
    model_path = iteration_folder / "model_weights.pth"
    print(f"Using model: {model_path}")
    print(f"Games to generate: {total_games}")
    print(f"Simulations per move: {num_simulations}")
    
    # Generate self-play data
    data_file = generate_selfplay_data(
        total_games=total_games,
        num_simulations=num_simulations,
        temperature=1.0,  # Default for self-play
        model_path=str(model_path),
        c_puct=2.0,  # Default
        cpu_utilization=0.9,  # Default
        max_processes_per_gpu=None,  # Auto-detect
        output_dir=str(iteration_folder),
        command_info={
            'phase': 'self-play',
            'iteration': iteration_folder.name,
            'total_games': total_games,
            'num_simulations': num_simulations
        }
    )
    
    print(f"Self-play complete! Data saved to: {data_file}")
    return data_file


def run_training_phase(iteration_folder, data_file, epochs, lr):
    """
    Run model training phase
    
    Args:
        iteration_folder: Path to current iteration folder
        data_file: Path to training data file
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        str: Path to newly trained model
    """
    print("\n" + "="*60)
    print("PHASE 2: MODEL TRAINING")
    print("="*60)
    
    base_model_path = iteration_folder / "model_weights.pth"
    iteration_num = int(iteration_folder.name.split("_")[1])
    new_model_path = iteration_folder / f"new_model_{iteration_num}.pth"
    
    print(f"Base model: {base_model_path}")
    print(f"Training data: {data_file}")
    print(f"Output model: {new_model_path}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    
    # Prepare training arguments
    training_args = [
        '--data', str(data_file),
        '--epochs', str(epochs),
        '--lr', str(lr),
        '--model_path', str(base_model_path),
        '--output', str(iteration_folder)  # For training plots
    ]
    
    # Run training by simulating command line arguments
    original_argv = sys.argv
    try:
        sys.argv = ['train_model.py'] + training_args
        final_model_path = train_model.main()
        
        # Move the final model to our naming convention
        if os.path.exists(final_model_path):
            shutil.move(final_model_path, new_model_path)
            print(f"Model saved as: {new_model_path}")
        
    finally:
        sys.argv = original_argv
    
    print(f"Training complete! New model: {new_model_path}")
    return str(new_model_path)


def run_evaluation_phase(iteration_folder, total_games, num_simulations):
    """
    Run model evaluation phase
    
    Args:
        iteration_folder: Path to current iteration folder
        total_games: Number of evaluation games
        num_simulations: MCTS simulations per move
    
    Returns:
        dict: Evaluation results
    """
    print("\n" + "="*60)
    print("PHASE 3: MODEL EVALUATION")
    print("="*60)
    
    iteration_num = int(iteration_folder.name.split("_")[1])
    old_model_path = iteration_folder / "model_weights.pth"
    new_model_path = iteration_folder / f"new_model_{iteration_num}.pth"
    
    print(f"Old model: {old_model_path}")
    print(f"New model: {new_model_path}")
    print(f"Games to play: {total_games}")
    print(f"Simulations per move: {num_simulations}")
    
    # Run evaluation
    results = evaluate_models(
        old_model_path=str(old_model_path),
        new_model_path=str(new_model_path),
        num_games=total_games,
        num_simulations=num_simulations,
        cpu_utilization=0.9  # Default
    )
    
    # Save results to the iteration folder
    from evaluate_models import save_evaluation_results
    save_evaluation_results(results, str(old_model_path), str(new_model_path), str(iteration_folder))
    
    score_rate = results.get('score_rate', 0)
    threshold = 55.0
    
    print(f"\nEvaluation complete!")
    print(f"New model score rate: {score_rate:.1f}%")
    if score_rate > threshold:
        print(f"RESULT: NEW MODEL ACCEPTED! (>{threshold}%)")
    else:
        print(f"RESULT: NEW MODEL REJECTED! (<={threshold}%)")
    
    return results


def main():
    """Main function to run complete training iteration"""
    parser = argparse.ArgumentParser(
        description='Complete training iteration for Aldarion Chess Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs a complete training iteration:
1. Sets up new iteration folder based on previous results
2. Generates self-play training data
3. Trains new model on the data
4. Evaluates new model vs current best

Example:
  python3 main.py --total_games_self 100 --num_simulations 800 --total_games_eval 50 --epochs 10 --lr 0.2
        """
    )
    
    parser.add_argument('--total_games_self', type=int, default=100,
                        help='Number of self-play games to generate (default: 100)')
    parser.add_argument('--num_simulations', type=int, default=800,
                        help='MCTS simulations per move for both self-play and evaluation (default: 800)')
    parser.add_argument('--total_games_eval', type=int, default=50,
                        help='Number of evaluation games to play (default: 50)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Learning rate for training (default: 0.2)')
    
    args = parser.parse_args()
    
    print("ALDARION CHESS ENGINE - COMPLETE TRAINING ITERATION")
    print("="*60)
    print(f"Self-play games: {args.total_games_self}")
    print(f"MCTS simulations: {args.num_simulations}")
    print(f"Evaluation games: {args.total_games_eval}")
    print(f"Training epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print("="*60)
    
    try:
        # Step 1: Setup iteration
        latest_iteration = find_latest_iteration()
        accept_new_model = analyze_previous_evaluation(latest_iteration)
        new_iteration = setup_new_iteration(latest_iteration, accept_new_model)
        iteration_folder = Path(f"Iterations/Iteration_{new_iteration}")
        
        # Step 2: Self-play data generation
        data_file = run_selfplay_phase(iteration_folder, args.total_games_self, args.num_simulations)
        
        # Step 3: Model training
        new_model_file = run_training_phase(iteration_folder, data_file, args.epochs, args.lr)
        
        # Step 4: Model evaluation
        eval_results = run_evaluation_phase(iteration_folder, args.total_games_eval, args.num_simulations)
        
        # Final summary
        print("\n" + "="*60)
        print("TRAINING ITERATION COMPLETE")
        print("="*60)
        print(f"Iteration: {new_iteration}")
        print(f"Folder: {iteration_folder}")
        print(f"New model score rate: {eval_results.get('score_rate', 0):.1f}%")
        print(f"Training data: {os.path.basename(data_file)}")
        print(f"New model: new_model_{new_iteration}.pth")
        
        score_rate = eval_results.get('score_rate', 0)
        if score_rate > 55.0:
            print("STATUS: New model will be accepted in next iteration!")
        else:
            print("STATUS: New model will be rejected in next iteration.")
            
        print("\nIteration complete! Ready for next iteration.")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during training iteration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()