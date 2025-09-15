#!/usr/bin/env python3
"""
Single AlphaZero Iteration Runner for Aldarion Chess Engine

This script runs one complete AlphaZero iteration:
1. Generate self-play training data using current best model
2. Train new neural network on that data
3. Evaluate new model against current best model
4. Accept/reject new model based on win rate
5. Save results and update champion

Perfect for running controlled, single iterations with full logging.
"""

import os
import sys
import subprocess
import argparse
import time
import shutil
import json
from datetime import datetime
from pathlib import Path


def run_command_with_output(command, description):
    """
    Run a subprocess command with real-time output and error handling
    
    Args:
        command: Command string to execute
        description: Human-readable description
    
    Returns:
        tuple: (success: bool, return_code: int)
    """
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    print()
    
    try:
        # Run command with real-time output
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            print(f"\n✅ {description} completed successfully!")
            return True, return_code
        else:
            print(f"\n❌ {description} failed with exit code {return_code}")
            return False, return_code
            
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} interrupted by user")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        return False, -1
    except Exception as e:
        print(f"\n💥 {description} failed with error: {e}")
        return False, -1


def find_current_best_model():
    """Find the current best model file"""
    # Check for canonical best model
    if os.path.exists("model_weights/model_weights.pth"):
        return "model_weights/model_weights.pth"
    
    # Check for any model in model_weights directory
    model_weights_dir = Path("model_weights")
    if model_weights_dir.exists():
        model_files = list(model_weights_dir.glob("*.pth"))
        if model_files:
            # Use the most recently modified model
            latest_model = max(model_files, key=os.path.getmtime)
            return str(latest_model)
    
    # No model found
    return None


def create_iteration_directory(iteration_num):
    """Create directory to store iteration results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    iteration_dir = f"iterations/iteration_{iteration_num:03d}_{timestamp}"
    os.makedirs(iteration_dir, exist_ok=True)
    return iteration_dir


def save_iteration_results(iteration_dir, results):
    """Save iteration results to JSON file"""
    results_file = os.path.join(iteration_dir, "iteration_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Iteration results saved to: {results_file}")
    return results_file


def run_single_iteration(args):
    """
    Run one complete AlphaZero iteration
    
    Returns:
        dict: Iteration results and statistics
    """
    iteration_start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create iteration directory for organizing results
    iteration_dir = create_iteration_directory(args.iteration_number)
    
    print("🚀 ALDARION CHESS ENGINE - SINGLE ALPHAZERO ITERATION")
    print("="*70)
    print(f"Iteration: {args.iteration_number}")
    print(f"Timestamp: {timestamp}")
    print(f"Results directory: {iteration_dir}")
    
    results = {
        'iteration_number': args.iteration_number,
        'timestamp': timestamp,
        'iteration_directory': iteration_dir,
        'parameters': {
            'selfplay_games': args.selfplay_games,
            'selfplay_simulations': args.selfplay_simulations,
            'training_epochs': args.training_epochs,
            'training_batch_size': args.training_batch_size,
            'training_lr': args.training_lr,
            'eval_games': args.eval_games,
            'eval_simulations': args.eval_simulations,
            'win_threshold': args.win_threshold,
            'cpu_utilization': args.cpu_utilization
        },
        'steps': {},
        'champion_changed': False,
        'success': False
    }
    
    # STEP 1: Find current best model
    print(f"\n📋 STEP 0: Locating current best model...")
    current_best_model = find_current_best_model()
    
    if current_best_model is None:
        print("❌ No current best model found! Please ensure model_weights/model_weights.pth exists.")
        results['error'] = "No current best model found"
        save_iteration_results(iteration_dir, results)
        return results
    
    print(f"✅ Current best model: {current_best_model}")
    results['current_best_model'] = current_best_model
    
    # Copy current best to iteration directory for record keeping
    shutil.copy2(current_best_model, os.path.join(iteration_dir, "old_model.pth"))
    
    try:
        # STEP 1: Generate self-play training data
        selfplay_filename = f"selfplay_data_iter_{args.iteration_number}_{timestamp}.pkl"
        selfplay_command = (
            f"python3 parallel_training_data.py "
            f"--total_games {args.selfplay_games} "
            f"--num_simulations {args.selfplay_simulations} "
            f"--cpu_utilization {args.cpu_utilization} "
            f"--model_path {current_best_model} "
            f"--output {selfplay_filename}"
        )
        
        step_start = time.time()
        success, return_code = run_command_with_output(
            selfplay_command, 
            f"Self-Play Data Generation ({args.selfplay_games} games)"
        )
        
        results['steps']['selfplay'] = {
            'success': success,
            'return_code': return_code,
            'duration_seconds': time.time() - step_start,
            'command': selfplay_command,
            'output_file': f"training_data/{selfplay_filename}"
        }
        
        if not success:
            results['error'] = "Self-play data generation failed"
            save_iteration_results(iteration_dir, results)
            return results
        
        selfplay_data_path = f"training_data/{selfplay_filename}"
        if not os.path.exists(selfplay_data_path):
            print(f"❌ Expected self-play data file not found: {selfplay_data_path}")
            results['error'] = f"Self-play data file not found: {selfplay_data_path}"
            save_iteration_results(iteration_dir, results)
            return results
        
        # STEP 2: Train new neural network
        new_model_filename = f"new_model_iter_{args.iteration_number}_{timestamp}.pth"
        training_command = (
            f"python3 train_model.py "
            f"--data {selfplay_filename} "
            f"--epochs {args.training_epochs} "
            f"--batch_size {args.training_batch_size} "
            f"--lr {args.training_lr} "
            f"--model_path {current_best_model} "
            f"--output_dir model_weights"
        )
        
        step_start = time.time()
        success, return_code = run_command_with_output(
            training_command,
            f"Neural Network Training ({args.training_epochs} epochs)"
        )
        
        results['steps']['training'] = {
            'success': success,
            'return_code': return_code,
            'duration_seconds': time.time() - step_start,
            'command': training_command
        }
        
        if not success:
            results['error'] = "Neural network training failed"
            save_iteration_results(iteration_dir, results)
            return results
        
        # Find the trained model (train_model.py creates model_weights_final.pth)
        trained_model_path = "model_weights/model_weights_final.pth"
        if not os.path.exists(trained_model_path):
            print(f"❌ Expected trained model not found: {trained_model_path}")
            results['error'] = f"Trained model not found: {trained_model_path}"
            save_iteration_results(iteration_dir, results)
            return results
        
        # Rename trained model to iteration-specific name
        new_model_path = f"model_weights/{new_model_filename}"
        shutil.move(trained_model_path, new_model_path)
        print(f"📦 New model saved as: {new_model_path}")
        results['new_model_path'] = new_model_path
        
        # Copy new model to iteration directory
        shutil.copy2(new_model_path, os.path.join(iteration_dir, "new_model.pth"))
        
        # STEP 3: Evaluate new model against current best
        evaluation_command = (
            f"python3 evaluate_models.py "
            f"--old_model {current_best_model} "
            f"--new_model {new_model_path} "
            f"--num_games {args.eval_games} "
            f"--num_simulations {args.eval_simulations} "
            f"--win_threshold {args.win_threshold} "
            f"--cpu_utilization {args.cpu_utilization}"
        )
        
        step_start = time.time()
        print(f"\n🥊 Starting model evaluation...")
        print(f"Old model: {os.path.basename(current_best_model)}")
        print(f"New model: {os.path.basename(new_model_path)}")
        print(f"Games: {args.eval_games}, Win threshold: {args.win_threshold}%")
        
        success, return_code = run_command_with_output(
            evaluation_command,
            f"Model Evaluation ({args.eval_games} games)"
        )
        
        results['steps']['evaluation'] = {
            'success': success,
            'return_code': return_code,
            'duration_seconds': time.time() - step_start,
            'command': evaluation_command,
            'new_model_accepted': return_code == 0  # evaluate_models.py returns 0 for accept, 1 for reject
        }
        
        # STEP 4: Update champion based on evaluation results
        if return_code == 0:
            # New model won! Update the champion
            print(f"\n🏆 NEW CHAMPION! New model accepted.")
            
            # Backup old champion
            backup_name = f"model_weights_backup_iter_{args.iteration_number}_{timestamp}.pth"
            backup_path = f"model_weights/{backup_name}"
            shutil.copy2(current_best_model, backup_path)
            print(f"📁 Old champion backed up as: {backup_path}")
            
            # Update main model file
            shutil.copy2(new_model_path, "model_weights/model_weights.pth")
            print(f"✅ Updated model_weights/model_weights.pth with new champion")
            
            results['champion_changed'] = True
            results['new_champion'] = new_model_path
            results['old_champion_backup'] = backup_path
            results['success'] = True
            
        else:
            # New model lost, keep old champion
            print(f"\n💀 New model rejected. Keeping current champion: {os.path.basename(current_best_model)}")
            
            # Clean up rejected model to save space
            os.remove(new_model_path)
            print(f"🗑️  Removed rejected model: {new_model_path}")
            
            results['champion_changed'] = False
            results['rejected_model'] = new_model_path
            results['success'] = True  # Iteration completed successfully even if model rejected
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Iteration interrupted by user")
        results['error'] = "Interrupted by user"
        results['success'] = False
        
    except Exception as e:
        print(f"\n💥 Iteration failed with error: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
        results['success'] = False
    
    # Final summary
    total_time = time.time() - iteration_start_time
    results['total_duration_seconds'] = total_time
    results['total_duration_minutes'] = total_time / 60
    
    print(f"\n{'='*70}")
    print(f"🏁 ITERATION {args.iteration_number} COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Champion changed: {'✅ YES' if results['champion_changed'] else '❌ NO'}")
    print(f"Iteration success: {'✅ SUCCESS' if results['success'] else '❌ FAILED'}")
    
    if results['success']:
        if results['champion_changed']:
            print(f"🏆 New champion: {os.path.basename(results['new_champion'])}")
        else:
            print(f"👑 Champion remains: {os.path.basename(current_best_model)}")
    
    # Save results
    save_iteration_results(iteration_dir, results)
    
    return results


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Run single AlphaZero iteration for Aldarion Chess Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Single Iteration Workflow:
1. Generate self-play data using current best model
2. Train new neural network on that data
3. Evaluate new model vs current best model
4. Accept new model if win rate exceeds threshold
5. Save all results and update champion

Examples:
  # Quick iteration (testing)
  python3 main.py --iteration_number 1 --selfplay_games 50 --training_epochs 5 --eval_games 20
  
  # Standard iteration
  python3 main.py --iteration_number 5 --selfplay_games 200 --training_epochs 15 --eval_games 50
  
  # High-quality iteration
  python3 main.py --iteration_number 10 --selfplay_games 500 --training_epochs 25 --eval_games 100 --selfplay_simulations 600
        """
    )
    
    # Iteration identification
    parser.add_argument('--iteration_number', type=int, required=True,
                        help='Iteration number for tracking and organization')
    
    # Self-play parameters
    parser.add_argument('--selfplay_games', type=int, default=200,
                        help='Number of self-play games to generate (default: 200)')
    parser.add_argument('--selfplay_simulations', type=int, default=400,
                        help='MCTS simulations per move during self-play (default: 400)')
    
    # Training parameters
    parser.add_argument('--training_epochs', type=int, default=15,
                        help='Number of training epochs (default: 15)')
    parser.add_argument('--training_batch_size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--training_lr', type=float, default=0.2,
                        help='Learning rate for SGD training (default: 0.2)')
    
    # Evaluation parameters
    parser.add_argument('--eval_games', type=int, default=50,
                        help='Number of evaluation games (default: 50)')
    parser.add_argument('--eval_simulations', type=int, default=200,
                        help='MCTS simulations per move during evaluation (default: 200)')
    parser.add_argument('--win_threshold', type=float, default=55.0,
                        help='Win rate threshold for accepting new model (default: 55.0%)')
    
    # System parameters
    parser.add_argument('--cpu_utilization', type=float, default=0.85,
                        help='CPU utilization for parallel processing (default: 0.85)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.iteration_number <= 0:
        print("Error: iteration_number must be positive")
        sys.exit(1)
    
    if not (0.0 <= args.cpu_utilization <= 1.0):
        print("Error: cpu_utilization must be between 0.0 and 1.0")
        sys.exit(1)
    
    if not (0.0 <= args.win_threshold <= 100.0):
        print("Error: win_threshold must be between 0.0 and 100.0")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("model_weights", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)
    os.makedirs("training_data_stats", exist_ok=True)
    os.makedirs("iterations", exist_ok=True)
    
    # Run the iteration
    try:
        results = run_single_iteration(args)
        
        # Exit with appropriate code
        if results['success']:
            if results['champion_changed']:
                print(f"\n🎉 Iteration {args.iteration_number} successful - NEW CHAMPION!")
                sys.exit(0)
            else:
                print(f"\n✅ Iteration {args.iteration_number} successful - champion unchanged")
                sys.exit(0)
        else:
            print(f"\n❌ Iteration {args.iteration_number} failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\nIteration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()