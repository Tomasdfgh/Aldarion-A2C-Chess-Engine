import MTCS as mt
import model as md
import torch

"""
Parallel Training Data Generation Commands:
==================================================

#background process run
Start the training in tmux:
tmux new-session -d -s training 'python3 -u parallel_training_data.py --total_games 60 --num_simulations 800
--cpu_utilization 0.9'

To view the output:
tmux attach -t training

To detach (leave it running in background):
Press Ctrl+B then D

To check if it's still running:
tmux list-sessions

To kill the session:
tmux kill-session -t training

# QUICK TESTING (low resource usage)
python3 parallel_training_data.py --total_games 8 --num_simulations 10
python3 parallel_training_data.py --total_games 16 --num_simulations 25 --max_processes_per_gpu 4

# LIGHT WORKLOAD (~30-50% hardware utilization)
python3 parallel_training_data.py --total_games 50 --num_simulations 50 --cpu_utilization 0.50
python3 parallel_training_data.py --total_games 100 --num_simulations 75 --max_processes_per_gpu 6

# MEDIUM WORKLOAD (~60-80% hardware utilization)
python3 parallel_training_data.py --total_games 200 --num_simulations 100 --cpu_utilization 0.75
python3 parallel_training_data.py --total_games 300 --num_simulations 150 --max_processes_per_gpu 10

# HEAVY WORKLOAD (~85-95% hardware utilization)
python3 parallel_training_data.py --total_games 500 --num_simulations 200 --cpu_utilization 0.90
python3 parallel_training_data.py --total_games 1000 --num_simulations 250 --cpu_utilization 0.95

# MAXIMUM INTENSITY (95%+ hardware utilization)
python3 parallel_training_data.py --total_games 2000 --num_simulations 300 --cpu_utilization 0.95
python3 parallel_training_data.py --total_games 5000 --num_simulations 500 --cpu_utilization 0.95

# OVERNIGHT RUNS (high-quality training data)
python3 parallel_training_data.py --total_games 10000 --num_simulations 400 --cpu_utilization 0.90
python3 parallel_training_data.py --total_games 20000 --num_simulations 600 --cpu_utilization 0.85

# PARAMETER EXPLANATIONS:
# --total_games: Number of self-play games to generate
# --num_simulations: MCTS simulations per move (higher = better quality, slower)
# --cpu_utilization: Target CPU usage (0.5 = 50%, 0.95 = 95%)
# --max_processes_per_gpu: Manual override for processes per GPU
# --temperature: Move selection temperature (default: 0.8)
# --output: Custom output filename

# HARDWARE-SPECIFIC NOTES:
# - System auto-detects GPUs and distributes workload evenly
# - Each process loads its own model copy (~500MB GPU memory)
# - Higher simulations = exponentially longer training time
# - Monitor GPU/CPU usage with: nvidia-smi, htop

To play a single game with AlphaZero features:
# mt.run_game(model, 0.8, 1500, device)
"""


if __name__ == "__main__":
    import os
    import subprocess
    import sys
    import time
    import argparse
    from datetime import datetime
    import glob
    '''
    def run_command(command, description):
        """Run a subprocess command with error handling"""
        print(f"\n{'='*60}")
        print(f"STEP: {description}")
        print(f"{'='*60}")
        print(f"Running: {command}")
        print()
        
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=False)
            print(f"\n✅ {description} completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {description} failed with exit code {e.returncode}")
            return False
        except KeyboardInterrupt:
            print(f"\n⚠️  {description} interrupted by user")
            return False
    
    def find_latest_training_data():
        """Find the most recent training data file"""
        pattern = "parallel_training_data_*.pkl"
        files = glob.glob(pattern)
        if not files:
            return None
        return max(files, key=os.path.getctime)  # Most recent file
    
    def get_version_number(model_path):
        """Extract version number from model path, or return 0 if not versioned"""
        try:
            if "_v" in model_path:
                return int(model_path.split("_v")[1].split(".")[0])
            return 0
        except:
            return 0
    
    def main_training_loop():
        """Main AlphaZero training loop"""
        parser = argparse.ArgumentParser(
            description='Aldarion Chess Engine - Complete Training Loop',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Training Loop Steps:
1. Self-Play Generation: Generate games with current best model
2. Model Training: Train neural network on self-play data  
3. Model Evaluation: Test new model against previous best
4. Model Selection: Keep new model if it wins >55% of games
5. Repeat: Use best model for next iteration

Examples:
  # Full training loop (default settings)
  python3 main.py
  
  # Quick training loop (fewer games/epochs for testing)
  python3 main.py --games 20 --epochs 5 --eval_games 10
  
  # High-intensity training
  python3 main.py --games 2000 --simulations 800 --epochs 20 --eval_games 100
            """
        )
        
        # Self-play parameters
        parser.add_argument('--games', type=int, default=200,
                            help='Games to generate per iteration (default: 200)')
        parser.add_argument('--simulations', type=int, default=400,
                            help='MCTS simulations per move (default: 400)')
        parser.add_argument('--cpu_utilization', type=float, default=0.85,
                            help='CPU utilization for self-play (default: 0.85)')
        
        # Training parameters  
        parser.add_argument('--epochs', type=int, default=10,
                            help='Training epochs per iteration (default: 10)')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='Training batch size (default: 32)')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate (default: 0.001)')
        
        # Evaluation parameters
        parser.add_argument('--eval_games', type=int, default=50,
                            help='Games for model evaluation (default: 50)')
        parser.add_argument('--eval_simulations', type=int, default=200, 
                            help='MCTS simulations for evaluation (default: 200)')
        parser.add_argument('--win_threshold', type=float, default=55.0,
                            help='Win rate threshold to accept new model (default: 55.0)')
        
        # Loop control
        parser.add_argument('--max_iterations', type=int, default=100,
                            help='Maximum training iterations (default: 100)')
        parser.add_argument('--start_iteration', type=int, default=1,
                            help='Starting iteration number (default: 1)')
        
        # Skip steps (for debugging/resuming)
        parser.add_argument('--skip_selfplay', action='store_true',
                            help='Skip self-play generation (use existing data)')
        parser.add_argument('--skip_training', action='store_true', 
                            help='Skip neural network training')
        parser.add_argument('--skip_evaluation', action='store_true',
                            help='Skip model evaluation (accept all new models)')
        
        args = parser.parse_args()
        
        print("🚀 ALDARION CHESS ENGINE - ALPHAZERO TRAINING LOOP")
        print("="*60)
        print(f"Configuration:")
        print(f"  Games per iteration: {args.games}")
        print(f"  MCTS simulations: {args.simulations}")  
        print(f"  Training epochs: {args.epochs}")
        print(f"  Evaluation games: {args.eval_games}")
        print(f"  Max iterations: {args.max_iterations}")
        print(f"  Win threshold: {args.win_threshold}%")
        print()
        
        # Initialize model tracking
        current_best_model = "model_weights.pth"
        if not os.path.exists(current_best_model):
            print("⚠️  No initial model found. Creating random model...")
            model = md.ChessNet()
            torch.save(model.state_dict(), current_best_model)
            print(f"✅ Initial model saved as {current_best_model}")
        
        iteration = args.start_iteration
        
        try:
            while iteration <= args.max_iterations:
                print(f"\n🔄 TRAINING ITERATION {iteration}/{args.max_iterations}")
                print("="*60)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # STEP 1: Self-Play Data Generation
                if not args.skip_selfplay:
                    selfplay_cmd = (f"python3 parallel_training_data.py "
                                  f"--total_games {args.games} "
                                  f"--num_simulations {args.simulations} "
                                  f"--cpu_utilization {args.cpu_utilization} "
                                  f"--model_path {current_best_model} "
                                  f"--output training_data_iter_{iteration}_{timestamp}.pkl")
                    
                    if not run_command(selfplay_cmd, f"Self-Play Generation (Iteration {iteration})"):
                        print("❌ Self-play failed. Stopping training loop.")
                        break
                    
                    latest_data = f"training_data_iter_{iteration}_{timestamp}.pkl"
                else:
                    latest_data = find_latest_training_data()
                    if latest_data is None:
                        print("❌ No training data found and self-play skipped!")
                        break
                    print(f"📁 Using existing training data: {latest_data}")
                
                # STEP 2: Neural Network Training
                if not args.skip_training:
                    new_model_path = f"model_weights_v{iteration}_{timestamp}.pth"
                    training_cmd = (f"python3 train_model.py "
                                  f"--data {latest_data} "
                                  f"--epochs {args.epochs} "
                                  f"--batch_size {args.batch_size} "
                                  f"--lr {args.lr} "
                                  f"--model_path {current_best_model} "
                                  f"--output_dir . ")
                    
                    if not run_command(training_cmd, f"Neural Network Training (Iteration {iteration})"):
                        print("❌ Training failed. Stopping training loop.")
                        break
                    
                    # Find the generated model file
                    if not os.path.exists("model_weights_final.pth"):
                        print("❌ Training completed but no final model found!")
                        break
                    
                    # Rename to versioned name
                    os.rename("model_weights_final.pth", new_model_path)
                    print(f"📦 New model saved as: {new_model_path}")
                else:
                    # Skip training, use current model
                    new_model_path = current_best_model
                    print(f"⏭️  Training skipped, using current model: {new_model_path}")
                
                # STEP 3: Model Evaluation
                if not args.skip_evaluation and new_model_path != current_best_model:
                    eval_cmd = (f"python3 evaluate_models.py "
                              f"--old_model {current_best_model} "
                              f"--new_model {new_model_path} "
                              f"--num_games {args.eval_games} "
                              f"--num_simulations {args.eval_simulations}")
                    
                    print(f"🥊 Evaluating: {os.path.basename(new_model_path)} vs {os.path.basename(current_best_model)}")
                    
                    evaluation_success = run_command(eval_cmd, f"Model Evaluation (Iteration {iteration})")
                    
                    if evaluation_success:
                        # New model won - update current best
                        print(f"🏆 NEW MODEL ACCEPTED! Updating best model.")
                        old_best = current_best_model
                        current_best_model = new_model_path
                        
                        # Keep backup of old model
                        if old_best != "model_weights.pth":
                            backup_name = f"model_weights_backup_v{iteration-1}.pth"
                            if os.path.exists(old_best):
                                os.rename(old_best, backup_name)
                        
                        # Update main model file
                        import shutil
                        shutil.copy2(current_best_model, "model_weights.pth")
                        
                    else:
                        # New model lost - keep old model
                        print(f"💀 New model rejected. Keeping current best: {os.path.basename(current_best_model)}")
                        # Clean up rejected model
                        if os.path.exists(new_model_path) and new_model_path != current_best_model:
                            os.remove(new_model_path)
                            print(f"🗑️  Removed rejected model: {new_model_path}")
                else:
                    # Skip evaluation - accept new model
                    if not args.skip_evaluation:
                        print("⏭️  No evaluation needed (same model)")
                    else:
                        print("⏭️  Evaluation skipped - accepting new model")
                        current_best_model = new_model_path
                        import shutil
                        shutil.copy2(current_best_model, "model_weights.pth")
                
                print(f"\n✅ ITERATION {iteration} COMPLETE")
                print(f"Current best model: {os.path.basename(current_best_model)}")
                
                iteration += 1
                
                # Brief pause between iterations
                time.sleep(5)
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Training loop interrupted by user at iteration {iteration}")
        except Exception as e:
            print(f"\n❌ Training loop failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n🏁 TRAINING LOOP FINISHED")
        print(f"Final best model: {current_best_model}")
        print("="*60)
    
    # Run the main training loop
    main_training_loop()
    '''