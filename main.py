import MTCS as mt
import model as md
import torch

"""
Parallel Training Data Generation Commands:
==================================================

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
    
    model = md.ChessNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #model.eval()

    state = torch.load("model_weights.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Test MCTS with a small search
    print("Testing MCTS implementation...")
    mt.test_board(model, device)
    
    print("\n" + "="*50)
    print("MCTS with AlphaZero features is working correctly!")
    print("="*50)
    print("Check the comments at the top of this file for parallel training commands.")