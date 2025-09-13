#!/usr/bin/env python3
"""
Script to generate training data using MCTS self-play
"""

import MTCS as mt
import model as md
import torch
import argparse
import pickle
import os
from datetime import datetime

def generate_games(model, num_games, num_simulations, temperature, device):
    """Generate multiple games for training data"""
    all_training_data = []
    
    print(f"Generating {num_games} games with {num_simulations} simulations each...")
    
    for game_num in range(num_games):
        print(f"\n{'='*60}")
        print(f"Game {game_num + 1}/{num_games}")
        print(f"{'='*60}")
        
        try:
            # Generate one game
            training_data = mt.run_game(model, temperature, num_simulations, device)
            all_training_data.extend(training_data)
            
            print(f"Game {game_num + 1} completed: {len(training_data)} examples collected")
            print(f"Total examples so far: {len(all_training_data)}")
            
        except Exception as e:
            print(f"Error in game {game_num + 1}: {e}")
            continue
    
    return all_training_data

def save_training_data(training_data, filename=None):
    """Save training data to file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_data_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"Training data saved to {filename}")
    return filename

def load_training_data(filename):
    """Load training data from file"""
    with open(filename, 'rb') as f:
        training_data = pickle.load(f)
    
    print(f"Loaded {len(training_data)} training examples from {filename}")
    return training_data

def main():
    parser = argparse.ArgumentParser(description='Generate MCTS training data')
    parser.add_argument('--num_games', type=int, default=5, 
                        help='Number of games to play (default: 5)')
    parser.add_argument('--num_simulations', type=int, default=100, 
                        help='Number of MCTS simulations per move (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.8, 
                        help='Temperature for move selection (default: 0.8)')
    parser.add_argument('--model_path', type=str, default='model_weights.pth',
                        help='Path to model weights (default: model_weights.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename for training data')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = md.ChessNet()
    model.to(device)
    
    if os.path.exists(args.model_path):
        state = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        print("Model loaded successfully")
    else:
        print(f"Warning: Model file {args.model_path} not found. Using random weights.")
    
    # Generate training data
    training_data = generate_games(
        model=model,
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        temperature=args.temperature,
        device=device
    )
    
    # Save training data
    if len(training_data) > 0:
        filename = save_training_data(training_data, args.output)
        
        # Print statistics
        print(f"\n{'='*60}")
        print("TRAINING DATA GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total training examples: {len(training_data)}")
        print(f"Saved to: {filename}")
        
        # Sample statistics
        outcomes = [example[2] for example in training_data]
        white_wins = sum(1 for outcome in outcomes if outcome > 0)
        black_wins = sum(1 for outcome in outcomes if outcome < 0)
        draws = sum(1 for outcome in outcomes if outcome == 0)
        
        print(f"\nGame outcomes in dataset:")
        print(f"White wins: {white_wins}")
        print(f"Black wins: {black_wins}")
        print(f"Draws: {draws}")
        
    else:
        print("No training data generated!")

if __name__ == "__main__":
    main()