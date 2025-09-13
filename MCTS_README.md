# MCTS Implementation for Aldarion A2C Chess Engine

## Overview

This is a complete Monte Carlo Tree Search (MCTS) implementation for the Aldarion A2C Chess Engine, following the AlphaZero methodology. The implementation integrates seamlessly with the existing neural network architecture and board representation.

## Key Features

### Core MCTS Components

1. **MTCSNode Class**: Enhanced node structure with parent references and expansion tracking
2. **select_node()**: UCB-based tree traversal to find leaf nodes
3. **expand_node()**: Creates children for all legal moves using neural network policy
4. **simulate()**: Uses neural network evaluation (no random rollouts)
5. **backpropagate()**: Updates visit counts, values, and Q-values up the tree
6. **mcts_search()**: Main MCTS loop running all four phases

### Integration Features

- **Neural Network Integration**: Uses existing ChessNet model with 119-channel input
- **Board Representation**: Works with existing `board_reader.py` functions
- **Policy Mapping**: Converts neural network policy to legal move probabilities
- **Training Data Generation**: Produces data in format compatible with training pipeline

## File Structure

### `/home/thomas-nguyen/Projects/Aldarion-A2C-Chess-Engine/MTCS.py`
Main MCTS implementation with all core functions.

### `/home/thomas-nguyen/Projects/Aldarion-A2C-Chess-Engine/generate_training_data.py`
Standalone script for generating training data through self-play.

### `/home/thomas-nguyen/Projects/Aldarion-A2C-Chess-Engine/main.py`
Updated test script that demonstrates MCTS functionality.

## Usage Examples

### Basic MCTS Search
```python
import MTCS as mt
import model as md
import torch

# Load model
model = md.ChessNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

# Test MCTS
mt.test_board(model, device)
```

### Get Best Move for Position
```python
# Get best move for a specific position
board_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
best_move, move_probs = mt.get_best_move(
    model=model,
    board_fen=board_fen,
    num_simulations=100,
    device=device,
    temperature=0.0  # Deterministic
)
print(f"Best move: {best_move}")
```

### Generate Training Data
```bash
# Generate training data with self-play
python generate_training_data.py --num_games 10 --num_simulations 100

# With custom parameters
python generate_training_data.py \
    --num_games 50 \
    --num_simulations 200 \
    --temperature 0.8 \
    --output my_training_data.pkl
```

### Play Full Game
```python
# Play a complete game and collect training data
training_data = mt.run_game(
    model=model,
    temperature=0.8,
    num_simulations=100,
    device=device
)

# training_data is a list of (board_state, move_probabilities, game_outcome) tuples
print(f"Generated {len(training_data)} training examples")
```

## Algorithm Details

### MCTS Phases

1. **Selection**: Uses PUCT formula for exploration vs exploitation
   ```
   UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
   ```

2. **Expansion**: Adds all legal moves as children with neural network prior probabilities

3. **Simulation**: Evaluates position using neural network value head (no random rollouts)

4. **Backpropagation**: Updates statistics from leaf to root, alternating value signs

### Key Parameters

- **num_simulations**: Number of MCTS iterations per move (default: 100)
- **temperature**: Controls move selection randomness (0.0 = deterministic, 1.0 = proportional to visits)
- **c_puct**: Exploration constant in UCB formula (default: 1.0)

## Neural Network Integration

### Input Format
- Uses `board_to_full_alphazero_input()` to create 119-channel input
- Maintains game history for proper position evaluation

### Policy Processing
- Converts 4672-dimensional policy output to legal move probabilities
- Handles both regular moves (4096) and promotions (576)
- Uses `board_to_legal_policy_hash()` for mapping

### Value Processing
- Uses tanh-activated value head output (-1 to +1)
- Adjusts perspective based on current player to move

## Training Data Format

Training data is generated as tuples of:
1. **Board State**: FEN string of the position
2. **Move Probabilities**: Dictionary mapping UCI moves to visit-count-based probabilities
3. **Game Outcome**: +1 (win), 0 (draw), -1 (loss) from player's perspective

Example:
```python
training_example = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    {"e2e4": 0.35, "d2d4": 0.28, "g1f3": 0.15, ...},
    0  # Draw
)
```

## Performance Considerations

### GPU Usage
- All neural network evaluations are performed on GPU when available
- Batch processing could be added for multiple position evaluations

### Memory Management
- Tree is rebuilt for each move to prevent memory growth
- Could implement tree reuse for stronger play

### Speed Optimizations
- Uses `torch.no_grad()` for inference
- Efficient policy conversion using existing board_reader functions

## Integration with Existing Codebase

### Compatible Functions
- `board_to_full_alphazero_input()`: Board → Neural network input
- `board_to_legal_policy_hash()`: Policy → Legal move probabilities
- `ChessNet`: Existing neural network architecture
- Training pipeline: Compatible data format

### Data Types
- All operations use `float32` for compatibility
- Proper device placement for GPU acceleration
- Chess library integration for move validation

## Testing

Run the test suite:
```bash
python main.py
```

Expected output shows MCTS finding reasonable moves with proper visit counts and Q-values.

## Future Enhancements

1. **Tree Reuse**: Reuse search tree between moves
2. **Batch Evaluation**: Evaluate multiple positions simultaneously
3. **Time Management**: Add time-based search limits
4. **Opening Book**: Integration with opening databases
5. **Endgame Tablebase**: Use perfect endgame knowledge

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `num_simulations` or implement tree pruning
2. **Slow Performance**: Ensure model is on GPU, reduce simulation count for testing
3. **Invalid Moves**: Check that `board_to_legal_policy_hash()` is working correctly
4. **Training Data Issues**: Verify data format matches training pipeline expectations

### Debug Functions

- `print_tree()`: Visualize search tree structure
- `test_board()`: Basic functionality verification
- Custom logging can be added to any MCTS phase

This implementation provides a solid foundation for AlphaZero-style self-play training and strong chess play when combined with a well-trained neural network.