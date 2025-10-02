# Aldarion: Actor-Critic, Reinforcement Learning Chess Engine

![356070291-1238b477-6651-4c25-8f51-3549290ad56d](https://github.com/user-attachments/assets/4a488daa-3fcf-4bdf-b92b-101eabec0b58)



<p align="center">
  <img src="https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/44381ed4-ac65-4c96-8513-901336e4223c" alt="Aldarion Chess Engine" width="300">
</p>

## Introduction

Welcome to Aldarion, a chess engine trained using an adapted AlphaZero algorithm. Aldarion leverages an actor-critic model, consisting of a policy head for determining the next move and a value head for predicting the probability of winning from the current state. Additionally, it employs Monte Carlo Tree Search (MCTS) to predict the best possible moves after running 300 simulations following each move. This document provides insights into how the board is captured, the policy vector structure and its utilization for move selection, MCTS traversal, training procedures, and a comprehensive overview of the model's architecture.

## Setup Instruction

### Prerequisites
- Linux or WSL (Windows Subsystem for Linux)
- Miniconda or Anaconda installed
- (Optional) NVIDIA GPU for faster training

### Create and activate the Environment
```bash
conda env create -f environment.yml
conda activate aldarion
```

### Verify if GPUs are detectable
```bash
python -c "import torch; import chess; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Example script commands

### Create Baseline Model with Random Weights

```bash
python -c "import os; import torch; import model as md; os.makedirs('model_weights', exist_ok=True); torch.save(md.ChessNet().state_dict(), 'model_weights/model_weights.pth'); print('Random weights saved to model_weights/model_weights.pth')"
```
Run that command to initialize a starting model with random weights. The weight will be saved in the model_weights folder as a .pth file. You can use this file in order to run selfplay to create training data (below).

### Run a selfplay trial

```bash
python3 selfplay_generate_data.py --total_games 250 --num_simulations  128 --cpu_utilization 0.9 --model_path <Path to the model weight (.pth) file>
```

This example command will trigger model selfplaying to collect training data with 250 selfplay games, 128 visits/move, and 90 percent of CPU utilization.

### Run Model Training

```bash
python3 train_model.py --data <Path to training data .pkl file> --validation_data <Path to Validation data .pkl file> --model_path <Path to the model weight (.pth) file> --epochs 5 --lr 0.0001 --batch_size 32
```

This example command will trigger model training on the model from the path to the weights. Both training and validation data can be obtained from the selfplay process.

### Run Evaluation Between two models

```bash
python3 evaluate_models.py --old_model <Path to the old model weight (.pth) file> --new_model <Path to the new model weight (.pth) file> --num_games 5 --num_simulations 5 --cpu_utilization 0.9
```

This example will run a competitive evaluation between two models inorder to determine which model is superior. Games will be deterministic, and opening states are randomly chosen from Chess960.

This project is still in development.
