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

## Training System Commands

### Initialize the System

```bash
python -m src.run init
```

Initialize the system with a random model to bootstrap the training process. This creates the initial best model required for self-play and training.

### Check System Status

```bash
python -m src.run status
```

Display the current system status including best model availability, number of candidate models, training data statistics, and system directories.

### Start Training Workers

The training system consists of three parallel workers that work together:

#### Start Self-Play Worker
```bash
python -m src.run self
```

Generates training data by playing games using the current best model. Runs continuously and saves game data for training.

#### Start Training Worker
```bash
python -m src.run opt
```

Trains new models using the latest self-play data. Automatically limits candidate pool to 20 models and waits for evaluation when pool is full.

#### Start Evaluation Worker
```bash
python -m src.run eval
```

Evaluates candidate models against the current best model. Promotes superior models and archives evaluated candidates in FIFO order.

The system automatically coordinates between workers: self-play generates data, training creates new models, and evaluation determines which models become the new best model.

This project is still in development.
