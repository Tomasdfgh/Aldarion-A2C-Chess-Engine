# Aldarion: Actor-Critic, Reinforcement Learning Chess Engine

<p align="center">
  <img src="https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/44381ed4-ac65-4c96-8513-901336e4223c" alt="Aldarion Chess Engine" width="300">
</p>

## Introduction

Welcome to Aldarion, a chess engine trained using an adapted AlphaZero algorithm. Aldarion leverages an actor-critic model, consisting of a policy head for determining the next move and a value head for predicting the probability of winning from the current state. Additionally, it employs Monte Carlo Tree Search (MCTS) to predict the best possible moves after running 300 simulations following each move. This document provides insights into how the board is captured, the policy vector structure and its utilization for move selection, MCTS traversal, training procedures, and a comprehensive overview of the model architecture.

## How the Model Reads the Board
The goal of processing the chessboard is to transform the physical game board into a format that the model can comprehend. With 6 distinct types of pieces in chess, the board is converted into a tensor with a shape of 9 x 8 x 8. The initial 6 features are dedicated to piece location, while the remaining 3 denote the player's turn. This transformation essentially creates an image with 9 features, unlike the typical RGB images with 3 features, and a size of 8 by 8. Each of the first 6 features corresponds to a piece type and its respective position on the board. Player turns are indicated by a value of 1 for the active player's pieces and -1 for the opponent's. The final 3 features signify whose turn it is; if White is to play next, these layers are filled with 1's, and if it's Black's turn, they're filled with -1's.

As a reference, for a starting chess board like this:

<p align="center">
  <img src="https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/afad76f7-7ae3-4f93-9198-3d45e5a55f41" width="450" alt="chessBoard">
  <br>
  <em>Figure 1: Starting Chess Board</em>
</p>

A visualization of the bit map would look like this:

<p align="center">
  <img src= "https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/ee5d36a6-4ced-4c7d-b8df-810f6aca4ab0" width="1200" alt="chessBoard">
  <br>
  <em>Figure 2: Visualization of Bit Map of Starting Chess Board</em>
</p>

For a randomized board like this:

<p align="center">
  <img src="https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/62857a22-438f-4894-a4e2-5c0da9aa7c58" width="450" alt="chessBoard">
  <br>
  <em>Figure 3: Randomized Board</em>
</p>

The visualization of the bit map of that board would look like this:

<p align="center">
  <img src= "https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/39cc65c3-ff34-4bd9-9e5d-fde155cc5e73" width="1200" alt="chessBoard">
  <br>
  <em>Figure 4: Visualization of Bit Map of Randomized Chess Board</em>
</p>


## The Policy: How to get a move from a policy

### Policy Structure
#### Non-pawn Promotional Moves
Designing a move-mapping structure for chess poses a unique challenge due to the nature of chess moves. Unlike games like Go, where each move involves simply placing a stone onto the board, chess moves require displacing a piece from its original position to a new one. Consequently, the policy for chess needs to incorporate an additional dimension to account for both the original and final positions of a piece for each move. A chessboard is 8 by 8, meaning there are 64 total squares on each board. To create a policy that incorporates a piece's original and final position, a policy vector of 4096 elements can be used to generate a move. A 4096-element policy vector (or 64 x 64 elements) means that for each square, there are 64 elements. This setup, however, does not consider pawn promotional moves.

<p align="center">
  <img src="https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/6a44b90d-74d0-4979-8c85-bdfee2c4632b" width="325" alt="chessBoard">
  <br>
  <em>Figure 5: Chess Board Layout</em>
</p>

#### Pawn Promotional Moves
Pawn promotion adds complexity to the policy because each pawn has the potential to promote to one of four different pieces. Therefore, the policy vector must be expanded to accommodate the possible promotions. With just 44 potential promotion moves per pawn (as illustrated in the diagram below), the policy vector requires an additional 176 elements to encompass all promotional possibilities. Consequently, normal moves are extracted from the initial 4096 elements, while promotional moves are derived from the final 176 elements, resulting in a total policy vector size of 4272 elements.

<p align = "center">
  <img src = "https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/f4769739-d6b0-43a3-9118-71d2723d543b" width = "325" alt = "promotionalChessBoard">
  <br>
  <em>Figure 6: Every Possible Pawn Promotional Move</em>
</p>

### How a move is selected
Due to the fact that the policy vector considers every possible moves in the chess domain, even moves that will never be played by any pieces, a majority of the elements in the policy vector will never be utilized. As a result, Before the best move is selected from the policy vector, every legal moves from a state is determined and used to apply a mask on the policy vector. Every non legal moves in the policy vector will be zerod out. Leaving the policy vector to only have non zero elements at index where legal moves are at. As a reference lets take a look at the randomized board again. It is currently white's turn to go:

<p align = "center">
  <img src = "https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/76850881-99fe-48a4-8477-a83d107ae293" width = "450" alt = "promotionalChessBoard">
  <br>
  <em>Figure 7: Every Legal Move of Current State</em>
</p>

There are 32 legal moves in this state, as a result, a mask will be applied to the rest of the elements in the policy. As a result, there will only be 32 non zero elements in the policy. To pick the next move, either sample from the non zero elements or pick the element with the highest probability. Map the index of that element to find the next move. lets say if the policy vector is as follows:


$$ [0, 0, .... , 0.46, ...., 0, 0] $$

Where the element with a probability of 0.46 percent has been selected and is at an index of 158 in the policy vector. Mapping this back, this means the move selected is c1f4 since they are on the 30th element of the 3rd 64 square board. In another word they are at

$$ 2 × 64 + 30 = 158 $$

Which means they are moving from the 3rd square to the 30th square in the board, or c1f4.

## Monte Carlo Tree Search Traversal

MCTS is a process used only during training that helps the model to predicts the best move to play. With Aldarion, MCTS has a small alteration from the normal MCTS traversal by instead of having the roll out step, Aldarion just uses its model to predict the value. There are 4 steps for every simulation of MCTS:

1. Selection: Starting from the root node of the search tree, traverse down the tree according to some selection policy until reaching a leaf node. The selection policy often balances exploration and exploitation to guide the search towards promising regions of the tree.
2. Expansion: Once a leaf node is reached, expand it by adding one or more child nodes representing possible future states or actions. These child nodes are typically generated based on legal moves or actions available from the current state. The probability of where each move is played is determined by passing the current state into the model.
3. Simulation: Differing from conventional simulation state, Aldarion simply passes this state into the model inorder to determine what the value of the state is.
4. Backpropagation: Update the statistics of all nodes traversed during the selection phase based on the outcome of the simulation. This involves incrementing visit counts and accumulating rewards or scores, which are used to guide future selections.

<p align = "center">
  <img src = "https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/c2786d7c-72d6-42fc-877d-3256f30164f1" width = "450" alt = "promotionalChessBoard">
  <br>
  <em>Figure 8: The Four steps of Monte Carlo Tree Search</em>
</p>

## Training Procedures

### Overview
One of the greatest advantage of Aldarion's AlphaZero is the ability to autogenerate data through repeated games of selfplay. Due to the nature of self generating data, there are two main aspects to the trianining process: Self-play to generate data, model training using the data generated from self play. These two training aspects are repeated over and over again until the training process is done. AlphaZero separates these two processes into two different threads and allow these two threads to run simultaneously. Aldarion simplifies this process by running training and self-play together on one thread, and the two processes will take turn to run one after another. In order for Aldarion to generate a diverse enough dataset, there will be two games of self play for each training task.

### Self-Play Data Collection
During games of self-play, every single move made by each team is recorded for training purposes. As a result, each game can generate roughly 100 to 200 training data points. For each data point, the state of the environment is recorded as a fen string. The fen string gives information to where each piece is currently positioned in the board and the team that is making the next book. As mentioned in MCTS, the purpose of using the tree during self-play is to abuse the branch that is most likely to result in a win; however, it would also allow the model to see which moves are the "right" move to make based on the number of times each child is visited from the current state. As a result, the result of the MCTS is also recorded as the "correct move distribution" to compare to the model's move distribution. Lastly, the score of the move is recorded for the value head to be trained. The score is determined based on the outcome of the game. The winning team will recieve a score of 1 for everysingle move that they made in the game, and -1 for every single move made by the losing team. If the game is a tie then every move for every team is scored a 0. For each move made, these items are recorded:

- State of the environment as a fen string
- MCTS move distribution
- Team that is playing the move
- Score of that move

### Training Mechanics
Training is a repeated process that periodically occurs as more data is generated from games of self-play. Data is extracted at random from the dataset to use to train the model. There are 3 majors components that are incorporated in the training process: The policy training, value training, and regularization. The loss function given is

$$ L = (z-v)^2 - πlog(p) + c||θ||^2 $$

The policy network is trained by comparing the policy predicted by the model to the policy determined from the Monte Carlo Tree Search through negative log likelihood. The value network is trained by comparing the predicted value from the model to the actual value determined from self-play. Regularization is done by taking the squared Euclidean norm of the weights of the model multiplied by a constant.

<p align = "center">
  <img src = "https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/c2d428c6-0d60-4134-98a9-ad68bbdeb126" width = "850" alt = "promotionalChessBoard">
  <br>
  <em>Figure 9: Aldarion's Training Procedure</em>
</p>

## Model Architecture

Aldarion is built from an actor-critic style architecture where it recieves the state as an input and have two outputs: the policy (actor), the value (critic). The policy makes the play, thus the actor, and the value judges that move, thus the critic. In order to get the exact code of the model, take a look at the model.py code script in this repository. Refer to the diagram below for a high level overview of the architecture:

<p align = "center">
  <img src = "https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/196c4999-0dda-48b5-95c3-4deeb850b065" width = "950" alt = "promotionalChessBoard">
  <br>
  <em>Figure 10: Aldarion Model's Architecture Overview</em>
</p>

There are 4 unique different elements in the model: the convolutional block, the residual block, the policy head, and the value head. The breakdown of their architectures are as follows:


<p align = "center">
  <img src = "https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/28a577cc-f67b-4bc9-8664-3270e383639d" width = "850" alt = "promotionalChessBoard">
  <br>
  <em>Figure 11: Model Main Features Overview</em>
</p>
