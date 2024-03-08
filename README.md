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
  <img src = "https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/ce2e2f27-06ac-4fe3-9800-2426f4a9d36d" width = "450" alt = "promotionalChessBoard">
  <br>
  <em>Figure 8: The Four steps of Monte Carlo Tree Search</em>
</p>
