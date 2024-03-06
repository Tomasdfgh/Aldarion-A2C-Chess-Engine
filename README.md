# Aldarion: Actor-Critic, Reinforcement Learning Chess Engine

<p align="center">
  <img src="https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/44381ed4-ac65-4c96-8513-901336e4223c" alt="Aldarion Chess Engine" width="300">
</p>

## Introduction

Welcome to Aldarion, a chess engine trained using an adapted AlphaZero algorithm. Aldarion leverages an actor-critic model, consisting of a policy head for determining the next move and a value head for predicting the probability of winning from the current state. Additionally, it employs Monte Carlo Tree Search (MCTS) to predict the best possible moves after running 300 simulations following each move. This document provides insights into how the board is captured, the policy vector structure and its utilization for move selection, MCTS traversal, data storage mechanisms, training procedures, and a comprehensive overview of the model architecture.

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
Due to the fact that the policy vector considers every possible moves in the chess domain, even moves that will never be played by any pieces, a majority of the elements in the policy vector will never be utilized. As a result, Before the best move is selected from the policy vector, every legal moves from a state is determined and used to apply a mask on the policy vector.
