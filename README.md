# Aldarion: Actor-Critic, Reinforcement Learning Chess Engine

<p align="center">
  <img src="https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/44381ed4-ac65-4c96-8513-901336e4223c" alt="Aldarion Chess Engine" width="200">
</p>

## Introduction

Welcome to Aldarion, a chess engine trained using an adapted AlphaZero algorithm. Aldarion leverages an actor-critic model, comprised of a policy head for determining the next move and a value head for predicting the probability of winning from the current state. Additionally, it employs Monte Carlo Tree Search (MCTS) to predict the best possible moves after running 300 simulations following each move. This document provides insights into the policy vector structure and its utilization for move selection, MCTS traversal, data storage mechanisms, training procedures, and a comprehensive overview of the model architecture.

## The Policy: How to get a move from a policy

### Policy Structure
#### Non-pawn Promotional Moves
Designing a move-mapping structure for chess poses a unique challenge due to the nature of chess moves. Unlike games like Go, where each move involves simply placing a stone onto the board, chess moves require displacing a piece from its original position to a new one. Consequently, the policy for chess needs to incorporate an additional dimension to account for both the original and final positions of a piece for each move. A Chess board is 8 by 8, that means there are 64 total squares in each board. To create a policy that incorporate a piece's original and final position, a policy vector of 4096 elements can be used to generate a move. A 4096 elements policy vector (or 64 x 64 elements) means that for each square, there are 64 elements. That means that each index in the policy means a specific move of a piece from one square to another square. This setup, however, does not consider pawn promotional moves.

<p align="center">
  <img src="https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/6a44b90d-74d0-4979-8c85-bdfee2c4632b" width="400" alt="chessBoard">
</p>

#### Pawn Promotional Moves
Pawn promotional moves are unique due to the fact that each pawn can be promoted to 4 different pieces. As a result, an extension to the policy has to be added in order to incorporate the consideration of which piece that pawn will be promoted to. Since the number of moves that each pawn can take to get promoted are limited to just 44 possible moves (diagram below), the policy vector will need to be extended by 176 more elements to consider all promotional possibilities with all promotional moves; therefore, each normal move will be extracted from the first 4096 elements of the policy vector and each promotional move will be extracted from the last 176 elements of the policy vector, making the policy vector to have a size of 4272 elements.

<p align = "center">
  <img src = "https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/f4769739-d6b0-43a3-9118-71d2723d543b" width = "400" alt = "promotionalChessBoard">
</p>

