# Aldarion: Actor-Critic, Reinforcement Learning Chess Engine

<p align="center">
  <img src="https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/44381ed4-ac65-4c96-8513-901336e4223c" alt="Aldarion Chess Engine" width="200">
</p>



## Introduction

Welcome to Aldarion, a chess engine trained using an adapted AlphaZero algorithm. Aldarion leverages an actor-critic model, comprising a policy head for determining the next move and a value head for predicting the probability of winning from the current state. Additionally, it employs Monte Carlo Tree Search (MCTS) to predict the best possible moves after running 300 simulations following each move. This document provides insights into the policy vector structure and its utilization for move selection, MCTS traversal, data storage mechanisms, training procedures, and a comprehensive overview of the model architecture.

## The Policy: How to get a move from a policy

### Policy Structure
#### Non-pawn Promotional Moves
Designing a move-mapping structure for chess poses a unique challenge due to the nature of chess moves. Unlike games like Go, where each move involves simply placing a stone onto the board, chess moves require displacing a piece from its original position to a new one. Consequently, the policy for chess needs to incorporate an additional dimension to account for both the original and final positions of a piece for each move. A Chess board is 8 by 8, that means there are 64 total squares in each board. To create a policy that incorporate a piece's original and final position, a policy vector of 4096 elements can be used to generate a move. A 4096 elements policy vector (or 64 x 64 elements) means that for each square, there are 64 elements. That means that each index in the policy means a specific move of a piece from one square to another square. This setup, however, does not consider pawn promotional moves.

#### Pawn Promotional Moves

This .readme page is still be developed
