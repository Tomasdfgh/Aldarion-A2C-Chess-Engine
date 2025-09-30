# Aldarion: Actor-Critic, Reinforcement Learning Chess Engine

![356070291-1238b477-6651-4c25-8f51-3549290ad56d](https://github.com/user-attachments/assets/4a488daa-3fcf-4bdf-b92b-101eabec0b58)



<p align="center">
  <img src="https://github.com/Tomasdfgh/Aldarion-A2C-Chess-Engine/assets/86145397/44381ed4-ac65-4c96-8513-901336e4223c" alt="Aldarion Chess Engine" width="300">
</p>

## Introduction

Welcome to Aldarion, a chess engine trained using an adapted AlphaZero algorithm. Aldarion leverages an actor-critic model, consisting of a policy head for determining the next move and a value head for predicting the probability of winning from the current state. Additionally, it employs Monte Carlo Tree Search (MCTS) to predict the best possible moves after running 300 simulations following each move. This document provides insights into how the board is captured, the policy vector structure and its utilization for move selection, MCTS traversal, training procedures, and a comprehensive overview of the model's architecture.

This project is still in development.
