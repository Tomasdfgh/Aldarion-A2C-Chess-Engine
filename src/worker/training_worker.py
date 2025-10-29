#!/usr/bin/env python3
"""
Training Worker for Aldarion Chess Engine
Continuous worker that trains models on the latest training data
"""

import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import chess
from datetime import datetime
import numpy as np
import random
import multiprocessing as mp

# Import Aldarion modules
from src.lib.model_manager import ModelManager
from src.lib.data_manager import DataManager
from src.agent import model as md
from src.agent import board as br
from config import Config

logger = logging.getLogger(__name__)


class ChessTrainingDataset(Dataset):
    """PyTorch Dataset for chess training data"""
    
    def __init__(self, data_files):
        self.training_data = []
        
        for data_file in data_files:
            if not os.path.exists(data_file):
                logger.warning(f"Training data file not found: {data_file}")
                continue

            try:
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                self.training_data.extend(data)
                logger.debug(f"Loaded {len(data)} examples from {os.path.basename(data_file)}")
            except Exception as e:
                logger.error(f"Error loading {data_file}: {e}")
                continue
    
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        board_fen, history_fens, move_probs, game_outcome = self.training_data[idx]
        
        # Reconstruct game history
        game_history = []
        for fen in history_fens:
            game_history.append(chess.Board(fen))
        
        current_board = chess.Board(board_fen)
        board_tensor = br.board_to_full_alphazero_input(current_board, game_history)

        # Convert move probabilities to policy vector
        policy_vector = torch.zeros((8, 8, 73), dtype=torch.float32)
        for move, prob in move_probs.items():
            try:
                r, c, pl = br.uci_to_policy_index(str(move), current_board.turn)
                policy_vector[r, c, pl] = float(prob)
            except:
                continue
        policy_vector = policy_vector.reshape(-1)
        
        # Normalize policy
        s = policy_vector.sum()
        if s > 0:
            policy_vector /= s
        
        # Create legal move mask
        legal_mask = br.create_legal_move_mask(chess.Board(board_fen)).flatten()
        if not legal_mask.any():
            return None
            
        value_target = torch.tensor([game_outcome], dtype=torch.float32)
        
        return board_tensor.float(), policy_vector, legal_mask, value_target


def collate_fn(batch):
    """Custom collate function to handle None items"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
    
    return torch.utils.data.dataloader.default_collate(batch)


def compute_loss(model_output, targets, legal_masks, loss_weights):
    """
    Compute AlphaZero loss with proper legal move masking
    
    Args:
        model_output: (policy_logits, value_pred)
        targets: (target_policy, target_value)
        legal_masks: Legal move masks
        loss_weights: [policy_weight, value_weight]
    """
    policy_logits, value_pred = model_output
    target_policy, target_value = targets
    
    # Mask illegal moves
    masked_logits = policy_logits.clone()
    masked_logits[~legal_masks.bool()] = -1000.0
    log_probs = F.log_softmax(masked_logits, dim=1)
    
    # Normalize target policy
    target_sum = target_policy.sum(dim=1, keepdim=True)
    normalized_target = target_policy / (target_sum + 1e-8)
    
    # Policy loss: cross-entropy
    policy_loss = -(normalized_target * log_probs).sum(dim=1).mean()
    
    # Value loss: MSE
    value_loss = nn.MSELoss()(value_pred.squeeze(), target_value.squeeze())
    
    # Weighted total loss
    total_loss = loss_weights[0] * policy_loss + loss_weights[1] * value_loss
    
    return {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss
    }


def train_epoch(model, dataloader, optimizer, device, loss_weights):
    """Train model for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    
    for batch_idx, (board_tensors, target_policies, legal_masks, target_values) in enumerate(dataloader):
        if len(board_tensors) == 0:
            continue
            
        board_tensors = board_tensors.to(device)
        target_policies = target_policies.to(device)
        legal_masks = legal_masks.to(device)
        target_values = target_values.to(device)
        
        optimizer.zero_grad()
        policy_logits, value_pred = model(board_tensors)
        
        losses = compute_loss(
            (policy_logits, value_pred), 
            (target_policies, target_values),
            legal_masks,
            loss_weights
        )
        
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        
        total_loss += losses['total_loss'].item()
        total_policy_loss += losses['policy_loss'].item()
        total_value_loss += losses['value_loss'].item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            logger.debug(f"Batch {batch_idx}/{len(dataloader)}: "
                        f"Loss={losses['total_loss'].item():.4f}")
    
    return {
        'avg_total_loss': total_loss / num_batches if num_batches > 0 else 0,
        'avg_policy_loss': total_policy_loss / num_batches if num_batches > 0 else 0,
        'avg_value_loss': total_value_loss / num_batches if num_batches > 0 else 0
    }


def start_training_worker(config):
    """
    Start the continuous training worker
    """
    mp.set_start_method('spawn', force=True)
    
    logger.info("Starting training worker")
    
    model_manager = ModelManager(config)
    data_manager = DataManager(config)
    
    # Training configuration
    tr_config = config.trainer
    logger.info(f"Training configuration:")
    logger.info(f"Batch size: {tr_config.batch_size}")
    logger.info(f"Dataset size: {tr_config.dataset_size}")
    logger.info(f"Loss weights: {tr_config.loss_weights}")
    logger.info(f"Load data steps: {tr_config.load_data_steps}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}\n")
    
    step_count = tr_config.start_total_steps
    dataloader = None
    
    try:
        while True:
            # Check if we have enough training data
            total_datapoints = data_manager.get_total_datapoints()
            if total_datapoints < tr_config.min_data_size_to_learn:
                time.sleep(30)
                continue
            
            # Check candidate pool size (limit to 20 models)
            candidate_models = model_manager.get_next_generation_model_dirs()
            if len(candidate_models) >= 20:
                time.sleep(30)
                continue

            #Check if there are any training data
            training_files = data_manager.get_game_data_filenames()
            if not training_files:
                time.sleep(30)
                continue
            
            step_count += 1
            
            # Load training data every N steps or if no dataloader exists
            if step_count % tr_config.load_data_steps == 1 or dataloader is None:
                
                logger.info(f"Starting training step #{step_count}")
                logger.info(f"Loading training data from {len(training_files)} files")
                dataset = ChessTrainingDataset(training_files)
                
                if len(dataset) == 0:
                    logger.debug("No training examples loaded, waiting...")
                    time.sleep(30)
                    step_count -= 1
                    continue
                
                dataloader = DataLoader(
                    dataset,
                    batch_size = tr_config.batch_size,
                    shuffle = True,
                    num_workers = 4,
                    pin_memory = True if device.type == 'cuda' else False,
                    collate_fn = collate_fn,
                    drop_last = True
                )
                
                logger.info(f"Loaded {len(dataset):,} training examples")
            
            # Skip training if no dataloader available
            if dataloader is None:
                step_count -= 1
                time.sleep(30)
                continue
            
            # Load current best model
            model = model_manager.load_best_model()
            if model is None:
                step_count -= 1
                time.sleep(30)
                continue
            
            model = model.to(device)
            
            # Setup optimizer
            optimizer = optim.SGD(
                model.parameters(),
                lr= config.trainer.lr,
                momentum= config.trainer.momentum,
                weight_decay= config.trainer.weight_decay
            )
            
            # Train for one epoch
            start_time = time.time()
            metrics = train_epoch(model, dataloader, optimizer, device, tr_config.loss_weights)
            training_time = time.time() - start_time
            
            # Save as next-generation model
            model_id = f"step_{step_count:06d}"
            model_dir = model_manager.save_next_generation_model(model, model_id)
            
            logger.info(f"Training step #{step_count} complete:")
            logger.info(f"Total loss: {metrics['avg_total_loss']:.4f}")
            logger.info(f"Policy loss: {metrics['avg_policy_loss']:.4f}")
            logger.info(f"Value loss: {metrics['avg_value_loss']:.4f}")
            logger.info(f"Training examples: {len(dataset):,}")
            logger.info(f"Training time: {training_time:.1f}s")
            logger.info(f"Model saved: {os.path.basename(model_dir)}\n")
            
            # Brief pause before next training step
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Training worker stopped by user")
        return True
    except Exception as e:
        logger.error(f"Training worker failed: {e}")
        logger.exception("Training worker error details")
        return False