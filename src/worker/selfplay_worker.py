#!/usr/bin/env python3
"""
Self-Play Worker for Aldarion Chess Engine
Continuous worker that generates training data using the latest best model
"""

import os
import sys
import time
import logging
from datetime import datetime
import multiprocessing as mp
import torch

# Import Aldarion modules
from src.lib.model_manager import ModelManager
from src.lib.data_manager import DataManager
from src.training.parallel_utils import run_parallel_task_execution
from src.training.parallel_workers import selfplay_worker_process

logger = logging.getLogger(__name__)


def start_selfplay_worker(config):
    """
    Start the continuous self-play worker
    """
    mp.set_start_method('spawn', force=True)
    
    model_manager = ModelManager(config)
    data_manager = DataManager(config)
    sp_config = config.selfplay
    
    logger.info("Starting self-play worker")
    logger.info(f"Self-play configuration:")
    logger.info(f"Games per batch: {sp_config.games_per_batch}")
    logger.info(f"Simulations per move: {sp_config.simulation_num_per_move}")
    logger.info(f"Max processes: {sp_config.max_processes}")
    logger.info(f"Temperature decay: {sp_config.tau_decay_rate}")
    logger.info(f"C-PUCT: {sp_config.c_puct}\n")
    
    generation_count = 0
    
    try:
        while True:
            generation_count += 1
            logger.info(f"Starting self-play generation #{generation_count}")
            
            # Load the current best model
            best_model = model_manager.load_best_model()
            if best_model is None:
                logger.info("No best model available, creating initial model")
                best_model = model_manager.create_initial_best_model()
            
            # Save current model temporarily for workers to load
            temp_model_path = os.path.join(config.resource.model_dir, "best", "temp_selfplay_model.pth")
            torch.save(best_model.state_dict(), temp_model_path)
            
            # Generate self-play data
            start_time = time.time()
            training_data, process_stats = generate_selfplay_batch(
                model_path=temp_model_path,
                config=config
            )
            generation_time = time.time() - start_time
            
            # Clean up temporary model
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            
            if not training_data:
                logger.warning("No training data generated, retrying...")
                time.sleep(10)
                continue
            
            # Save training data using DataManager
            saved_file = data_manager.write_game_data_to_file(training_data)
            
            # Log generation summary
            total_games = sum(stats.get('tasks_completed', 0) for stats in process_stats)
            total_examples = len(training_data)
            
            # Calculate game outcome statistics
            white_wins = 0
            black_wins = 0
            draws = 0
            
            for stats in process_stats:
                game_outcomes = stats.get('game_outcomes', {})
                white_wins += game_outcomes.get('white_wins', 0)
                black_wins += game_outcomes.get('black_wins', 0)
                draws += game_outcomes.get('draws', 0)
            
            draw_rate = (draws / total_games * 100) if total_games > 0 else 0
            
            logger.info(f"Generation #{generation_count} complete:")
            logger.info(f"Games played: {total_games}")
            logger.info(f"Training examples: {total_examples:,}")
            logger.info(f"Game outcomes: W{white_wins} B{black_wins} D{draws} (draw rate: {draw_rate:.1f}%)")
            logger.info(f"Time: {generation_time:.1f}s ({generation_time/60:.1f}m)")
            logger.info(f"Data saved: {os.path.basename(saved_file)}")
            
            data_manager.cycle_old_data_if_needed()
            time.sleep(2)
            
    except KeyboardInterrupt:
        logger.info("Self-play worker stopped by user")
        return True
    except Exception as e:
        logger.error(f"Self-play worker failed: {e}")
        logger.exception("Self-play worker error details")
        return False


def generate_selfplay_batch(model_path, config):
    """
    Generate a batch of self-play games
    """
    sp_config = config.selfplay
    
    task_config = {
        'total_tasks': sp_config.games_per_batch,
        'num_simulations': sp_config.simulation_num_per_move,
        'temperature': 1.0,  # Start with high temperature
        'c_puct': sp_config.c_puct,
        'model_path': model_path,
        'tau_decay_rate': sp_config.tau_decay_rate,
        'noise_eps': sp_config.noise_eps,
        'dirichlet_alpha': sp_config.dirichlet_alpha,
        'max_game_length': sp_config.max_game_length,
        'seed': int(time.time()) % 10000  # Dynamic seed
    }
    
    # Calculate CPU utilization based on max_processes
    total_cpus = mp.cpu_count()
    cpu_utilization = min(0.9, sp_config.max_processes / total_cpus)
    training_data, process_statistics = run_parallel_task_execution(
        task_config=task_config,
        worker_function=selfplay_worker_process,
        cpu_utilization=cpu_utilization
    )
    
    return training_data, process_statistics