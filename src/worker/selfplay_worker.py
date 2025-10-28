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

# Import Aldarion modules
from src.lib.model_manager import ModelManager
from src.lib.data_manager import DataManager
from src.training.parallel_utils import run_parallel_task_execution
from src.training.parallel_workers import selfplay_worker_process

logger = logging.getLogger(__name__)


def start_selfplay_worker(config):
    """
    Start the continuous self-play worker
    
    Args:
        config: Configuration object with selfplay, model, and resource settings
    """
    # Set multiprocessing start method for CUDA compatibility
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    logger.info("Starting self-play worker")
    
    model_manager = ModelManager(config)
    data_manager = DataManager(config)
    
    # Self-play configuration
    sp_config = config.selfplay
    
    logger.info(f"Self-play configuration:")
    logger.info(f"  Games per batch: {sp_config.games_per_batch}")
    logger.info(f"  Simulations per move: {sp_config.simulation_num_per_move}")
    logger.info(f"  Max processes: {sp_config.max_processes}")
    logger.info(f"  Temperature decay: {sp_config.tau_decay_rate}")
    logger.info(f"  C-PUCT: {sp_config.c_puct}")
    
    generation_count = 0
    
    try:
        while True:
            generation_count += 1
            logger.info(f"🎮 Starting self-play generation #{generation_count}")
            
            # Load the current best model
            best_model = model_manager.load_best_model()
            if best_model is None:
                logger.error("No best model available, cannot generate self-play data")
                logger.info("Run: python run.py init --type mini")
                time.sleep(30)
                continue
            
            # Save current model temporarily for workers to load
            temp_model_path = os.path.join(config.resource.model_dir, "temp_selfplay_model.pth")
            model_manager.save_model_weights(best_model, temp_model_path)
            
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
            
            logger.info(f"✅ Generation #{generation_count} complete:")
            logger.info(f"   Games played: {total_games}")
            logger.info(f"   Training examples: {total_examples:,}")
            logger.info(f"   Time: {generation_time:.1f}s ({generation_time/60:.1f}m)")
            logger.info(f"   Data saved: {os.path.basename(saved_file)}")
            
            # Check and cycle old data if needed
            data_manager.cycle_old_data_if_needed()
            
            # Brief pause before next generation
            time.sleep(5)
            
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
    
    Args:
        model_path: Path to model weights file
        config: Configuration object
        
    Returns:
        tuple: (training_data, process_statistics)
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
        'resign_threshold': sp_config.resign_threshold,
        'min_resign_turn': sp_config.min_resign_turn,
        'max_game_length': sp_config.max_game_length,
        'seed': int(time.time()) % 10000  # Dynamic seed
    }
    
    # Calculate CPU utilization based on max_processes
    import multiprocessing as mp
    total_cpus = mp.cpu_count()
    cpu_utilization = min(0.9, sp_config.max_processes / total_cpus)
    
    logger.debug(f"Generating {sp_config.games_per_batch} games with {sp_config.simulation_num_per_move} simulations per move")
    
    training_data, process_statistics = run_parallel_task_execution(
        task_config=task_config,
        worker_function=selfplay_worker_process,
        cpu_utilization=cpu_utilization
    )
    
    return training_data, process_statistics


def save_model_weights(model, path):
    """
    Standalone function to save model weights
    
    Args:
        model: PyTorch model
        path: Path to save weights
    """
    import torch
    torch.save(model.state_dict(), path)


# Add the missing save_model_weights method to ModelManager for compatibility
def patch_model_manager():
    """Add save_model_weights method to ModelManager if not present"""
    if not hasattr(ModelManager, 'save_model_weights'):
        ModelManager.save_model_weights = lambda self, model, path: save_model_weights(model, path)

# Apply the patch
patch_model_manager()


if __name__ == "__main__":
    # For testing the worker directly
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    # Load mini config for testing
    from src.config.mini import get_config
    config = get_config()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    start_selfplay_worker(config)