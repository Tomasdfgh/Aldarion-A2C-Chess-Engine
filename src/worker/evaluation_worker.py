#!/usr/bin/env python3
"""
Evaluation Worker for Aldarion Chess Engine
Continuous worker that evaluates next-generation models against the best model
"""

import os
import sys
import time
import logging
from datetime import datetime

# Import Aldarion modules
from src.lib.model_manager import ModelManager
from src.training.parallel_utils import run_parallel_task_execution
from src.training.parallel_workers import evaluation_worker_process

logger = logging.getLogger(__name__)


def start_evaluation_worker(config):
    """
    Start the continuous evaluation worker
    
    Args:
        config: Configuration object with eval, model, and resource settings
    """
    # Set multiprocessing start method for CUDA compatibility
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    logger.info("Starting evaluation worker")
    
    model_manager = ModelManager(config)
    
    # Evaluation configuration
    eval_config = config.eval
    
    logger.info(f"Evaluation configuration:")
    logger.info(f"  Games per evaluation: {eval_config.game_num}")
    logger.info(f"  Replace rate threshold: {eval_config.replace_rate}")
    logger.info(f"  Max game length: {eval_config.max_game_length}")
    logger.info(f"  Simulations per move: {eval_config.play_config.simulation_num_per_move}")
    
    evaluation_count = 0
    
    try:
        while True:
            # Check for next-generation models to evaluate
            ng_model_dirs = model_manager.get_next_generation_model_dirs()
            if not ng_model_dirs:
                logger.debug("No next-generation models to evaluate, waiting...")
                time.sleep(30)
                continue
            
            evaluation_count += 1
            logger.info(f"⚖️ Starting evaluation #{evaluation_count}")
            
            # Load best model
            best_model_path = config.resource.model_best_weight_path
            if not os.path.exists(best_model_path):
                logger.error("No best model found for evaluation")
                time.sleep(30)
                continue
            
            # Evaluate models (newest first if configured)
            if eval_config.evaluate_latest_first:
                ng_model_dirs = ng_model_dirs[:1]  # Only evaluate latest
            
            for model_dir in ng_model_dirs:
                model_name = os.path.basename(model_dir)
                ng_model_path = os.path.join(model_dir, config.resource.next_generation_model_weight_filename)
                
                if not os.path.exists(ng_model_path):
                    logger.warning(f"Model weights not found: {ng_model_path}")
                    continue
                
                logger.info(f"Evaluating {model_name} vs best model")
                
                # Run evaluation
                start_time = time.time()
                results = evaluate_models(
                    old_model_path=best_model_path,
                    new_model_path=ng_model_path,
                    config=config
                )
                evaluation_time = time.time() - start_time
                
                if results is None:
                    logger.error(f"Evaluation failed for {model_name}")
                    continue
                
                # Calculate score rate (wins + 0.5 * draws)
                score_rate = results['score_rate']
                
                logger.info(f"✅ Evaluation complete for {model_name}:")
                logger.info(f"   Games played: {results['total_games']}")
                logger.info(f"   Score rate: {score_rate:.1f}%")
                logger.info(f"   New model wins: {results['new_model_wins']}")
                logger.info(f"   Old model wins: {results['old_model_wins']}")
                logger.info(f"   Draws: {results['draws']}")
                logger.info(f"   Evaluation time: {evaluation_time:.1f}s ({evaluation_time/60:.1f}m)")
                
                # Decide whether to promote the model
                threshold = eval_config.replace_rate * 100  # Convert to percentage
                
                if score_rate > threshold:
                    logger.info(f"🎉 PROMOTING MODEL: {model_name}")
                    logger.info(f"   Score rate {score_rate:.1f}% > {threshold:.1f}% threshold")
                    
                    # Load and promote the new model
                    new_model = model_manager.load_next_generation_model(model_dir)
                    if new_model:
                        model_manager.save_as_best_model(new_model)
                        logger.info(f"   {model_name} is now the best model")
                        
                        # Archive the old candidate model
                        model_manager.archive_model(model_dir)
                        logger.info(f"   {model_name} archived")
                    else:
                        logger.error(f"Failed to load model {model_name} for promotion")
                else:
                    logger.info(f"❌ REJECTING MODEL: {model_name}")
                    logger.info(f"   Score rate {score_rate:.1f}% <= {threshold:.1f}% threshold")
                    
                    # Archive the failed candidate model
                    model_manager.archive_model(model_dir)
                    logger.info(f"   {model_name} archived")
                
                # Clean up old archives periodically
                if evaluation_count % 10 == 0:
                    removed = model_manager.cleanup_old_archives(max_archives=50)
                    if removed > 0:
                        logger.info(f"Cleaned up {removed} old archived models")
                
                # Brief pause between evaluations
                time.sleep(5)
            
            # Longer pause before checking for new models
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Evaluation worker stopped by user")
        return True
    except Exception as e:
        logger.error(f"Evaluation worker failed: {e}")
        logger.exception("Evaluation worker error details")
        return False


def evaluate_models(old_model_path, new_model_path, config):
    """
    Evaluate two models against each other
    
    Args:
        old_model_path: Path to old (best) model weights
        new_model_path: Path to new (candidate) model weights
        config: Configuration object
        
    Returns:
        dict: Evaluation results or None if failed
    """
    eval_config = config.eval
    
    task_config = {
        'total_tasks': eval_config.game_num,
        'num_simulations': eval_config.play_config.simulation_num_per_move,
        'old_model_path': old_model_path,
        'new_model_path': new_model_path,
        'max_game_length': eval_config.max_game_length,
        'c_puct': eval_config.play_config.c_puct,
        'tau_decay_rate': eval_config.play_config.tau_decay_rate,
        'noise_eps': eval_config.play_config.noise_eps,
        'seed': int(time.time()) % 10000  # Dynamic seed
    }
    
    # Calculate CPU utilization based on max_processes
    import multiprocessing as mp
    total_cpus = mp.cpu_count()
    cpu_utilization = min(0.9, eval_config.play_config.max_processes / total_cpus)
    
    logger.debug(f"Running {eval_config.game_num} evaluation games")
    
    try:
        game_results, process_statistics = run_parallel_task_execution(
            task_config=task_config,
            worker_function=evaluation_worker_process,
            cpu_utilization=cpu_utilization
        )
        
        if not game_results:
            logger.error("No evaluation games completed")
            return None
        
        # Process results
        successful_games = [r for r in game_results if 'error' not in r]
        failed_games = [r for r in game_results if 'error' in r]
        
        if len(failed_games) > 0:
            logger.warning(f"{len(failed_games)} evaluation games failed")
        
        if len(successful_games) == 0:
            logger.error("No successful evaluation games")
            return None
        
        # Count results from new model's perspective
        new_model_wins = 0
        old_model_wins = 0
        draws = 0
        
        for game in successful_games:
            result = game['result']
            white_is_new = game['white_is_new']
            
            if result == 1.0:  # White wins
                if white_is_new:
                    new_model_wins += 1
                else:
                    old_model_wins += 1
            elif result == -1.0:  # Black wins
                if not white_is_new:  # Black is new model
                    new_model_wins += 1
                else:
                    old_model_wins += 1
            else:  # Draw
                draws += 1
        
        total_games_played = len(successful_games)
        
        # Calculate score-based win rate (AlphaZero style: win=1.0, draw=0.5, loss=0.0)
        new_model_score = new_model_wins + 0.5 * draws
        score_rate = new_model_score / total_games_played * 100
        
        return {
            'new_model_wins': new_model_wins,
            'old_model_wins': old_model_wins,
            'draws': draws,
            'total_games': total_games_played,
            'score_rate': score_rate,
            'new_model_score': new_model_score,
            'successful_games': successful_games,
            'failed_games': failed_games,
            'process_statistics': process_statistics
        }
        
    except Exception as e:
        logger.error(f"Evaluation execution failed: {e}")
        logger.exception("Evaluation error details")
        return None


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
    
    start_evaluation_worker(config)