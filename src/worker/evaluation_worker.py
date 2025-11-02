#!/usr/bin/env python3
"""
Evaluation Worker for Aldarion Chess Engine
Continuous worker that evaluates next-generation models against the best model
"""

import os
import time
import logging
import multiprocessing as mp

# Import Aldarion modules
from src.lib.model_manager import ModelManager
from src.training.parallel_utils import run_parallel_task_execution
from src.training.parallel_workers import evaluation_worker_process

logger = logging.getLogger(__name__)


def start_evaluation_worker(config):
    """
    Start the continuous evaluation worker
    """
    
    mp.set_start_method('spawn', force=True)
    model_manager = ModelManager(config)
    eval_config = config.eval
    
    logger.info(f" ")
    logger.info("Starting evaluation worker")
    logger.info(f"Evaluation configuration:")
    logger.info(f"Games per evaluation: {eval_config.game_num}")
    logger.info(f"Replace rate threshold: {eval_config.replace_rate}")
    logger.info(f"Max game length: {eval_config.max_game_length}")
    logger.info(f"Simulations per move: {eval_config.simulation_num_per_move}")
    
    evaluation_count = 0
    
    try:
        while True:

            # Check for next-generation models to evaluate
            ng_model_dirs = model_manager.get_next_generation_model_dirs()
            if not ng_model_dirs:
                logger.info(f"Can't find next candidate model")
                time.sleep(30)
                continue
            
            # Reverse to evaluate oldest models first (FIFO)
            ng_model_dirs.reverse()
            
            # Load best model
            best_model_path = config.resource.model_best_weight_path
            if not os.path.exists(best_model_path):
                logger.info(f"Can't find best model")
                time.sleep(30)
                continue
            
            evaluation_count += 1
            logger.info(f" ")
            logger.info(f"Starting evaluation #{evaluation_count}")
            
            # Evaluate models (newest first if configured)
            if eval_config.evaluate_latest_first:
                ng_model_dirs = ng_model_dirs[:1]  # Only evaluate latest
            
            for model_dir in ng_model_dirs:
                model_name = os.path.basename(model_dir)
                ng_model_path = os.path.join(model_dir, config.resource.next_generation_model_weight_filename)
                
                if not os.path.exists(ng_model_path):
                    logger.warning(f"Model weights not found: {ng_model_path}")
                    continue
                
                logger.info(f" ")
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
                
                logger.info(f"Evaluation complete for {model_name}:")
                logger.info(f"Games played: {results['total_games']}")
                logger.info(f"Score rate: {score_rate:.1f}%")
                logger.info(f"New model wins: {results['new_model_wins']}")
                logger.info(f"Old model wins: {results['old_model_wins']}")
                logger.info(f"Draws: {results['draws']}")
                logger.info(f"Evaluation time: {evaluation_time:.1f}s ({evaluation_time/60:.1f}m)")
                
                threshold = eval_config.replace_rate * 100
                if score_rate > threshold:
                    logger.info(f"=====PROMOTING MODEL: {model_name}=====")
                    logger.info(f"Score rate {score_rate:.1f}% > {threshold:.1f}% threshold")
                    
                    # Load and promote the new model
                    new_model = model_manager.load_next_generation_model(model_dir)
                    if new_model:
                        model_manager.save_as_best_model(new_model)
                        logger.info(f"{model_name} is now the best model")
                        
                        # Archive the old candidate model
                        model_manager.archive_model(model_dir)
                        logger.info(f"{model_name} archived")
                    else:
                        logger.error(f"Failed to load model {model_name} for promotion")

                else:
                    logger.info(f"=-=-=REJECTING MODEL: {model_name}=-=-=")
                    logger.info(f"Score rate {score_rate:.1f}% <= {threshold:.1f}% threshold")
                    
                    # Archive the failed candidate model
                    model_manager.archive_model(model_dir)
                    logger.info(f"{model_name} archived")
                
                # Clean up old archives periodically
                if evaluation_count % 10 == 0:
                    removed = model_manager.cleanup_old_archives(max_archives=10)
                    if removed > 0:
                        logger.info(f"Cleaned up {removed} old archived models")
                
                time.sleep(5)
            
            time.sleep(30)
            
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
    """
    eval_config = config.eval
    
    task_config = {
        'total_tasks': eval_config.game_num,
        'num_simulations': eval_config.simulation_num_per_move,
        'old_model_path': old_model_path,
        'new_model_path': new_model_path,
        'max_game_length': eval_config.max_game_length,
        'c_puct': eval_config.c_puct,
        'tau_decay_rate': eval_config.tau_decay_rate,
        'noise_eps': eval_config.noise_eps,
        'seed': int(time.time()) % 10000  # Dynamic seed
    }
    total_cpus = mp.cpu_count()
    cpu_utilization = min(0.9, eval_config.max_processes / total_cpus)
    
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