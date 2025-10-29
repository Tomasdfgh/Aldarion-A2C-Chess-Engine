#!/usr/bin/env python3
"""
Main entry point for Aldarion Chess Engine parallel training system
"""
import sys
import argparse
import logging
import os

#Local Imports
from src.worker.selfplay_worker import start_selfplay_worker
from src.worker.training_worker import start_training_worker
from src.worker.evaluation_worker import start_evaluation_worker
from src.lib.model_manager import ModelManager
from src.lib.data_manager import DataManager
from .config import Config


def setup_logging(log_level=logging.INFO, worker_type=None, config=None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    
    # Add file handler if worker type specified
    if worker_type and config:
        log_filename = os.path.join(config.resource.log_dir, f"{worker_type}.log")
        file_handler = logging.FileHandler(log_filename, mode='a')
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )

def show_status(config):
    print("Aldarion System Status")
    print("=" * 60)
    
    model_manager = ModelManager(config)
    data_manager = DataManager(config)
    
    # Best model status
    best_model = model_manager.load_best_model()
    if best_model:
        print("Best model: Available")
    else:
        print("Best model: Not found")
        print("Run: python run.py init")
    
    print(f"Next-generation models: {len(model_manager.get_next_generation_model_dirs())}")
    print(f"Training data: {data_manager.get_total_datapoints()} datapoints in {len(data_manager.get_game_data_filenames())} files")
    print(f"Project directory: {config.resource.project_dir}")
    
    print("\nTo start workers:")
    print("   python run.py self     # Self-play")
    print("   python run.py opt      # Training")  
    print("   python run.py eval     # Evaluation")


def initialize_system(config):
    """Initialize the system with a random model"""
    print("Initializing Aldarion System")
    print("=" * 60)
    
    model_manager = ModelManager(config)
    model_manager.create_initial_best_model()
    
    print("\nSystem initialized! Ready to start workers.")
    show_status(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['self', 'opt', 'eval', 'status', 'init'])
    args = parser.parse_args()
    config = Config()
    
    if args.command == 'self':     
        setup_logging(log_level=logging.INFO, worker_type='selfplay', config=config)
        print("Starting Self-Play Worker")
        print("=" * 60)
        start_selfplay_worker(config)
        
    elif args.command == 'opt':
        setup_logging(log_level=logging.INFO, worker_type='training', config=config)
        print("Starting Training Worker")
        print("=" * 60)
        start_training_worker(config)

    elif args.command == 'eval': 
        setup_logging(log_level=logging.INFO, worker_type='evaluation', config=config)
        print("Starting Evaluation Worker")
        print("=" * 60)
        start_evaluation_worker(config)

    elif args.command == 'status': show_status(config)
    elif args.command == 'init': initialize_system(config)
    
    return None


if __name__ == "__main__":
    sys.exit(main())