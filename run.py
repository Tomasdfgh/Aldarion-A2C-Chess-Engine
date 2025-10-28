#!/usr/bin/env python3
"""
Main entry point for Aldarion Chess Engine parallel training system
Modeled exactly after chess-alpha-zero's run.py

Usage:
    python run.py self [--type mini]     # Start self-play worker
    python run.py opt [--type mini]      # Start training worker  
    python run.py eval [--type mini]     # Start evaluation worker
"""
import sys
import argparse
import logging
from datetime import datetime


def setup_logging(log_level=logging.INFO, worker_type=None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]  # Console output
    
    # Add file handler if worker type specified
    if worker_type:
        log_filename = f"logs/{worker_type}.log"
        file_handler = logging.FileHandler(log_filename, mode='a')
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )


def load_config(config_type='normal'):
    """Load configuration based on type"""
    if config_type == 'mini':
        from src.config.mini import get_config
        print("🔧 Using mini configuration (fast testing)")
    elif config_type == 'normal':
        from src.config.normal import get_config
        print("🔧 Using normal configuration (production)")
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    return get_config()


def run_selfplay_worker(config):
    """Start the self-play worker"""
    setup_logging(worker_type='selfplay')
    print("🎮 Starting Self-Play Worker...")
    print("=" * 60)
    
    try:
        from src.worker.selfplay_worker import start_selfplay_worker
        start_selfplay_worker(config)
    except ImportError:
        print("❌ Self-play worker not implemented yet")
        print("📝 Run: Convert selfplay_generate_data.py to workers/selfplay_worker.py")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Self-play worker stopped by user")
        return True
    except Exception as e:
        print(f"❌ Self-play worker failed: {e}")
        logging.exception("Self-play worker error")
        return False


def run_training_worker(config):
    """Start the training worker"""
    setup_logging(worker_type='training')
    print("🧠 Starting Training Worker...")
    print("=" * 60)
    
    try:
        from src.worker.training_worker import start_training_worker
        start_training_worker(config)
    except ImportError:
        print("❌ Training worker not implemented yet")
        print("📝 Run: Convert train_model.py to workers/training_worker.py")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Training worker stopped by user")
        return True
    except Exception as e:
        print(f"❌ Training worker failed: {e}")
        logging.exception("Training worker error")
        return False


def run_evaluation_worker(config):
    """Start the evaluation worker"""
    setup_logging(worker_type='evaluation')
    print("⚖️ Starting Evaluation Worker...")
    print("=" * 60)
    
    try:
        from src.worker.evaluation_worker import start_evaluation_worker
        start_evaluation_worker(config)
    except ImportError:
        print("❌ Evaluation worker not implemented yet")
        print("📝 Run: Convert evaluate_models.py to workers/evaluation_worker.py")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Evaluation worker stopped by user")
        return True
    except Exception as e:
        print(f"❌ Evaluation worker failed: {e}")
        logging.exception("Evaluation worker error")
        return False


def show_status(config):
    """Show current system status"""
    print("📊 Aldarion System Status")
    print("=" * 60)
    
    # Check infrastructure
    from src.lib.model_manager import ModelManager
    from src.lib.data_manager import DataManager
    
    model_manager = ModelManager(config)
    data_manager = DataManager(config)
    
    # Best model status
    best_model = model_manager.load_best_model()
    if best_model:
        print("✅ Best model: Available")
    else:
        print("❌ Best model: Not found")
        print("💡 Run: python run.py init --type mini")
    
    # Next-generation models
    ng_dirs = model_manager.get_next_generation_model_dirs()
    print(f"📋 Next-generation models: {len(ng_dirs)}")
    
    # Training data
    total_datapoints = data_manager.get_total_datapoints()
    files = data_manager.get_game_data_filenames()
    print(f"📊 Training data: {total_datapoints} datapoints in {len(files)} files")
    
    # Configuration
    print(f"🔧 Configuration type: {getattr(config, 'config_type', 'normal')}")
    print(f"🏠 Project directory: {config.resource.project_dir}")
    
    print("\n🚀 To start workers:")
    print("   python run.py self --type mini    # Self-play")
    print("   python run.py opt --type mini     # Training")  
    print("   python run.py eval --type mini    # Evaluation")


def initialize_system(config):
    """Initialize the system with a random model"""
    print("🔧 Initializing Aldarion System...")
    print("=" * 60)
    
    from src.lib.model_manager import ModelManager
    
    model_manager = ModelManager(config)
    
    # Create initial best model if it doesn't exist
    best_model = model_manager.load_best_model()
    if best_model is None:
        print("Creating initial best model with random weights...")
        model_manager.create_initial_best_model()
        print("✅ Initial best model created")
    else:
        print("✅ Best model already exists")
    
    print("\n🎉 System initialized! Ready to start workers.")
    show_status(config)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Aldarion Chess Engine - Parallel Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py self --type mini     # Start self-play worker (testing)
    python run.py opt --type normal    # Start training worker (production)
    python run.py eval                 # Start evaluation worker (default: normal)
    python run.py status               # Show system status
    python run.py init --type mini     # Initialize system
        """
    )
    
    parser.add_argument(
        'command', 
        choices=['self', 'opt', 'eval', 'status', 'init'],
        help='Worker to start (self=self-play, opt=training, eval=evaluation)'
    )
    
    parser.add_argument(
        '--type', 
        choices=['mini', 'normal'], 
        default='normal',
        help='Configuration type (mini=testing, normal=production)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(getattr(logging, args.log_level))
    
    # Load configuration
    try:
        config = load_config(args.type)
        config.config_type = args.type  # Store for status display
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Display header
    print("🏛️  ALDARION CHESS ENGINE")
    print("   Parallel Training System")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Execute command
    success = True
    
    if args.command == 'self':
        success = run_selfplay_worker(config)
    elif args.command == 'opt':
        success = run_training_worker(config)
    elif args.command == 'eval':
        success = run_evaluation_worker(config)
    elif args.command == 'status':
        show_status(config)
    elif args.command == 'init':
        initialize_system(config)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())