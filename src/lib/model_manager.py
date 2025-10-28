"""
Model Management System for Aldarion Chess Engine
Handles "best model" vs "next-generation models" like chess-alpha-zero
"""
import os
import json
import torch
import shutil
import glob
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages best model and next-generation model lifecycle"""
    
    def __init__(self, config):
        self.config = config
        self.resource = config.resource
        
    def load_best_model(self):
        """
        Load the current best model
        
        Returns:
            ChessNet model or None if no best model exists
        """
        from src.agent.model import ChessNet  # Import your model
        
        config_path = self.resource.model_best_config_path
        weight_path = self.resource.model_best_weight_path
        
        if not (os.path.exists(config_path) and os.path.exists(weight_path)):
            logger.debug("No best model found")
            return None
        
        try:
            # Load model config
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            
            # Create model
            model = ChessNet()
            
            # Load weights
            model.load_state_dict(torch.load(weight_path, map_location='cpu'))
            
            logger.info(f"Loaded best model from {os.path.basename(weight_path)}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load best model: {e}")
            return None
    
    def save_as_best_model(self, model):
        """
        Save a model as the new best model, backing up the previous best model
        
        Args:
            model: ChessNet model to save
        """
        config_path = self.resource.model_best_config_path
        weight_path = self.resource.model_best_weight_path
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Backup existing best model before overwriting
            if os.path.exists(weight_path):
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")[:-3]
                backup_dir = os.path.join(self.resource.next_generation_model_dir, "copies", f"best_model_dethroned_{timestamp}")
                os.makedirs(backup_dir, exist_ok=True)
                
                # Copy current best model to backup
                import shutil
                backup_config = os.path.join(backup_dir, "model_config.json")
                backup_weight = os.path.join(backup_dir, "model_weight.pth")
                
                if os.path.exists(config_path):
                    shutil.copy2(config_path, backup_config)
                shutil.copy2(weight_path, backup_weight)
                
                logger.info(f"Backed up dethroned best model to {os.path.basename(backup_dir)}")
            
            # Save model config (basic info about architecture)
            model_config = {
                'input_channels': 119,
                'residual_blocks': 7,
                'filters': 256,
                'policy_size': 4672,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            # Save model weights
            torch.save(model.state_dict(), weight_path)
            
            logger.info(f"Saved new best model to {os.path.basename(weight_path)}")
            
        except Exception as e:
            logger.error(f"Failed to save best model: {e}")
            raise
    
    def create_initial_best_model(self):
        """
        Create an initial best model with random weights if none exists
        
        Returns:
            ChessNet model
        """
        from src.agent.model import ChessNet
        
        if self.load_best_model() is not None:
            logger.info("Best model already exists, not creating new one")
            return self.load_best_model()
        
        logger.info("Creating initial best model with random weights")
        
        model = ChessNet()
        self.save_as_best_model(model)
        
        return model
    
    def save_next_generation_model(self, model, model_id=None):
        """
        Save a model as a next-generation candidate
        
        Args:
            model: ChessNet model to save
            model_id: Optional model ID, auto-generated if None
            
        Returns:
            Path to saved model directory
        """
        if model_id is None:
            model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")[:-3]  # Include milliseconds
        
        model_dir_name = self.resource.next_generation_model_dirname_tmpl % model_id
        model_dir = os.path.join(self.resource.next_generation_model_dir, model_dir_name)
        
        try:
            # Create model directory
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model config
            config_path = os.path.join(model_dir, self.resource.next_generation_model_config_filename)
            model_config = {
                'input_channels': 119,
                'residual_blocks': 7,
                'filters': 256,
                'policy_size': 4672,
                'model_id': model_id,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            # Save model weights
            weight_path = os.path.join(model_dir, self.resource.next_generation_model_weight_filename)
            torch.save(model.state_dict(), weight_path)
            
            logger.info(f"Saved next-generation model {model_id}")
            return model_dir
            
        except Exception as e:
            logger.error(f"Failed to save next-generation model: {e}")
            raise
    
    def get_next_generation_model_dirs(self):
        """
        Get list of next-generation model directories, sorted by creation time
        
        Returns:
            List of directory paths, newest first
        """
        pattern = os.path.join(self.resource.next_generation_model_dir, "model_*")
        dirs = glob.glob(pattern)
        
        # Filter out the copies directory
        dirs = [d for d in dirs if os.path.isdir(d) and not d.endswith('copies')]
        
        # Sort by modification time, newest first
        dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        logger.debug(f"Found {len(dirs)} next-generation models")
        return dirs
    
    def load_next_generation_model(self, model_dir):
        """
        Load a next-generation model from its directory
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            ChessNet model
        """
        from src.agent.model import ChessNet
        
        config_path = os.path.join(model_dir, self.resource.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, self.resource.next_generation_model_weight_filename)
        
        if not (os.path.exists(config_path) and os.path.exists(weight_path)):
            raise FileNotFoundError(f"Model files not found in {model_dir}")
        
        try:
            # Load model config (for validation)
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            
            # Create model
            model = ChessNet()
            
            # Load weights
            model.load_state_dict(torch.load(weight_path, map_location='cpu'))
            
            logger.debug(f"Loaded next-generation model from {os.path.basename(model_dir)}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load next-generation model from {model_dir}: {e}")
            raise
    
    def load_latest_next_generation_model(self):
        """
        Load the most recently saved next-generation model
        Falls back to best model if no next-generation models exist
        
        Returns:
            ChessNet model or None
        """
        dirs = self.get_next_generation_model_dirs()
        
        if not dirs:
            logger.debug("No next-generation models found, loading best model")
            return self.load_best_model()
        
        latest_dir = dirs[0]  # Newest first
        return self.load_next_generation_model(latest_dir)
    
    def archive_model(self, model_dir):
        """
        Move a next-generation model to the archive (copies directory)
        
        Args:
            model_dir: Path to model directory to archive
        """
        copies_dir = os.path.join(self.resource.next_generation_model_dir, "copies")
        os.makedirs(copies_dir, exist_ok=True)
        
        model_name = os.path.basename(model_dir)
        archive_path = os.path.join(copies_dir, model_name)
        
        try:
            shutil.move(model_dir, archive_path)
            logger.info(f"Archived model {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to archive model {model_dir}: {e}")
            raise
    
    def cleanup_old_archives(self, max_archives=50):
        """
        Remove old archived models to save disk space
        
        Args:
            max_archives: Maximum number of archived models to keep
        """
        copies_dir = os.path.join(self.resource.next_generation_model_dir, "copies")
        
        if not os.path.exists(copies_dir):
            return 0
        
        pattern = os.path.join(copies_dir, "model_*")
        archived_dirs = glob.glob(pattern)
        archived_dirs = [d for d in archived_dirs if os.path.isdir(d)]
        
        if len(archived_dirs) <= max_archives:
            return 0
        
        # Sort by modification time, oldest first
        archived_dirs.sort(key=lambda x: os.path.getmtime(x))
        
        # Remove oldest archives
        removed_count = 0
        for old_dir in archived_dirs[:-max_archives]:
            try:
                shutil.rmtree(old_dir)
                removed_count += 1
                logger.debug(f"Removed old archive {os.path.basename(old_dir)}")
            except Exception as e:
                logger.warning(f"Failed to remove archive {old_dir}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old archived models")
        
        return removed_count


# Standalone functions for compatibility with existing code
def load_best_model_weight(model, config=None):
    """
    Standalone function for loading best model weights
    Returns True if successful, False otherwise
    """
    if config is None:
        from src.config.normal import get_config
        config = get_config()
    
    model_manager = ModelManager(config)
    best_model = model_manager.load_best_model()
    
    if best_model is None:
        return False
    
    # Copy weights to the provided model
    model.load_state_dict(best_model.state_dict())
    return True


def save_as_best_model(model, config=None):
    """
    Standalone function for saving a model as best
    """
    if config is None:
        from src.config.normal import get_config
        config = get_config()
    
    model_manager = ModelManager(config)
    model_manager.save_as_best_model(model)


def get_next_generation_model_dirs(resource_config):
    """
    Standalone function for compatibility with existing code
    """
    config = type('Config', (), {'resource': resource_config})()
    model_manager = ModelManager(config)
    return model_manager.get_next_generation_model_dirs()