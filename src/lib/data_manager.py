"""
Data Management System for Aldarion Chess Engine
Handles rolling window of training data with FIFO removal
Modeled after chess-alpha-zero's data management
"""
import os
import pickle
import glob
from datetime import datetime
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DataManager:
    """Manages rolling window of training data files"""
    
    def __init__(self, config):
        self.config = config
        self.resource = config.resource
        self.play_data_config = config.trainer
        
    def get_game_data_filenames(self):
        """
        Get list of all training data files, sorted by modification time
        Returns newest files first
        """
        pattern = os.path.join(self.resource.play_data_dir, "play_*.pkl")
        files = glob.glob(pattern)
        
        # Sort by modification time, newest first
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        logger.debug(f"Found {len(files)} training data files")
        return files
    
    def write_game_data_to_file(self, game_data, filename=None):
        """
        Write game data to a new file
        
        Args:
            game_data: List of (board_fen, history_fens, move_probabilities_dict, game_outcome) tuples
            filename: Optional filename, auto-generated if None
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")[:-3]  # Include milliseconds
            filename = self.resource.play_data_filename_tmpl % timestamp
        
        filepath = os.path.join(self.resource.play_data_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(game_data, f)
            
            logger.info(f"Saved {len(game_data)} games to {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save game data to {filepath}: {e}")
            raise
    
    def read_game_data_from_file(self, filepath):
        """
        Read game data from a file
        
        Returns:
            List of (board_fen, history_fens, move_probabilities_dict, game_outcome) tuples
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            logger.debug(f"Loaded {len(data)} games from {os.path.basename(filepath)}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load game data from {filepath}: {e}")
            raise
    
    def get_total_datapoints(self):
        """
        Count total number of datapoints across all files
        
        Returns:
            Total datapoint count
        """
        files = self.get_game_data_filenames()
        total_datapoints = 0
        
        for filepath in files:
            try:
                game_data = self.read_game_data_from_file(filepath)
                # Each game generates multiple datapoints (one per move)
                for game in game_data:
                    _, history_fens, _, _ = game
                    total_datapoints += len(history_fens)  # Number of moves/datapoints
            except Exception as e:
                logger.warning(f"Skipping corrupted file {filepath}: {e}")
                continue
        
        logger.debug(f"Total datapoints across all files: {total_datapoints}")
        return total_datapoints
    
    def cycle_old_data_if_needed(self):
        """
        Remove oldest data files if total datapoints exceed threshold
        Implements FIFO removal to maintain rolling window
        """
        max_datapoints = self.play_data_config.dataset_size
        cycle_threshold = max_datapoints // 2  # Remove when > 50K datapoints
        
        total_datapoints = self.get_total_datapoints()
        
        if total_datapoints <= cycle_threshold:
            logger.debug(f"Data cycling not needed: {total_datapoints} <= {cycle_threshold}")
            return 0
        
        files = self.get_game_data_filenames()
        files.reverse()  # Oldest first for removal
        
        removed_count = 0
        current_datapoints = total_datapoints
        
        logger.info(f"Starting data cycling: {current_datapoints} datapoints > {cycle_threshold} threshold")
        
        for filepath in files:
            if current_datapoints <= cycle_threshold:
                break
                
            try:
                # Count datapoints in this file before removing
                game_data = self.read_game_data_from_file(filepath)
                file_datapoints = sum(len(game[1]) for game in game_data)  # Sum of move counts
                
                # Remove the file
                os.remove(filepath)
                current_datapoints -= file_datapoints
                removed_count += 1
                
                logger.info(f"Removed {os.path.basename(filepath)} ({file_datapoints} datapoints)")
                
            except Exception as e:
                logger.warning(f"Failed to remove {filepath}: {e}")
                continue
        
        logger.info(f"Data cycling complete: removed {removed_count} files, "
                   f"{current_datapoints} datapoints remaining")
        return removed_count
    
    def load_data_for_training(self, max_datapoints=None):
        """
        Load training data up to max_datapoints limit
        Returns data in the format expected by training pipeline
        
        Args:
            max_datapoints: Maximum datapoints to load (defaults to config.dataset_size)
            
        Returns:
            List of training datapoints ready for DataLoader
        """
        if max_datapoints is None:
            max_datapoints = self.play_data_config.dataset_size
        
        files = self.get_game_data_filenames()  # Newest first
        training_data = []
        loaded_datapoints = 0
        
        logger.info(f"Loading training data (max {max_datapoints} datapoints)...")
        
        for filepath in files:
            if loaded_datapoints >= max_datapoints:
                break
                
            try:
                game_data = self.read_game_data_from_file(filepath)
                
                # Convert game data to individual training datapoints
                for game in game_data:
                    board_fen, history_fens, move_probs, game_outcome = game
                    
                    # Each position in the game becomes a training datapoint
                    for i, fen in enumerate(history_fens):
                        if loaded_datapoints >= max_datapoints:
                            break
                            
                        # Create training sample
                        sample = (fen, history_fens[:i+1], move_probs.get(i, {}), game_outcome)
                        training_data.append(sample)
                        loaded_datapoints += 1
                    
                    if loaded_datapoints >= max_datapoints:
                        break
                        
            except Exception as e:
                logger.warning(f"Skipping corrupted file {filepath}: {e}")
                continue
        
        logger.info(f"Loaded {loaded_datapoints} datapoints from {len(files)} files for training")
        return training_data
    
    def cleanup_empty_files(self):
        """Remove any empty or corrupted data files"""
        files = self.get_game_data_filenames()
        removed_count = 0
        
        for filepath in files:
            try:
                game_data = self.read_game_data_from_file(filepath)
                if not game_data:  # Empty file
                    os.remove(filepath)
                    removed_count += 1
                    logger.info(f"Removed empty file {os.path.basename(filepath)}")
            except:
                # Corrupted file
                os.remove(filepath)
                removed_count += 1
                logger.info(f"Removed corrupted file {os.path.basename(filepath)}")
        
        return removed_count


def get_game_data_filenames(resource_config):
    """
    Standalone function for compatibility with existing code
    """
    data_manager = DataManager(type('Config', (), {'resource': resource_config, 'trainer': type('Trainer', (), {'dataset_size': 100000})()})())
    return data_manager.get_game_data_filenames()


def write_game_data_to_file(game_data, resource_config, filename=None):
    """
    Standalone function for compatibility with existing code
    """
    data_manager = DataManager(type('Config', (), {'resource': resource_config, 'trainer': type('Trainer', (), {'dataset_size': 100000})()})())
    return data_manager.write_game_data_to_file(game_data, filename)


def read_game_data_from_file(filepath):
    """
    Standalone function for compatibility with existing code
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)