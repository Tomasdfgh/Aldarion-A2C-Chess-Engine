"""
Data Management System for Aldarion Chess Engine
Handles rolling window of training data with FIFO removal
"""
import os
import pickle
import glob
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataManager:
    
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
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")[:-3]
            filename = self.resource.play_data_filename_tmpl % timestamp
        
        filepath = os.path.join(self.resource.play_data_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(game_data, f)
            
            logger.info(f"Saved {len(game_data)} data points to {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save game data to {filepath}: {e}")
            raise
    
    def read_game_data_from_file(self, filepath):
        """
        Read game data from a file
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
        """
        files = self.get_game_data_filenames()
        total_datapoints = 0
        
        for filepath in files:
            try:
                game_data = self.read_game_data_from_file(filepath)
                total_datapoints += len(game_data)
            except Exception as e:
                logger.warning(f"Skipping corrupted file {filepath}: {e}")
                continue
        
        return total_datapoints
    
    def cycle_old_data_if_needed(self):
        """
        Remove oldest data files if total datapoints exceed threshold
        Implements FIFO removal to maintain rolling window
        """
        max_datapoints = self.play_data_config.dataset_size
        total_datapoints = self.get_total_datapoints()
        files = self.get_game_data_filenames()

        if len(files) <= 1:
            return 0
            
        # Exclude the newest file from removal, then reversing it.
        files = files[1:][::-1]
        
        removed_count = 0
        current_datapoints = total_datapoints
        
        if current_datapoints > max_datapoints:
            logger.info(f"Starting data cycling: {current_datapoints} datapoints > {max_datapoints} threshold")
        
        for filepath in files:
            
            if current_datapoints <= max_datapoints:
                break
                
            try:
                # Count datapoints in this file before removing
                game_data = self.read_game_data_from_file(filepath)
                file_datapoints = len(game_data)
                
                # Remove the file
                os.remove(filepath)
                current_datapoints -= file_datapoints
                removed_count += 1
                
                logger.info(f"Removed {os.path.basename(filepath)} ({file_datapoints} datapoints)")
                
            except Exception as e:
                logger.warning(f"Failed to remove {filepath}: {e}")
                continue
        
        logger.info(f"Data cycling complete: removed {removed_count} files, {current_datapoints} datapoints remaining\n")
        
        return removed_count