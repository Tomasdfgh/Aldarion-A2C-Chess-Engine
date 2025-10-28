"""
Mini (Testing) configuration for Aldarion Chess Engine
Optimized for 2x RTX A4000 (16GB each) + 32-core Threadripper
Fast testing with smaller scale parameters
"""
from config import Config


class MiniConfig(Config):
    """Testing configuration with smaller parameters for faster iteration"""
    
    def __init__(self):
        super().__init__()
        
        # Training Worker Configuration (Optimized for A4000s)
        self.trainer.batch_size = 512  # Larger batch size for 16GB GPUs
        self.trainer.dataset_size = 10000  # Smaller window for testing (10K datapoints)
        self.trainer.epoch_to_checkpoint = 1  # Save every epoch
        self.trainer.cleaning_processes = 8  # More processes for 32-core CPU
        self.trainer.vram_frac = 0.9  # Conservative GPU memory usage
        self.trainer.loss_weights = [1.0, 0.5]  # Same loss weights
        
        # Self-Play Worker Configuration (Fast testing)
        self.selfplay.max_processes = 6  # More processes for 32 cores
        self.selfplay.search_threads = 16
        self.selfplay.simulation_num_per_move = 100  # Fewer sims for speed
        self.selfplay.c_puct = 1.5
        self.selfplay.noise_eps = 0.25
        self.selfplay.dirichlet_alpha = 0.3
        self.selfplay.tau_decay_rate = 0.99
        self.selfplay.virtual_loss = 3
        self.selfplay.resign_threshold = -0.8
        self.selfplay.min_resign_turn = 5
        self.selfplay.max_game_length = 200  # Shorter games for testing
        self.selfplay.games_per_batch = 6  # Balanced for continuous training
        
        # Evaluation Worker Configuration (Quick evaluation)
        self.eval.game_num = 10  # Only 10 games for fast testing
        self.eval.replace_rate = 0.55  # Same promotion threshold
        self.eval.play_config.simulation_num_per_move = 50  # Fast evaluation
        self.eval.play_config.max_processes = 4  # More parallel evaluation
        self.eval.play_config.c_puct = 1.0
        self.eval.play_config.tau_decay_rate = 0.6
        self.eval.play_config.noise_eps = 0
        self.eval.evaluate_latest_first = True


def get_config():
    """Factory function to get mini configuration"""
    return MiniConfig()