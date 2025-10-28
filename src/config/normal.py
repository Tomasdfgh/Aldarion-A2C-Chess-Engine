"""
Normal (Production) configuration for Aldarion Chess Engine
Modeled after chess-alpha-zero's normal.py configuration
"""
from config import Config


class NormalConfig(Config):
    """Production configuration with full-scale training parameters"""
    
    def __init__(self):
        super().__init__()
        
        # Training Worker Configuration (Optimized for 2x A4000 16GB)
        self.trainer.batch_size = 768  # Larger batch for 16GB GPUs
        self.trainer.dataset_size = 100000  # 100K datapoint rolling window
        self.trainer.epoch_to_checkpoint = 1  # Save every epoch
        self.trainer.cleaning_processes = 10  # More processes for 32-core CPU
        self.trainer.vram_frac = 1.0
        self.trainer.loss_weights = [1.0, 0.5]  # Reduced value loss weight
        
        # Self-Play Worker Configuration (Optimized for 32-core CPU)
        self.selfplay.max_processes = 8  # More processes for 32 cores
        self.selfplay.search_threads = 16
        self.selfplay.simulation_num_per_move = 800  # Full MCTS simulations
        self.selfplay.c_puct = 1.5
        self.selfplay.noise_eps = 0.25
        self.selfplay.dirichlet_alpha = 0.3
        self.selfplay.tau_decay_rate = 0.99
        self.selfplay.virtual_loss = 3
        self.selfplay.resign_threshold = -0.8
        self.selfplay.min_resign_turn = 5
        self.selfplay.max_game_length = 1000
        self.selfplay.games_per_batch = 50  # Games per save cycle
        
        # Evaluation Worker Configuration
        self.eval.game_num = 50  # Full 50-game evaluation
        self.eval.replace_rate = 0.55  # 55% win rate to promote
        self.eval.play_config.simulation_num_per_move = 200  # Evaluation simulations
        self.eval.play_config.max_processes = 6  # More evaluation processes
        self.eval.play_config.c_puct = 1.0  # Lower exploration in evaluation
        self.eval.play_config.tau_decay_rate = 0.6
        self.eval.play_config.noise_eps = 0  # No noise in evaluation
        self.eval.evaluate_latest_first = True


def get_config():
    """Factory function to get normal configuration"""
    return NormalConfig()