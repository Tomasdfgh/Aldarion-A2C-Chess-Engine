"""
Base configuration system for Aldarion Chess Engine
Modeled after chess-alpha-zero's configuration architecture
"""
import os


class Config:
    """Base configuration class containing all subsystem configs"""
    
    def __init__(self):
        self.trainer = TrainerConfig()
        self.selfplay = SelfPlayConfig() 
        self.eval = EvaluateConfig()
        self.model = ModelConfig()
        self.resource = ResourceConfig()


class TrainerConfig:
    """Configuration for the training worker (opt)"""
    
    def __init__(self):
        self.min_data_size_to_learn = 0
        self.cleaning_processes = 5  # For parallel data loading
        self.vram_frac = 1.0
        self.batch_size = 384  # Tune to your GPU memory
        self.epoch_to_checkpoint = 1  # Save model every epoch
        self.dataset_size = 100000  # Rolling window size
        self.start_total_steps = 0
        self.save_model_steps = 25  # Not used in continuous mode, but kept for compatibility
        self.load_data_steps = 100  # How often to refresh data window
        self.loss_weights = [1.0, 0.5]  # [policy, value] - value reduced to prevent overfitting


class SelfPlayConfig:
    """Configuration for the self-play worker (self)"""
    
    def __init__(self):
        self.max_processes = 3  # Parallel self-play processes
        self.search_threads = 16  # MCTS search threads
        self.vram_frac = 1.0
        self.simulation_num_per_move = 800  # MCTS simulations per move
        self.thinking_loop = 1
        self.logging_thinking = False
        self.c_puct = 1.5  # MCTS exploration parameter
        self.noise_eps = 0.25  # Dirichlet noise
        self.dirichlet_alpha = 0.3  # Dirichlet noise alpha
        self.tau_decay_rate = 0.99  # Temperature decay
        self.virtual_loss = 3  # For parallel MCTS
        self.resign_threshold = -0.8  # When to resign
        self.min_resign_turn = 5  # Minimum moves before resign
        self.max_game_length = 1000  # Maximum game length
        self.games_per_batch = 50  # Games to generate before saving


class EvaluateConfig:
    """Configuration for the evaluation worker (eval)"""
    
    def __init__(self):
        self.vram_frac = 1.0
        self.game_num = 50  # Games per evaluation
        self.replace_rate = 0.55  # Win rate needed to promote model (55%)
        self.play_config = EvalPlayConfig()
        self.evaluate_latest_first = True  # Evaluate newest models first
        self.max_game_length = 1000


class EvalPlayConfig:
    """Play configuration for evaluation games"""
    
    def __init__(self):
        self.max_processes = 3
        self.search_threads = 16
        self.simulation_num_per_move = 200  # Fewer sims for faster evaluation
        self.thinking_loop = 1
        self.c_puct = 1.0  # Lower = prefer mean action value
        self.tau_decay_rate = 0.6
        self.noise_eps = 0  # No noise in evaluation
        

class PlayDataConfig:
    """Configuration for training data management"""
    
    def __init__(self):
        self.min_elo_policy = 500  # Not used but kept for compatibility
        self.max_elo_policy = 1800
        self.nb_game_in_file = 50  # Games per data file
        self.max_file_num = 150  # Maximum data files to keep
        self.sl_nb_game_in_file = 250  # For supervised learning mode


class ModelConfig:
    """Configuration for the neural network model"""
    
    def __init__(self):
        self.cnn_filter_num = 256
        self.cnn_first_filter_size = 3
        self.cnn_filter_size = 3
        self.res_layer_num = 7  # Number of residual blocks
        self.l2_reg = 1e-4
        self.value_fc_size = 256
        self.distributed = False
        self.input_depth = 119  # Aldarion's sophisticated input encoding


class ResourceConfig:
    """Configuration for file paths and resources"""
    
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Model paths
        self.model_dir = os.path.join(self.project_dir, "data", "models")
        self.model_best_config_path = os.path.join(self.model_dir, "best", "model_config.json")
        self.model_best_weight_path = os.path.join(self.model_dir, "best", "model_weight.pth")
        
        # Next generation model paths
        self.next_generation_model_dir = os.path.join(self.model_dir, "candidates")
        self.next_generation_model_dirname_tmpl = "model_%s"
        self.next_generation_model_config_filename = "model_config.json"
        self.next_generation_model_weight_filename = "model_weight.pth"
        
        # Training data paths
        self.play_data_dir = os.path.join(self.project_dir, "data", "training_data")
        self.play_data_filename_tmpl = "play_%s.pkl"
        
        # Log paths
        self.log_dir = os.path.join(self.project_dir, "logs")
        
        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "best"), exist_ok=True)
        os.makedirs(self.next_generation_model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.next_generation_model_dir, "copies"), exist_ok=True)
        os.makedirs(self.play_data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)