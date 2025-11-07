"""
Base configuration system for Aldarion Chess Engine
"""
import os


class Config:
    
    def __init__(self):
        self.trainer = TrainerConfig()
        self.selfplay = SelfPlayConfig() 
        self.eval = EvaluateConfig()
        self.resource = ResourceConfig()


class TrainerConfig:
    
    def __init__(self):
        self.min_data_size_to_learn = 40000
        self.batch_size = 512
        self.epoch_to_checkpoint = 1
        self.dataset_size = 1000000
        self.start_total_steps = 0
        self.load_data_steps = 5
        self.loss_weights = [1.0, 1.0]  # [policy, value]
        self.lr = 0.005
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.max_candidate_pool_size = 5


class SelfPlayConfig:
    
    def __init__(self):
        self.max_processes = 12
        self.simulation_num_per_move = 450
        self.c_puct = 1.5
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.3
        self.tau_decay_rate = 0.99
        self.max_game_length = 1000
        self.games_per_batch = 12


class EvaluateConfig:
    
    def __init__(self):
        self.game_num = 64
        self.replace_rate = 0.55
        self.evaluate_latest_first = False
        self.max_game_length = 1000
        self.max_processes = 16
        self.simulation_num_per_move = 250
        self.c_puct = 1.0
        self.tau_decay_rate = 0.6
        self.noise_eps = 0

class ResourceConfig:
    """Configuration for file paths and resources"""
    
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
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