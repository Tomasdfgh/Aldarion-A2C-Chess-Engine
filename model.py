import torch.nn as nn
from torch.utils.data import Dataset

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()

        # Convolutional layers for board state representation
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)  # Added one channel for color
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Fully connected layers for policy (Actor Network)
        self.fc1_policy = nn.Linear(128 * 8 * 8, 512)
        self.fc2_policy = nn.Linear(512, 64)
        self.fc3_policy = nn.Linear(64, 64)
        self.fc4_policy = nn.Linear(64, 4096)

        # Fully connected layers for promotions
        self.fc1_promotion = nn.Linear(128 * 8 * 8, 256)
        self.fc2_promotion = nn.Linear(256, 4)

        # Fully connected layers for value (Critic Network)
        self.fc1_value = nn.Linear(128 * 8 * 8, 256)
        self.fc2_value = nn.Linear(256, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Board state representation
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(-1, 128 * 8 * 8)  # Flatten

        # Policy network for black
        policy = self.relu(self.fc1_policy(x))
        policy = self.relu(self.fc2_policy(policy))
        policy = self.relu(self.fc3_policy(policy))
        policy = self.fc4_policy(policy)
        policy = self.softmax(policy)

        # Promotions Network for white
        promotions = self.relu(self.fc1_promotion(x))
        promotions = self.softmax(self.fc2_promotion(promotions))

        # Value network for black
        value = self.relu(self.fc1_value(x))
        value = self.tanh(self.fc2_value(value))

        return policy, promotions, value
