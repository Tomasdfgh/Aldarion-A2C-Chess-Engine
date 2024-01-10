import torch.nn as nn
from torch.utils.data import Dataset

class ConvertData(Dataset):
    def __init__(self,array,transform = None):
        self.array = array
        self.transform = transform

    def __getitem__(self,index):
        image, label = self.array[index]
        if image.mode == 'RGBA':
            image = image.convert('RGB')  # Convert RGBA to RGB
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.array)

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()

        # Convolutional layers for board state representation
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)  # Added one channel for color
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Fully connected layers for policy for black
        self.fc1_policy_black = nn.Linear(128 * 8 * 8, 512)
        self.fc2_policy_black = nn.Linear(512, 64)
        self.fc3_policy_black = nn.Linear(64, 64)
        self.fc4_policy_black = nn.Linear(64, 4096)

        # Fully connected layers for policy for white
        self.fc1_policy_white = nn.Linear(128 * 8 * 8, 512)
        self.fc2_policy_white = nn.Linear(512, 64)
        self.fc3_policy_white = nn.Linear(64, 64)
        self.fc4_policy_white = nn.Linear(64, 4096)

        # Fully connected layers for white promotions
        self.fc1_promotion_white = nn.Linear(128 * 8 * 8, 256)
        self.fc2_promotion_white = nn.Linear(256, 4)

        # Fully connected layers for black promotions
        self.fc1_promotion_black = nn.Linear(128 * 8 * 8, 256)
        self.fc2_promotion_black = nn.Linear(256, 4)

        # Fully connected layers for value for black
        self.fc1_value_black = nn.Linear(128 * 8 * 8, 256)
        self.fc2_value_black = nn.Linear(256, 1)

        # Fully connected layers for value for white
        self.fc1_value_white = nn.Linear(128 * 8 * 8, 256)
        self.fc2_value_white = nn.Linear(256, 1)

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
        policy_x_black = self.relu(self.fc1_policy_black(x))
        policy_x_black = self.relu(self.fc2_policy_black(policy_x_black))
        policy_x_black = self.relu(self.fc3_policy_black(policy_x_black))
        policy_output_black = self.fc4_policy_black(policy_x_black)
        policy_output_black = self.softmax(policy_output_black)

        # Policy network for white
        policy_x_white = self.relu(self.fc1_policy_white(x))
        policy_x_white = self.relu(self.fc2_policy_white(policy_x_white))
        policy_x_white = self.relu(self.fc3_policy_white(policy_x_white))
        policy_output_white = self.fc4_policy_white(policy_x_white)
        policy_output_white = self.softmax(policy_output_white)

        # Promotions Network for white
        promotions_x = self.relu(self.fc1_promotion_white(x))
        pro_output_white = self.softmax(self.fc2_promotion_white(promotions_x))

        promotions_x_black = self.relu(self.fc1_value_black(x))
        pro_output_black = self.softmax(self.fc2_promotion_black(promotions_x_black))

        # Value network for black
        value_x_black = self.relu(self.fc1_value_black(x))
        value_output_black = self.tanh(self.fc2_value_black(value_x_black))

        # Value network for white
        value_x_white = self.relu(self.fc1_value_white(x))
        value_output_white = self.tanh(self.fc2_value_white(value_x_white))

        return policy_output_black, policy_output_white, pro_output_white, pro_output_black, value_output_black, value_output_white
