import torch.nn as nn

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()

        #First Convolutional Layer
        self.conv1 = nn.Conv2d(119, 256, kernel_size = 3, padding = 1)

        #First Residual Layer
        self.conv2 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)

        #Second Residual Layer
        self.conv4 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)

        #Third Residual Layer
        self.conv6 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv7 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)

        #Fourth Residual Layer
        self.conv8 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv9 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)

        #Fifth Residual Layer
        self.conv10 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv11 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)

        #Sixth Residual Layer
        self.conv12 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv13 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)

        #Seventh Residual Layer
        self.conv14 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv15 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)


        #Value Head
        self.conv_Value = nn.Conv2d(256, 1, kernel_size = 1)
        self.linear1 = nn.Linear(1 * 8 * 8, 256)
        self.linear2 = nn.Linear(256, 1)

        #Policy Head
        self.conv_Policy = nn.Conv2d(256, 2, kernel_size = 1)
        self.linear3 = nn.Linear(2 * 8 * 8, 4672)

        # Batch Normalization layers - one for each conv layer
        self.bn1 = nn.BatchNorm2d(256)   # for conv1
        self.bn2 = nn.BatchNorm2d(256)   # for conv2
        self.bn3 = nn.BatchNorm2d(256)   # for conv3
        self.bn4 = nn.BatchNorm2d(256)   # for conv4
        self.bn5 = nn.BatchNorm2d(256)   # for conv5
        self.bn6 = nn.BatchNorm2d(256)   # for conv6
        self.bn7 = nn.BatchNorm2d(256)   # for conv7
        self.bn8 = nn.BatchNorm2d(256)   # for conv8
        self.bn9 = nn.BatchNorm2d(256)   # for conv9
        self.bn10 = nn.BatchNorm2d(256)  # for conv10
        self.bn11 = nn.BatchNorm2d(256)  # for conv11
        self.bn12 = nn.BatchNorm2d(256)  # for conv12
        self.bn13 = nn.BatchNorm2d(256)  # for conv13
        self.bn14 = nn.BatchNorm2d(256)  # for conv14
        self.bn15 = nn.BatchNorm2d(256)  # for conv15
        
        # Head batch norms
        self.bn_value = nn.BatchNorm2d(1)    # for value head
        self.bn_policy = nn.BatchNorm2d(2)   # for policy head
        
        # Activation functions and identity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.skip_connection = nn.Identity()

    def forward(self, x):

        #First Convolutional Layer
        x = self.relu(self.bn1(self.conv1(x)))

        #First Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += identity
        x = self.relu(x)

        #Second Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x += identity
        x = self.relu(x)

        #Third Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.bn7(self.conv7(x))
        x += identity
        x = self.relu(x)

        #Fourth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.bn9(self.conv9(x))
        x += identity
        x = self.relu(x)

        #Fifth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn10(self.conv10(x)))
        x = self.bn11(self.conv11(x))
        x += identity
        x = self.relu(x)

        #Sixth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn12(self.conv12(x)))
        x = self.bn13(self.conv13(x))
        x += identity
        x = self.relu(x)

        #Seventh Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn14(self.conv14(x)))
        x = self.bn15(self.conv15(x))
        x += identity
        x = self.relu(x)

        #Value Head
        value = self.relu(self.bn_value(self.conv_Value(x)))
        value = value.view(-1, 1*8*8)
        value = self.tanh(self.linear2(self.relu(self.linear1(value))))

        #Policy Head
        policy = self.relu(self.bn_policy(self.conv_Policy(x)))
        policy = policy.view(-1, 2*8*8)
        policy = self.linear3(policy)  # Return raw logits, no softmax

        return policy, value