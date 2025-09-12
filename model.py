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

        #Eight Residual Layer
        self.conv16 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv17 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)

        #Ninth Residual Layer
        self.conv18 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv19 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)

        #Tenth Residual Layer
        self.conv20 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv21 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)

        #Value Head
        self.conv_Value = nn.Conv2d(256, 1, kernel_size = 1)
        self.linear1 = nn.Linear(1 * 8 * 8, 256)
        self.linear2 = nn.Linear(256, 1)

        #Policy Head
        self.conv_Policy = nn.Conv2d(256, 2, kernel_size = 1)
        self.linear3 = nn.Linear(2 * 8 * 8, 4672)

        # Activation functions, Batch Normalization, and identity
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.batch_norm = nn.BatchNorm2d(256)
        self.batch_norm_one = nn.BatchNorm2d(1)
        self.batch_norm_two = nn.BatchNorm2d(2)
        self.skip_connection = nn.Identity()

    def forward(self, x):

        #First Convolutional Layer
        x = self.relu(self.batch_norm(self.conv1(x)))

        #First Residual Layer
        identity = self.skip_connection(x)
        x = self.batch_norm(self.conv3(self.relu(self.batch_norm(self.conv2(x)))))
        x += identity
        x = self.relu(x)

        #Second Residual Layer
        identity = self.skip_connection(x)
        x = self.batch_norm(self.conv5(self.relu(self.batch_norm(self.conv4(x)))))
        x += identity
        x = self.relu(x)

        #Third Residual Layer
        identity = self.skip_connection(x)
        x = self.batch_norm(self.conv7(self.relu(self.batch_norm(self.conv6(x)))))
        x += identity
        x = self.relu(x)

        #Fourth Residual Layer
        identity = self.skip_connection(x)
        x = self.batch_norm(self.conv9(self.relu(self.batch_norm(self.conv8(x)))))
        x += identity
        x = self.relu(x)

        #Fifth Residual Layer
        identity = self.skip_connection(x)
        x = self.batch_norm(self.conv11(self.relu(self.batch_norm(self.conv10(x)))))
        x += identity
        x = self.relu(x)

        #Sixth Residual Layer
        identity = self.skip_connection(x)
        x = self.batch_norm(self.conv13(self.relu(self.batch_norm(self.conv12(x)))))
        x += identity
        x = self.relu(x)

        #Seventh Residual Layer
        identity = self.skip_connection(x)
        x = self.batch_norm(self.conv15(self.relu(self.batch_norm(self.conv14(x)))))
        x += identity
        x = self.relu(x)

        #Eighth Residual Layer
        identity = self.skip_connection(x)
        x = self.batch_norm(self.conv17(self.relu(self.batch_norm(self.conv16(x)))))
        x += identity
        x = self.relu(x)

        #Ninth Residual Layer
        identity = self.skip_connection(x)
        x = self.batch_norm(self.conv19(self.relu(self.batch_norm(self.conv18(x)))))
        x += identity
        x = self.relu(x)

        #Tenth Residual Layer
        identity = self.skip_connection(x)
        x = self.batch_norm(self.conv21(self.relu(self.batch_norm(self.conv20(x)))))
        x += identity
        x = self.relu(x)

        #Value Head
        value = self.relu(self.batch_norm_one(self.conv_Value(x)))
        value = value.view(-1, 1*8*8)
        value = self.tanh(self.linear2(self.relu(self.linear1(value))))

        #Policy Head
        policy = self.relu(self.batch_norm_two(self.conv_Policy(x)))
        policy = policy.view(-1, 2*8*8)
        policy = self.softmax(self.linear3(policy))

        return policy, value