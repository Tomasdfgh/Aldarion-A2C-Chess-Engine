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

        #Eleventh through Thirtieth Residual Layers (20 more blocks)
        self.conv22 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv23 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv24 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv25 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv26 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv27 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv28 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv29 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv30 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv31 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv32 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv33 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv34 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv35 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv36 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv37 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv38 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv39 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv40 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv41 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv42 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv43 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv44 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv45 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv46 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv47 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv48 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv49 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv50 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv51 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv52 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv53 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv54 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv55 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv56 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv57 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv58 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv59 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv60 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
        self.conv61 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)

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
        self.bn16 = nn.BatchNorm2d(256)  # for conv16
        self.bn17 = nn.BatchNorm2d(256)  # for conv17
        self.bn18 = nn.BatchNorm2d(256)  # for conv18
        self.bn19 = nn.BatchNorm2d(256)  # for conv19
        self.bn20 = nn.BatchNorm2d(256)  # for conv20
        self.bn21 = nn.BatchNorm2d(256)  # for conv21
        
        # BatchNorms for additional residual blocks (22-61)
        self.bn22 = nn.BatchNorm2d(256)  # for conv22
        self.bn23 = nn.BatchNorm2d(256)  # for conv23
        self.bn24 = nn.BatchNorm2d(256)  # for conv24
        self.bn25 = nn.BatchNorm2d(256)  # for conv25
        self.bn26 = nn.BatchNorm2d(256)  # for conv26
        self.bn27 = nn.BatchNorm2d(256)  # for conv27
        self.bn28 = nn.BatchNorm2d(256)  # for conv28
        self.bn29 = nn.BatchNorm2d(256)  # for conv29
        self.bn30 = nn.BatchNorm2d(256)  # for conv30
        self.bn31 = nn.BatchNorm2d(256)  # for conv31
        self.bn32 = nn.BatchNorm2d(256)  # for conv32
        self.bn33 = nn.BatchNorm2d(256)  # for conv33
        self.bn34 = nn.BatchNorm2d(256)  # for conv34
        self.bn35 = nn.BatchNorm2d(256)  # for conv35
        self.bn36 = nn.BatchNorm2d(256)  # for conv36
        self.bn37 = nn.BatchNorm2d(256)  # for conv37
        self.bn38 = nn.BatchNorm2d(256)  # for conv38
        self.bn39 = nn.BatchNorm2d(256)  # for conv39
        self.bn40 = nn.BatchNorm2d(256)  # for conv40
        self.bn41 = nn.BatchNorm2d(256)  # for conv41
        self.bn42 = nn.BatchNorm2d(256)  # for conv42
        self.bn43 = nn.BatchNorm2d(256)  # for conv43
        self.bn44 = nn.BatchNorm2d(256)  # for conv44
        self.bn45 = nn.BatchNorm2d(256)  # for conv45
        self.bn46 = nn.BatchNorm2d(256)  # for conv46
        self.bn47 = nn.BatchNorm2d(256)  # for conv47
        self.bn48 = nn.BatchNorm2d(256)  # for conv48
        self.bn49 = nn.BatchNorm2d(256)  # for conv49
        self.bn50 = nn.BatchNorm2d(256)  # for conv50
        self.bn51 = nn.BatchNorm2d(256)  # for conv51
        self.bn52 = nn.BatchNorm2d(256)  # for conv52
        self.bn53 = nn.BatchNorm2d(256)  # for conv53
        self.bn54 = nn.BatchNorm2d(256)  # for conv54
        self.bn55 = nn.BatchNorm2d(256)  # for conv55
        self.bn56 = nn.BatchNorm2d(256)  # for conv56
        self.bn57 = nn.BatchNorm2d(256)  # for conv57
        self.bn58 = nn.BatchNorm2d(256)  # for conv58
        self.bn59 = nn.BatchNorm2d(256)  # for conv59
        self.bn60 = nn.BatchNorm2d(256)  # for conv60
        self.bn61 = nn.BatchNorm2d(256)  # for conv61
        
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

        #Eighth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn16(self.conv16(x)))
        x = self.bn17(self.conv17(x))
        x += identity
        x = self.relu(x)

        #Ninth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn18(self.conv18(x)))
        x = self.bn19(self.conv19(x))
        x += identity
        x = self.relu(x)

        #Tenth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn20(self.conv20(x)))
        x = self.bn21(self.conv21(x))
        x += identity
        x = self.relu(x)

        #Eleventh Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn22(self.conv22(x)))
        x = self.bn23(self.conv23(x))
        x += identity
        x = self.relu(x)

        #Twelfth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn24(self.conv24(x)))
        x = self.bn25(self.conv25(x))
        x += identity
        x = self.relu(x)

        #Thirteenth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn26(self.conv26(x)))
        x = self.bn27(self.conv27(x))
        x += identity
        x = self.relu(x)

        #Fourteenth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn28(self.conv28(x)))
        x = self.bn29(self.conv29(x))
        x += identity
        x = self.relu(x)

        #Fifteenth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn30(self.conv30(x)))
        x = self.bn31(self.conv31(x))
        x += identity
        x = self.relu(x)

        #Sixteenth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn32(self.conv32(x)))
        x = self.bn33(self.conv33(x))
        x += identity
        x = self.relu(x)

        #Seventeenth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn34(self.conv34(x)))
        x = self.bn35(self.conv35(x))
        x += identity
        x = self.relu(x)

        #Eighteenth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn36(self.conv36(x)))
        x = self.bn37(self.conv37(x))
        x += identity
        x = self.relu(x)

        #Nineteenth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn38(self.conv38(x)))
        x = self.bn39(self.conv39(x))
        x += identity
        x = self.relu(x)

        #Twentieth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn40(self.conv40(x)))
        x = self.bn41(self.conv41(x))
        x += identity
        x = self.relu(x)

        #Twenty-first Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn42(self.conv42(x)))
        x = self.bn43(self.conv43(x))
        x += identity
        x = self.relu(x)

        #Twenty-second Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn44(self.conv44(x)))
        x = self.bn45(self.conv45(x))
        x += identity
        x = self.relu(x)

        #Twenty-third Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn46(self.conv46(x)))
        x = self.bn47(self.conv47(x))
        x += identity
        x = self.relu(x)

        #Twenty-fourth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn48(self.conv48(x)))
        x = self.bn49(self.conv49(x))
        x += identity
        x = self.relu(x)

        #Twenty-fifth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn50(self.conv50(x)))
        x = self.bn51(self.conv51(x))
        x += identity
        x = self.relu(x)

        #Twenty-sixth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn52(self.conv52(x)))
        x = self.bn53(self.conv53(x))
        x += identity
        x = self.relu(x)

        #Twenty-seventh Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn54(self.conv54(x)))
        x = self.bn55(self.conv55(x))
        x += identity
        x = self.relu(x)

        #Twenty-eighth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn56(self.conv56(x)))
        x = self.bn57(self.conv57(x))
        x += identity
        x = self.relu(x)

        #Twenty-ninth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn58(self.conv58(x)))
        x = self.bn59(self.conv59(x))
        x += identity
        x = self.relu(x)

        #Thirtieth Residual Layer
        identity = self.skip_connection(x)
        x = self.relu(self.bn60(self.conv60(x)))
        x = self.bn61(self.conv61(x))
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