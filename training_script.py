import model as md
import board_reader as br
import board_display as bd
import MTCS as mt
import chess
import matplotlib.pyplot as plt
from io import BytesIO
import torch
import ast
import random
from torch.utils.data import Dataset
import torch.nn.functional as F

class ConvertData(Dataset):
    def __init__(self, data, all_moves):
        self.data_array = data
        self.all_moves = all_moves

    def __getitem__(self, index):
        state_cell, mtcs_prob_cell, value_cell = self.data_array[index]
        state = state_cell.value  # Get the value of the cell
        state = br.board_to_array(state, chess.Board(state).turn)
        
        value = value_cell.value
        value = torch.tensor(float(value))
        
        mtcs_prob_str = mtcs_prob_cell.value
        mtcs_prob_dict = eval(mtcs_prob_str)
        for i in self.all_moves:
            if i not in mtcs_prob_dict:
                mtcs_prob_dict[i] = 0
        
        return state, mtcs_prob_dict, value, state_cell.value

    def __len__(self):
        return len(self.data_array)

def get_model_weights(model):
    weights = []
    for param in model.parameters():
        weights.append(param.data.clone())
    return weights


def training(model, trainingSplit, validationSplit, batch_size, sheet, optimizer, epoch):
    print("Number of rows in excel: " + str(sheet.max_row))
    pro_mapper = br.pro_mapper()
    dataset = []
    for _ in range(1,101):
        ran_num = random.randint(1, sheet.max_row)
        dataset.append((sheet[ran_num][0],sheet[ran_num][1],sheet[ran_num][3]))
    all_moves = br.get_all_moves()
    
    chessSet = ConvertData(dataset, all_moves)
    train_set, validation_set, test_set = torch.utils.data.random_split(chessSet, [int((trainingSplit/100) * len(chessSet)),int((validationSplit/100) * len(chessSet)),len(chessSet) - int((trainingSplit/100) * len(chessSet)) - int((validationSplit/100) * len(chessSet))])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = False)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True)

    #'''
    for epoch in range(1, epoch + 1):
        print("Epoch: " + str(epoch))
        training_loss = 0
        validation_loss = 0
        testing_loss = 0

        for i, (state, mtcs_dict, value_label, fen) in enumerate(train_loader):
            
            #----Prepping the Data----#
            policy, value_pred = model(state)
            value_pred = value_pred.reshape(-1)
            pro_policy = policy[:, 4096:]
            policy = policy[:, :4096].reshape(policy.shape[0], 64, 64)

            policy_hash = {}
            for z in mtcs_dict.copy():

                coords = br.legal_move_to_coord(str(z))

                #Getting policy for non pawn promotional moves
                if len(coords) == 2:
                    policy_hash[z] = policy[:, (coords[0][0] + ((coords[0][1] - 1) * 8)) - 1,(coords[1][0] + ((coords[1][1] - 1) * 8)) -1]

                #Getting policy for pawn promotional moves
                if len(coords) == 3:
                    policy_hash[z] = pro_policy[:, pro_mapper[(coords[0], coords[1], coords[2])]]
    

            #----Calculating Loss----#
            #Overall loss
            loss = 0

            # Reward Loss (MSE)
            reward_loss = F.mse_loss(value_label.double(), value_pred.double())

            # Policy Loss (NLL)
            policy_loss = 0
            for i in policy_hash:
                policy_loss += torch.dot(mtcs_dict[i].to(torch.float64), torch.log(policy_hash[i] + 1e-6).to(torch.float64))

            #Model Weight Loss
            weight_loss = 0
            weight_loss_list = get_model_weights(model)
            for i in weight_loss_list:
                weight_loss += i.sum()
            

            #Summing up all the loss
            loss = reward_loss - policy_loss + (10**(-4) * weight_loss)
            training_loss += loss.detach()

            #Performing back prob and gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #Repeating the exact same training process but for validation
        for i, (state, mtcs_dict, value_label, fen) in enumerate(validation_loader):

            #----Prepping the Data----#
            policy, value_pred = model(state)
            value_pred = value_pred.reshape(-1)
            pro_policy = policy[:, 4096:]
            policy =  policy[:, : 4096].reshape(policy.shape[0], 64,64)

            policy_hash = {}
            for z in mtcs_dict.copy():

                coords = br.legal_move_to_coord(str(z))

                if len(coords) == 2:
                    policy_hash[z] = policy[:, (coords[0][0] + ((coords[0][1] - 1) * 8)) - 1,(coords[1][0] + ((coords[1][1] - 1) * 8)) -1]

                if len(coords) == 3:
                    policy_hash[z] = pro_policy[:, pro_mapper[(coords[0], coords[1], coords[2])]]

            #----Calculating Loss----#
            valid_loss = 0

            valid_reward_loss = F.mse_loss(value_label, value_pred)

            valid_policy_loss = 0
            for i in policy_hash:
                valid_policy_loss += torch.dot(mtcs_dict[i].to(torch.float64), torch.log(policy_hash[i] + 1e-6).to(torch.float64))

            valid_weight_loss = 0
            valid_weight_loss_list = get_model_weights(model)
            for i in valid_weight_loss_list:
                valid_weight_loss += i.sum()

            valid_loss = valid_reward_loss - valid_policy_loss + (10**(-4) * valid_weight_loss)
            validation_loss += valid_loss.detach()

        print("Training Loss: " + str(training_loss.item()))
        print("Validation Loss: " + str(validation_loss.item()))
        print('\n')

    #Testing Set
    with torch.no_grad():

        for state, mtcs_dict, value_label, fen in test_loader:

            #----Prepping the Data----#
            policy, value_pred = model(state)
            value_pred = value_pred.reshape(-1)
            pro_policy = policy[:, 4096:]
            policy = policy[:, : 4096].reshape(policy.shape[0], 64, 64)

            policy_hash = {}
            for z in mtcs_dict.copy():

                coords = br.legal_move_to_coord(str(z))

                if len(coords) == 2:
                    policy_hash[z] = policy[:, (coords[0][0] + ((coords[0][1] - 1) * 8)) - 1,(coords[1][0] + ((coords[1][1] - 1) * 8)) -1]

                if len(coords) == 3:
                    policy_hash[z] = pro_policy[:, pro_mapper[(coords[0], coords[1], coords[2])]]

            #----Calculating Loss----#
            test_loss = 0

            test_reward_loss = F.mse_loss(value_pred, value_label)

            test_policy_loss = 0
            for i in policy_hash:
                test_policy_loss += torch.dot(mtcs_dict[i].to(torch.float64), torch.log(policy_hash[i] + 1e-6).to(torch.float64))

            test_weight_loss = 0
            test_weight_loss_list = get_model_weights(model)
            for i in test_weight_loss_list:
                test_weight_loss += i.sum()

            test_loss = test_reward_loss - test_policy_loss + (10 ** (-4) * test_weight_loss)
            testing_loss += test_loss.detach()

    print("Testing Loss: " + str(testing_loss.item()))
    torch.save()
    #'''