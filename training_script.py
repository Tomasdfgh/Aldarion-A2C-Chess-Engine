import chess
import chess.svg
import cairosvg
import matplotlib.pyplot as plt
from io import BytesIO
import model as md
import board_reader as br
import torch
import board_display as bd
import torch.optim as optim
import os

#losses to consider:
#White Policy
#Black Policy

#White value
#Black value

#White promotions
#Black promotions


#'''
model = md.ChessNet().double()
model.train()

gamma = 0.99

#Building the Optimizers
#White Policy Optimizer
params_white_policy = list(model.conv1.parameters()) + list(model.conv2.parameters()) + list(model.conv3.parameters()) + list(model.fc1_policy_white.parameters()) + list(model.fc2_policy_white.parameters()) + list(model.fc3_policy_white.parameters()) + list(model.fc4_policy_white.parameters())
optim_white_policy = optim.AdamW(params_white_policy, lr = 10)

#Black Policy Optimizer
params_black_policy = list(model.conv1.parameters()) + list(model.conv2.parameters()) + list(model.conv3.parameters()) + list(model.fc1_policy_black.parameters()) + list(model.fc2_policy_black.parameters()) + list(model.fc3_policy_black.parameters()) + list(model.fc4_policy_black.parameters())
optim_black_policy = optim.AdamW(params_black_policy, lr = 10)

#White Promotions Optimizer
params_white_pro = list(model.conv1.parameters()) + list(model.conv2.parameters()) + list(model.conv3.parameters()) + list(model.fc1_promotion_white.parameters()) + list(model.fc2_promotion_white.parameters())
optim_white_pro = optim.AdamW(params_white_pro, lr = 10)

#Black Promotions Optimizer 
params_black_pro = list(model.conv1.parameters()) + list(model.conv2.parameters()) + list(model.conv3.parameters()) + list(model.fc1_promotion_black.parameters()) + list(model.fc2_promotion_black.parameters())
optim_black_pro = optim.AdamW(params_black_pro, lr = 10)

#White Value Optimizer
params_white_value = list(model.conv1.parameters()) + list(model.conv2.parameters()) + list(model.conv3.parameters()) + list(model.fc1_value_white.parameters()) + list(model.fc2_value_white.parameters())
optim_white_value = optim.AdamW(params_white_pro, lr = 10)

#Black Value Optimizer
params_black_value = list(model.conv1.parameters()) + list(model.conv2.parameters()) + list(model.conv3.parameters()) + list(model.fc1_value_black.parameters()) + list(model.fc2_value_black.parameters())
optim_black_value = optim.AdamW(params_black_pro, lr = 10)

torch.autograd.set_detect_anomaly(True)

#Training Loop
for _ in range(100):

    # #model_path = 'AldarionChessEngine.pth'
    # if os.path.exists(model_path):
    #     # Load the saved model
    #     model.load_state_dict(torch.load(model_path))
    #     print("Loaded existing model.")

    #Initializing the board
    board = chess.Board()

    fig, ax = bd.get_fig_ax()
    bd.render_board(board, fig, ax)

    promotion_loss_black = 0
    promotion_loss_white = 0

    policy_loss_black = 0
    policy_loss_white = 0

    value_loss_black = 0
    value_loss_white = 0

    reward_black = 0
    reward_white = 0

    move_turn = 0

    move_num = []
    policy_loss = []

    inverter = {True: False, False: True}

    while True:

        move_turn += 1
        move_num.append(move_turn)

        #------------------------------------------WHITE's TURN------------------------------------------#

        #Getting the move distributions
        #print("White's Turn")
        board_before = board.fen()
        array = br.board_to_array(board.fen())
        array = torch.tensor(array)
        move1_black, move1_white, pro_output_white, pro_output_black, value_black, value_white = model(array)

        params_before = {}
        for name, param in model.named_parameters():
            params_before[name] = param.clone()

        move1_white = move1_white.clone()
        
        #Applying the move
        best_move, move_ind, pro_ind = br.best_moves(board.fen(), move1_white[0], pro_output_white[0])
        move = chess.Move.from_uci(best_move)
        bd.render_board(board, fig, ax, move = str(move))
        board_after = board.fen()

        #Finding the value of the next state
        array2 = br.board_to_array(board.fen())
        array2 = torch.tensor(array2)
        _, _, _, _, _, next_value_white = model(array2)

        #Calculating the reward
        reward_white = br.get_reward(board_before, board_after)
        #print("Reward: " + str(reward_white))

        #Updating the Critic
        td_error = reward_white + gamma * next_value_white - value_white
        value_loss_white = td_error.pow(2).mean()
        optim_white_value.zero_grad()
        value_loss_white.backward(retain_graph=True)
        #print("Critic Error: " + str(value_loss_white.item()))

        #Updating the Actor
        advantage = td_error.detach()
        #print("Advantage: " + str(advantage.item()))
        selected_prob = move1_white[0, move_ind]
        policy_loss_white = -torch.log(selected_prob) * advantage
        optim_white_policy.zero_grad()
        policy_loss_white.backward(retain_graph=True)
        policy_loss.append(policy_loss_white.item())
        #print("Policy Error: " + str(policy_loss_white.item()))

        #Updating the Promotions
        if pro_ind is not None:
            promotion_loss_white = -torch.log(pro_output_white[0][pro_ind]) * advantage
            optim_white_pro.zero_grad()
            promotion_loss_white.backward(retain_graph=True)
            #print("Promotion Error: " + str(promotion_loss_white.item()))
        #print("\n\n\n")

        optim_white_value.step()
        params_after = {}
        for name, param in model.named_parameters():
            params_after[name] = param.clone()
        print("check length: " + str(len(params_before) == len(params_after)))
        print("length: " + str(len(params_after)))
        for i in params_after:
            print(str(i) + ": " + str(inverter[torch.equal(params_after[i], params_before[i])]))

        optim_white_policy.step()
        if pro_ind is not None:
            optim_white_pro.step()

        if board.is_game_over():
            print("AFTER WHITE MOVED")
            if board.result() == '1/2-1/2':
                print("Tied")
            else:
                if board.result() == "1-0":
                    print("White Won")
                if board.result() == "0-1":
                    print("Black Won")
            break

        #------------------------------------------BLACK's TURN------------------------------------------#

        #Getting the move distributions
        #print("Black's Turn")
        board_before_2 = board.fen()
        array = br.board_to_array(board.fen())
        array = torch.tensor(array)
        move2_black, move2_white, pro_output_white, pro_output_black, value_black, value_white = model(array)

        if board.turn:
            move2_clone = move2_white.clone()
            pro2_ = pro_output_white.clone()
        else:
            move2_clone = move2_black.clone()
            pro2_ = pro_output_black.clone()

        #Applying the move
        best_move2, move_ind2, pro_ind2 = br.best_moves(board.fen(), move2_clone[0], pro2_[0])
        move2 = chess.Move.from_uci(best_move2)
        bd.render_board(board, fig, ax, move = str(move2))
        board_after_2 = board.fen()

        #Finding the value of the next state
        array_black2 = br.board_to_array(board.fen())
        array_black2 = torch.tensor(array_black2)
        _, _, _, _, next_value_black, _ = model(array_black2)

        #Calculating the reward
        reward_black = br.get_reward(board_before_2, board_after_2)
        #print("Reward: " + str(reward_black))

        #Updating the Critic
        td_error = reward_white + gamma * next_value_black - value_black
        value_loss_black = td_error.pow(2).mean()
        optim_black_value.zero_grad()
        value_loss_black.backward(retain_graph = True)
        #print("Critic Error: " + str(value_loss_black.item()))

        #Updating the Actor
        advantage = td_error.detach().clone()
        #print("Advantage: " + str(advantage.item()))
        policy_loss_black = -torch.log(move2_clone[0][move_ind2]) * advantage
        optim_black_policy.zero_grad()
        policy_loss_black.backward(retain_graph = True)
        #print("Policy Error: " + str(policy_loss_black.item()))

        #Updating the Promotions
        if pro_ind2 is not None:
            promotion_loss_black = -torch.log(pro2_[0][pro_ind2]) * advantage
            #print(promotion_loss_black.shape)
            #print(promotion_loss_black)
            optim_black_pro.zero_grad()
            policy_loss_black.backward(retain_graph = True)
            #print("Promotion Error: " + str(promotion_loss_black.item()))
        #print("\n\n\n")

        optim_black_policy.step()
        optim_black_value.step()
        if pro_ind2 is not None:
            optim_black_pro.step()

        if board.is_game_over():
            print("AFTER BLACK MOVED")
            if board.result() == '1/2-1/2':
                print("Tied")
            else:
                if board.result() == "1-0":
                    print("White Won")
                if board.result() == "0-1":
                    print("Black Won")
            break

    #------------------------------------------GAME OVER------------------------------------------#
    plt.close(fig)

    #torch.save(model.state_dict(), model_path)
#'''


# import torch

# # Create two tensors
# tensor1 = torch.tensor([1, 2, 2])
# tensor2 = torch.tensor([1, 2, 2.00000000000000000001])

# # Check if the tensors are equal
# are_equal = torch.equal(tensor1, tensor2)

# # Print the result
# print(f"Are the tensors equal? {are_equal}")