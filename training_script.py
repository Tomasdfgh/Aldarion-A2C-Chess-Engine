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
import openpyxl

model = md.ChessNet().double()
model.train()

gamma = 0.95

beta = 0.3

optimizer = optim.AdamW(model.parameters(), lr = 0.001)

torch.autograd.set_detect_anomaly(True)

fig2, ax2 = bd.get_fig_ax()

#Load up the workbook
workbook = openpyxl.load_workbook('Training_Result.xlsx')
sheet = workbook.active
game_num = sheet[sheet.max_row][0].value


#Plotting Utlities
game_num_ = []
white_actor_ = []
white_critic_ = []

black_actor_ = []
black_critic_ = []


first = True
for i in sheet.iter_rows(values_only=True):
    if first:
        first = False
        continue
    game_num_.append(i[0])
    white_actor_.append(i[1])
    white_critic_.append(i[2])
    black_actor_.append(i[5])
    black_critic_.append(i[6])


#'''
for _ in range(10000):

    game_num += 1

    # Load the saved model
    model_path = 'AldarionChessEngine.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    #Initializing the board
    board = chess.Board()

    fig, ax = bd.get_fig_ax()
    bd.render_board(board, fig, ax)

    #Variables to include in the Excel Sheet
    white_actor_loss = 0
    white_critic_loss = 0
    white_pro_loss = 0
    white_reward = 0

    black_actor_loss = 0
    black_critic_loss = 0
    black_pro_loss = 0
    black_reward = 0
    game_result = -1 # -1: no result yet, 0: tied, 1: white wins, 2: black wins

    while True:
        
        reward_black = 0
        reward_white = 0

        #------------------------------------------WHITE's TURN------------------------------------------#

        #Getting the move distributions
        board_before = board.fen()
        array = br.board_to_array(board.fen())
        array = torch.tensor(array)
        move1_black, move1_white, pro_output_white, pro_output_black, value_black, value_white = model(array)
        move1_white = move1_white.clone()
        
        #Applying the move
        best_move, move_ind, pro_ind, act_space_prob = br.best_moves(board.fen(), move1_white[0], pro_output_white[0])
        move = chess.Move.from_uci(best_move)
        bd.render_board(board, fig, ax, move = str(move))
        board_after = board.fen()

        #Finding the value of the next state
        array2 = br.board_to_array(board.fen())
        array2 = torch.tensor(array2)
        _, _, _, _, _, next_value_white = model(array2)

        #Zeroing out the gradients
        optimizer.zero_grad()

        #Assigning 100 reward to white if White win
        if board.is_game_over():
            if board.result() == "1-0":
                reward_white += 50

        #Calculating the reward
        reward_white += br.get_reward(board_before, board_after)
        white_reward += reward_white

        #Obtaining Critic Loss
        td_error = value_white - (reward_white + gamma * next_value_white)
        value_loss_white = td_error.pow(2)
        white_critic_loss += td_error.pow(2).item()
        value_loss_white.backward(retain_graph=True)

        #Obtaining Entropy Loss
        entropy_loss_white = 0
        for i in act_space_prob:
            entropy_loss_white += i *  -torch.log(torch.tensor(i))

        #Obtaining Actor Loss
        advantage = reward_white + next_value_white - value_white
        policy_loss_white = (-torch.log(move1_white[0, move_ind]) * advantage) - (beta * entropy_loss_white)
        white_actor_loss += policy_loss_white.item()
        policy_loss_white.backward(retain_graph=True)

        #Obtaining Promotion Loss
        if pro_ind is not None:
            promotion_loss_white = -torch.log(pro_output_white[0][pro_ind]) * advantage
            white_pro_loss += promotion_loss_white.item()
            promotion_loss_white.backward(retain_graph=True)
        optimizer.step()

        if board.is_game_over():
            if board.result() == '1/2-1/2':
                game_result = 0
            else:
                if board.result() == "1-0":
                    game_result = 1
                if board.result() == "0-1":
                    game_result = 2
            break

        #------------------------------------------BLACK's TURN------------------------------------------#

        #Getting the move distributions
        board_before_2 = board.fen()
        array = br.board_to_array(board.fen())
        array = torch.tensor(array)
        move2_black, move2_white, pro_output_white, pro_output_black, value_black, value_white = model(array)
        move2_black = move2_black.clone()

        #Applying the move
        best_move2, move_ind2, pro_ind2, act_space_prob2 = br.best_moves(board.fen(), move2_black[0], pro_output_black[0])
        move2 = chess.Move.from_uci(best_move2)
        bd.render_board(board, fig, ax, move = str(move2))
        board_after_2 = board.fen() 

        #Finding the value of the next state
        array_black2 = br.board_to_array(board.fen())
        array_black2 = torch.tensor(array_black2)
        _, _, _, _, next_value_black, _ = model(array_black2)

        #Zeroing out the Gradients
        optimizer.zero_grad()

        #Assigning 100 reward to black if Black win
        if board.is_game_over():
            if board.result() == "0-1":
                reward_black += 50

        #Calculating the reward
        reward_black += br.get_reward(board_before_2, board_after_2)
        black_reward += reward_black

        #Obtaining Critic Loss
        td_error = value_black - (reward_black + gamma * next_value_black)
        value_loss_black = td_error.pow(2)
        black_critic_loss += value_loss_black.item()
        value_loss_black.backward(retain_graph = True)

        #Obtaining Entropy Loss
        entropy_loss_black = 0
        for i in act_space_prob2:
            entropy_loss_black += i *  -torch.log(torch.tensor(i))

        #Obtaining Actor Loss
        advantage = reward_black + next_value_black - value_black
        policy_loss_black = (-torch.log(move2_black[0, move_ind2]) * advantage) - (beta * entropy_loss_black)
        black_actor_loss += policy_loss_black.item()
        policy_loss_black.backward(retain_graph = True)

        #Obtaining Promotion Loss
        if pro_ind2 is not None:
            promotion_loss_black = -torch.log(pro_output_black[0][pro_ind2]) * advantage
            black_pro_loss += promotion_loss_black.item()
            policy_loss_black.backward(retain_graph = True)
        optimizer.step()

        if board.is_game_over():
            if board.result() == '1/2-1/2':
                game_result = 0
            else:
                if board.result() == "1-0":
                    game_result = 1
                if board.result() == "0-1":
                    game_result = 2
            break

    #------------------------------------------GAME OVER------------------------------------------#
    plt.close(fig)

    new_entry = [game_num, white_actor_loss, white_critic_loss, white_pro_loss, white_reward, black_actor_loss, black_critic_loss, black_pro_loss, black_reward, game_result]
    sheet.append(new_entry)
    workbook.save('Training_Result.xlsx')
    torch.save(model.state_dict(), model_path)

    #Plotting
    game_num_.append(new_entry[0])
    white_actor_.append(new_entry[1])
    white_critic_.append(new_entry[2])
    black_actor_.append(new_entry[5])
    black_critic_.append(new_entry[6])
    bd.plot_loss(fig2, ax2, game_num_, white_actor_, white_critic_, black_actor_, black_critic_)
#'''