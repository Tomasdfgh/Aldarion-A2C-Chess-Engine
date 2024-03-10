import chess
import chess.svg
import numpy as np
import torch
import torch.nn.functional as F
import random
import math

##################################################################################################################################
#														Board Reader			 												 #
#																																 #
#	This script contains all accessory functions. Important functions are board_to_array, which takes in a board in fen string	 #
#	and whose turn it is, then convert that board into a tensor array with shape 6,8,8 which represents the state of the board.	 #
#	best_moves and get_reward are now obsolete functions that is no longer in use. best_moves takes in a policy tensor and 		 #
#	finds the best move to play from that tensor. The MTCS and A2C approach no longer requires for that function to work;		 #
#	however, I will keep it around just incase it ever comes into use again. get reward takes in the board before and after a 	 #
#	move to determine what the reward of that board is. pro_mapper is a function that returns the hashmap to find the index of 	 #
#	a promotional move from a policy. Take a look at MTCS.py to see how pro_mapper is being used.							     #
#																																 #
##################################################################################################################################


#Function to convert the board to a matrix
def board_to_matrix(board):
    matrix = np.zeros((8, 8), dtype=np.dtype('U2'))

    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            if piece is not None:
                # Adjust indexing for NumPy (rank and file are 0-based in NumPy)
                matrix[7 - rank, file] = piece.symbol()
            else:
                matrix[7 - rank, file] = '.'

    return matrix


#Function to find the coordinates of any piece
def find_piece_coordinates(board, piece):
    coordinates = []

    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == piece:
                coordinates.append((i, j))

    return coordinates

#Function to assign the coordinate into the array
def set_values(matrix, coordinates, value):
    for coord in coordinates:
        x, y = coord
        matrix[x][y] = value
    return matrix

#This function converts the board into a 6,8,8 array
def board_to_array(board, turn):
	#set up all the mappings
	piece_mapping = {'R': 'r', 'N': 'n', 'B': 'b', 'Q': 'q', 'K': 'k', 'P':'p'}
	turn_mapping_you = {True: 1, False: -1}
	turn_mapping_op = {True: -1, False: 1}

	board = chess.Board(board)

	#First convert the board into a matrix
	board = board_to_matrix(board)

	array = []

	for i in piece_mapping:
		matrix = np.zeros((8, 8))
		your_cord = find_piece_coordinates(board, i)
		op_cord = find_piece_coordinates(board, piece_mapping[i])
		matrix = set_values(matrix, your_cord, turn_mapping_you[turn])
		matrix = set_values(matrix, op_cord, turn_mapping_op[turn])
		array.append(matrix)

	if turn:
		team_matrix = np.ones((8,8))
	else:
		team_matrix = np.full((8,8),-1)

	for i in range(3):
		array.append(team_matrix)

	return torch.tensor(np.array(array))


def legal_move_to_coord(leg_mov):
	mapping = {'a': 1, 'b': 2, 'c': 3, 'd' : 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
	start_coord = leg_mov[:2]
	end_coord = leg_mov[2:4]


	return [(mapping[start_coord[:len(start_coord)//2]], int(start_coord[len(start_coord)//2:])),(mapping[end_coord[:len(end_coord)//2]], int(end_coord[len(end_coord)//2:]))] if len(leg_mov) == 4 else [(mapping[start_coord[:len(start_coord)//2]], int(start_coord[len(start_coord)//2:])),(mapping[end_coord[:len(end_coord)//2]], int(end_coord[len(end_coord)//2:])), leg_mov[-1]] 

#Takes in a Fen String
def get_mask(board, pro_output):
	mask = torch.zeros(64,64)
	board = chess.Board(board)
	bool_map = {True: 'White', False: 'Black'}
	pro_ = None
	for i in board.legal_moves:
		coords = legal_move_to_coord(str(i))
		mask[(coords[0][0] + ((coords[0][1] - 1) * 8)) - 1][(coords[1][0] + ((coords[1][1] - 1) * 8)) -1] = 1
	return mask

def multi_indices_to_flat_index(indices, shape):
	flat_index = torch.flatten_multi_index(indices.t(), shape)
	return flat_index

#Takes in the fen board and policy_output tensor and outputs a uci best move
def best_moves(board, policy_output, pro_output):
	mapping_re = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5:'e', 6:'f', 7:'g', 0:'h'}
	policy_output = policy_output.reshape(64,-1)
	mask = get_mask(board, pro_output)
	indices = torch.nonzero(mask != 0, as_tuple=False)
	
	policy_output += 1e-10
	policy_output *= mask

	action_space_prob = []
	for i in indices:
		action_space_prob.append(policy_output[i[0], i[1]].item())

	shape_1 = policy_output.shape[1]
	policy_output = policy_output.view(-1)
	pro_move = None
	max_coordinates = torch.multinomial(policy_output, 1).item()
	move_prob = policy_output[max_coordinates]
	max_coordinates_2d = divmod(max_coordinates, shape_1)
	max_coordinates_2d = (max_coordinates_2d[0] + 1, max_coordinates_2d[1] + 1)
	move = str(mapping_re[max_coordinates_2d[0] % 8]) + str(math.ceil(max_coordinates_2d[0]/8)) + str(mapping_re[max_coordinates_2d[1] % 8]) + str(math.ceil(max_coordinates_2d[1]/8))
	
	legal = []
	board = chess.Board(board)
	for i in board.legal_moves:
		legal.append(str(i))

	if ((move + 'q') in legal) or ((move + 'n') in legal) or ((move + 'b') in legal) or ((move + 'r') in legal):
		pro_mapping = {0: 'q', 1: 'r', 2: 'b', 3: 'n'}
		pro_move = torch.multinomial(pro_output, 1).item()
		move += pro_mapping[pro_move]
	return move, max_coordinates, pro_move, action_space_prob

def generate_random_board():
    # Create a new chess board
    board = chess.Board()

    # Make a series of random legal moves
    for _ in range(random.randint(30,100)):  # You can adjust the number of moves
        legal_moves = [move for move in board.legal_moves]
        if not legal_moves:
            break  # No legal moves left
        random_move = legal_moves.pop()
        board.push(random_move)
    return board

#Reward rules are as follows:
#	- Captured pieces get the following reward:
#		- pawn: 1
#		- knight or bishop: 3
#		- Rook: 5
#		- Queen: 9
#	- Win a game: 100
#	- Lose a game: -100

def get_reward(board_before, board_after):
	#Both boards need to be fen and team is a BOOL: True for White and False for Black
	piece_count_before = {'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0, 'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0}
	piece_count_after = {'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0, 'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0}
	piece_captured = {'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0, 'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0}
	score_mapping = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 100, 'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 100}

	reward = 0
	board_before = chess.Board(board_before)
	board_after = chess.Board(board_after)
	board_before = board_to_matrix(board_before)
	board_after = board_to_matrix(board_after)
	for row in range(len(board_before)):
		for square in range(len(board_before[row])):
			if board_before[row][square] in piece_count_before:
				piece_count_before[board_before[row][square]] += 1
			if board_after[row][square] in piece_count_after:
				piece_count_after[board_after[row][square]] += 1

	for i in piece_count_before:
		piece_captured[i] = piece_count_before[i] - piece_count_after[i]
		if piece_captured[i] < 0:
			piece_captured[i] += 1
			if i == 'Q' or i == 'R' or i == 'B' or i == 'N':
				piece_captured['P'] -= 1
			if i == 'q' or i == 'r' or i == 'b' or i == 'n':
				piece_captured['p'] -= 1
	for i in piece_captured:
		reward += piece_captured[i] * score_mapping[i]
	return reward


def pro_mapper():
	pro_mapper = {	((1, 2), (1, 1), 'q'): 0, ((1, 2), (1, 1), 'r'): 1, ((1, 2), (1, 1), 'b'): 2, ((1, 2), (1, 1), 'n'): 3, 
				# (1, 2), (2, 1)
				((1, 2), (2, 1), 'q'): 4, ((1, 2), (2, 1), 'r'): 5, ((1, 2), (2, 1), 'b'): 6, ((1, 2), (2, 1), 'n'): 7, 
				# (2, 2), (1, 1)
				((2, 2), (1, 1), 'q'): 8, ((2, 2), (1, 1), 'r'): 9, ((2, 2), (1, 1), 'b'): 10, ((2, 2), (1, 1), 'n'): 11, 
				# (2, 2), (2, 1)
				((2, 2), (2, 1), 'q'): 12, ((2, 2), (2, 1), 'r'): 13, ((2, 2), (2, 1), 'b'): 14, ((2, 2), (2, 1), 'n'): 15,
				# (2, 2), (3, 1)
				((2, 2), (3, 1), 'q'): 16, ((2, 2), (3, 1), 'r'): 17, ((2, 2), (3, 1), 'b'): 18, ((2, 2), (3, 1), 'n'): 19,
				# (3, 2), (2, 1)
				((3, 2), (2, 1), 'q'): 20, ((3, 2), (2, 1), 'r'): 21, ((3, 2), (2, 1), 'b'): 22, ((3, 2), (2, 1), 'n'): 23,
				# (3, 2), (3, 1)
				((3, 2), (3, 1), 'q'): 24, ((3, 2), (3, 1), 'r'): 25, ((3, 2), (3, 1), 'b'): 26, ((3, 2), (3, 1), 'n'): 27,
				# (3, 2), (4, 1)
				((3, 2), (4, 1), 'q'): 28, ((3, 2), (4, 1), 'r'): 29, ((3, 2), (4, 1), 'b'): 30, ((3, 2), (4, 1), 'n'): 31,
				# (4, 2), (3, 1)
				((4, 2), (3, 1), 'q'): 32, ((4, 2), (3, 1), 'r'): 33, ((4, 2), (3, 1), 'b'): 34, ((4, 2), (3, 1), 'n'): 35,
				# (4, 2), (4, 1)
				((4, 2), (4, 1), 'q'): 36, ((4, 2), (4, 1), 'r'): 37, ((4, 2), (4, 1), 'b'): 38, ((4, 2), (4, 1), 'n'): 39,
				# (4, 2), (5, 1)
				((4, 2), (5, 1), 'q'): 40, ((4, 2), (5, 1), 'r'): 41, ((4, 2), (5, 1), 'b'): 42, ((4, 2), (5, 1), 'n'): 43,
				# (5, 2), (4, 1)
				((5, 2), (4, 1), 'q'): 44, ((5, 2), (4, 1), 'r'): 45, ((5, 2), (4, 1), 'b'): 46, ((5, 2), (4, 1), 'n'): 47,
				# (5, 2), (5, 1)
				((5, 2), (5, 1), 'q'): 48, ((5, 2), (5, 1), 'r'): 49, ((5, 2), (5, 1), 'b'): 50, ((5, 2), (5, 1), 'n'): 51,
				# (5, 2), (6, 1)
				((5, 2), (6, 1), 'q'): 52, ((5, 2), (6, 1), 'r'): 53, ((5, 2), (6, 1), 'b'): 54, ((5, 2), (6, 1), 'n'): 55,
				# (6, 2), (5, 1)
				((6, 2), (5, 1), 'q'): 56, ((6, 2), (5, 1), 'r'): 57, ((6, 2), (5, 1), 'b'): 58, ((6, 2), (5, 1), 'n'): 59,
				# (6, 2), (6, 1)
				((6, 2), (6, 1), 'q'): 60, ((6, 2), (6, 1), 'r'): 61, ((6, 2), (6, 1), 'b'): 62, ((6, 2), (6, 1), 'n'): 63,
				# (6, 2), (7, 1)
				((6, 2), (7, 1), 'q'): 64, ((6, 2), (7, 1), 'r'): 65, ((6, 2), (7, 1), 'b'): 66, ((6, 2), (7, 1), 'n'): 67,
				# (7, 2), (6, 1)
				((7, 2), (6, 1), 'q'): 68, ((7, 2), (6, 1), 'r'): 69, ((7, 2), (6, 1), 'b'): 70, ((7, 2), (6, 1), 'n'): 71,
				# (7, 2), (7, 1)
				((7, 2), (7, 1), 'q'): 72, ((7, 2), (7, 1), 'r'): 73, ((7, 2), (7, 1), 'b'): 74, ((7, 2), (7, 1), 'n'): 75,
				# (7, 2), (8, 1)
				((7, 2), (8, 1), 'q'): 76, ((7, 2), (8, 1), 'r'): 77, ((7, 2), (8, 1), 'b'): 78, ((7, 2), (8, 1), 'n'): 79,
				# (8, 2), (7, 1)
				((8, 2), (7, 1), 'q'): 80, ((8, 2), (7, 1), 'r'): 81, ((8, 2), (7, 1), 'b'): 82, ((8, 2), (7, 1), 'n'): 83,
				# (8, 2), (8, 1)
				((8, 2), (8, 1), 'q'): 84, ((8, 2), (8, 1), 'r'): 85, ((8, 2), (8, 1), 'b'): 86, ((8, 2), (8, 1), 'n'): 87,
				# (1, 7), (1, 8)
				((1, 7), (1, 8), 'q'): 88, ((1, 7), (1, 8), 'r'): 89, ((1, 7), (1, 8), 'b'): 90, ((1, 7), (1, 8), 'n'): 91,
				# (1, 7), (2, 8)
				((1, 7), (2, 8), 'q'): 92, ((1, 7), (2, 8), 'r'): 93, ((1, 7), (2, 8), 'b'): 94, ((1, 7), (2, 8), 'n'): 95,
				# (2, 7), (1, 8)
				((2, 7), (1, 8), 'q'): 96, ((2, 7), (1, 8), 'r'): 97, ((2, 7), (1, 8), 'b'): 98, ((2, 7), (1, 8), 'n'): 99,
				# (2, 7), (2, 8)
				((2, 7), (2, 8), 'q'): 100, ((2, 7), (2, 8), 'r'): 101, ((2, 7), (2, 8), 'b'): 102, ((2, 7), (2, 8), 'n'): 103,
				# (2, 7), (3, 8)
				((2, 7), (3, 8), 'q'): 104, ((2, 7), (3, 8), 'r'): 105, ((2, 7), (3, 8), 'b'): 106, ((2, 7), (3, 8), 'n'): 107,
				# (3, 7), (2, 8)
				((3, 7), (2, 8), 'q'): 108, ((3, 7), (2, 8), 'r'): 109, ((3, 7), (2, 8), 'b'): 110, ((3, 7), (2, 8), 'n'): 111,
				# (3, 7), (3, 8)
				((3, 7), (3, 8), 'q'): 112, ((3, 7), (3, 8), 'r'): 113, ((3, 7), (3, 8), 'b'): 114, ((3, 7), (3, 8), 'n'): 115,
				# (3, 7), (4, 8)
				((3, 7), (4, 8), 'q'): 116, ((3, 7), (4, 8), 'r'): 117, ((3, 7), (4, 8), 'b'): 118, ((3, 7), (4, 8), 'n'): 119,
				# (4, 7), (3, 8)
				((4, 7), (3, 8), 'q'): 120, ((4, 7), (3, 8), 'r'): 121, ((4, 7), (3, 8), 'b'): 122, ((4, 7), (3, 8), 'n'): 123,
				# (4, 7), (4, 8)
				((4, 7), (4, 8), 'q'): 124, ((4, 7), (4, 8), 'r'): 125, ((4, 7), (4, 8), 'b'): 126, ((4, 7), (4, 8), 'n'): 127,
				# (4, 7), (5, 8)
				((4, 7), (5, 8), 'q'): 128, ((4, 7), (5, 8), 'r'): 129, ((4, 7), (5, 8), 'b'): 130, ((4, 7), (5, 8), 'n'): 131,
				# (5, 7), (4, 8)
				((5, 7), (4, 8), 'q'): 132, ((5, 7), (4, 8), 'r'): 133, ((5, 7), (4, 8), 'b'): 134, ((5, 7), (4, 8), 'n'): 135,
				# (5, 7), (5, 8)
				((5, 7), (5, 8), 'q'): 136, ((5, 7), (5, 8), 'r'): 137, ((5, 7), (5, 8), 'b'): 138, ((5, 7), (5, 8), 'n'): 139,
				# (5, 7), (6, 8)
				((5, 7), (6, 8), 'q'): 140, ((5, 7), (6, 8), 'r'): 141, ((5, 7), (6, 8), 'b'): 142, ((5, 7), (6, 8), 'n'): 143,
				# (6, 7), (5, 8)
				((6, 7), (5, 8), 'q'): 144, ((6, 7), (5, 8), 'r'): 145, ((6, 7), (5, 8), 'b'): 146, ((6, 7), (5, 8), 'n'): 147,
				# (6, 7), (6, 8)
				((6, 7), (6, 8), 'q'): 148, ((6, 7), (6, 8), 'r'): 149, ((6, 7), (6, 8), 'b'): 150, ((6, 7), (6, 8), 'n'): 151,
				# (6, 7), (7, 8)
				((6, 7), (7, 8), 'q'): 152, ((6, 7), (7, 8), 'r'): 153, ((6, 7), (7, 8), 'b'): 154, ((6, 7), (7, 8), 'n'): 155,
				# (7, 7), (6, 8)
				((7, 7), (6, 8), 'q'): 156, ((7, 7), (6, 8), 'r'): 157, ((7, 7), (6, 8), 'b'): 158, ((7, 7), (6, 8), 'n'): 159,
				# (7, 7), (7, 8)
				((7, 7), (7, 8), 'q'): 160, ((7, 7), (7, 8), 'r'): 161, ((7, 7), (7, 8), 'b'): 162, ((7, 7), (7, 8), 'n'): 163,
				# (7, 7), (8, 8)
				((7, 7), (8, 8), 'q'): 164, ((7, 7), (8, 8), 'r'): 165, ((7, 7), (8, 8), 'b'): 166, ((7, 7), (8, 8), 'n'): 167,
				# (8, 7), (7, 8)
				((8, 7), (7, 8), 'q'): 168, ((8, 7), (7, 8), 'r'): 169, ((8, 7), (7, 8), 'b'): 170, ((8, 7), (7, 8), 'n'): 171,
				# (8, 7), (8, 8)
				((8, 7), (8, 8), 'q'): 172, ((8, 7), (8, 8), 'r'): 173, ((8, 7), (8, 8), 'b'): 174, ((8, 7), (8, 8), 'n'): 175,
				}

	return pro_mapper

def generate_chess_moves():
    #Return every single possible moves in Chess, even the impossible ones that will never be played
    moves = []
    files = 'abcdefgh'
    ranks = '12345678'

    for start_file in files:
        for start_rank in ranks:
            for end_file in files:
                for end_rank in ranks:
                    move = start_file + start_rank + end_file + end_rank
                    moves.append(move)

    return moves

def get_all_moves():
    #Initializing the all_moves hashmap
    all_moves = {}

    #Getting every possible moves not including pawn promotional moves
    all_moves_list = generate_chess_moves()
    all_pro_moves_list = []
    alph_to_num = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
    for i in all_moves_list:
        #Cases for black upgrades
        if int(i[1]) == 2 and int(i[3]) == 1:
            if abs(alph_to_num[i[0]] - alph_to_num[i[2]]) <= 1:
                all_pro_moves_list.append(i)

        #Cases for white upgrades
        if int(i[1]) == 7 and int(i[3]) == 8:
            if abs(alph_to_num[i[0]] - alph_to_num[i[2]]) <= 1:
                all_pro_moves_list.append(i)

    #Adding Promotional moves
    for i in range(len(all_pro_moves_list.copy())):
        temp = [all_pro_moves_list[i] + 'b', all_pro_moves_list[i] + 'n', all_pro_moves_list[i] + 'r']
        all_pro_moves_list[i] += 'q'
        all_pro_moves_list += temp

    for i in (all_pro_moves_list + all_moves_list):
        all_moves[i] = 0

    return all_moves

def array_to_tensor(array):
	return torch.tensor(array)

def prob_sampler(array, num_sam):
	return torch.multinomial(array, num_sam, replacement=True)