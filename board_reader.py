import chess
import chess.svg
import numpy as np
import torch
import torch.nn.functional as F
import random
import math
#[Up-Down, Side]


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
def board_to_array(board):
	#set up all the mappings
	piece_mapping = {'R': 'r', 'N': 'n', 'B': 'b', 'Q': 'q', 'K': 'k', 'P':'p'}

	board = chess.Board(board)

	#First convert the board into a matrix
	board = board_to_matrix(board)

	array = []

	for i in piece_mapping:
		matrix = np.zeros((8, 8))
		your_cord = find_piece_coordinates(board, i)
		op_cord = find_piece_coordinates(board, piece_mapping[i])
		matrix = set_values(matrix, your_cord, 1)
		matrix = set_values(matrix, op_cord, -1)
		array.append(matrix)
	return np.array(array)


def legal_move_to_coord(leg_mov):
	mapping = {'a': 1, 'b': 2, 'c': 3, 'd' : 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
	start_coord = leg_mov[:2]
	end_coord = leg_mov[2:4]

	return [(mapping[start_coord[:len(start_coord)//2]], int(start_coord[len(start_coord)//2:])),(mapping[end_coord[:len(end_coord)//2]], int(end_coord[len(end_coord)//2:]))]

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


#Takes in the fen board and policy_output tensor and outputs a uci best move
def best_moves(board, policy_output, pro_output):
	mapping_re = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5:'e', 6:'f', 7:'g', 0:'h'}
	policy_output = policy_output.reshape(64,-1)
	mask = get_mask(board, pro_output)
	
	policy_output += 1e-10
	policy_output *= mask
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
	return move, max_coordinates, pro_move

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