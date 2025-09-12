import chess
import chess.svg
import numpy as np
import torch
import torch.nn.functional as F
import random
import math


#This function converts the current board into 14 current position planes (12 pieces + 2 repetition counters).
def board_to_array(board, turn, game_history=None):
	board_obj = chess.Board(board)
	
	array = np.zeros((14, 8, 8))
	piece_types = [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]
	
	# Fill piece planes (planes 0-11)
	# AlphaZero format: All white pieces (0-5), then all black pieces (6-11)
	for piece_idx, piece_type in enumerate(piece_types):

		white_squares = board_obj.pieces(piece_type, chess.WHITE)
		for square in white_squares:
			row = 7 - (square // 8)  # Convert to matrix coordinates
			col = square % 8
			array[piece_idx][row][col] = 1
		
		black_squares = board_obj.pieces(piece_type, chess.BLACK)
		for square in black_squares:
			row = 7 - (square // 8)  # Convert to matrix coordinates
			col = square % 8
			array[piece_idx + 6][row][col] = 1
	
	# Repetition Counters (2 planes: 12-13)
	if game_history:
		# Repetition Counter 1 (plane 12) - Normalized repetition count for draw detection
		# Include turn, castling rights, and en passant for accurate repetition rules
		position_key = ' '.join(board_obj.fen().split()[:4])  # board + turn + castling + en passant
		repetition_count = sum(1 for hist_pos in game_history 
		                      if ' '.join(hist_pos.fen().split()[:4]) == position_key)
		array[12] = np.full((8, 8), min(repetition_count, 3) / 3.0)  # Normalize to 0-1, cap at 3
		
		# Repetition Counter 2 (plane 13) - Halfmove clock for 50-move rule
		halfmove_clock = board_obj.halfmove_clock
		array[13] = np.full((8, 8), min(halfmove_clock, 50) / 50.0)  # Normalize and cap at 50
	else:
		# Default values when no history available
		array[12] = np.full((8, 8), 1/3.0)  # First occurrence (normalized)
		array[13] = np.full((8, 8), board_obj.halfmove_clock / 50.0)  # Real halfmove clock
	
	return torch.tensor(array)

def board_to_full_alphazero_input(current_board, game_history=None):
	"""
	Creates the full AlphaZero input: 119 planes total
	- 112 planes: 8 time steps × 14 planes each (12 pieces + 2 repetition counters)
	- 7 planes: Current game state information
	
	Args:
		current_board: chess.Board object OR FEN string for current position
		game_history: List of chess.Board objects representing game history
	
	Returns:
		torch.Tensor of shape (119, 8, 8)
	"""
	
	# Convert current_board to chess.Board object if it's a string
	if isinstance(current_board, str):
		current_board = chess.Board(current_board)
	
	if game_history is None:
		game_history = []
	
	# Ensure we have at least the current board in history
	full_history = game_history + [current_board]
	
	# Get exactly 8 positions, padding with zeros if needed
	last_8_positions = []
	for i in range(8):
		history_index = len(full_history) - 8 + i
		if history_index >= 0:
			last_8_positions.append(full_history[history_index])
		else:
			last_8_positions.append(None)  # Will be zero-padded
	
	# Create arrays for each time step (112 planes total)
	position_arrays = []
	for i, board_pos in enumerate(last_8_positions):
		if board_pos is not None:
			# Real position with proper history slice
			relevant_history = full_history[:len(full_history)-8+i+1]
			position_array = board_to_array(board_pos.fen(), board_pos.turn, relevant_history)
		else:
			# Zero padding for missing history
			position_array = torch.zeros((14, 8, 8))
		position_arrays.append(position_array)
	
	# Concatenate all position arrays (112 planes)
	position_planes = torch.cat(position_arrays, dim=0)
	
	# Get game state information for current position (7 planes)
	game_state_planes = board_to_game_state_array(current_board.fen(), current_board.turn)
	
	# Combine everything (112 + 7 = 119 planes)
	full_input = torch.cat([position_planes, game_state_planes], dim=0)
	
	return full_input

def board_to_game_state_array(board, turn):
	board_obj = chess.Board(board)
	
	# Create 7 planes for game state information
	array = np.zeros((7, 8, 8))
	
	# SECTION 2: GAME STATE INFORMATION (7 planes)
	
	# Castling Rights (4 planes: 0-3)
	# White King-side Castling (plane 0)
	if board_obj.has_kingside_castling_rights(chess.WHITE):
		array[0] = np.ones((8, 8))
	
	# White Queen-side Castling (plane 1)
	if board_obj.has_queenside_castling_rights(chess.WHITE):
		array[1] = np.ones((8, 8))
	
	# Black King-side Castling (plane 2)
	if board_obj.has_kingside_castling_rights(chess.BLACK):
		array[2] = np.ones((8, 8))
	
	# Black Queen-side Castling (plane 3)
	if board_obj.has_queenside_castling_rights(chess.BLACK):
		array[3] = np.ones((8, 8))
	
	# En Passant (1 plane: 4)
	if board_obj.ep_square is not None:
		ep_row = 7 - (board_obj.ep_square // 8)
		ep_col = board_obj.ep_square % 8
		array[4][ep_row][ep_col] = 1
	
	# Current Player Color (plane 5)
	if board_obj.turn == chess.WHITE:
		array[5] = np.ones((8, 8))
	else:
		array[5] = np.zeros((8, 8))
	
	# Total Move Count (plane 6) - normalized
	move_count = board_obj.fullmove_number / 100.0
	array[6] = np.full((8, 8), move_count)
	
	return torch.tensor(array)

def board_to_legal_policy_hash(board, policy):
	
	policy = policy.reshape(8,8,73)
	legal_moves = list(board.legal_moves)
	policy_distribution = {}
	
	for move in legal_moves:
		try:
			row, col, plane = uci_to_policy_index(str(move))
			policy_distribution[str(move)] = policy[row, col, plane].item()
		except ValueError as e:
			print(f"   {move} -> ERROR: {e}")
	
	# Normalize the distribution
	policy_distribution = normalize_hash(policy_distribution)
	
	return policy_distribution



def legal_move_to_coord(leg_mov):
	mapping = {'a': 1, 'b': 2, 'c': 3, 'd' : 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
	start_coord = leg_mov[:2]
	end_coord = leg_mov[2:4]


	return [(mapping[start_coord[:len(start_coord)//2]], int(start_coord[len(start_coord)//2:])),(mapping[end_coord[:len(end_coord)//2]], int(end_coord[len(end_coord)//2:]))] if len(leg_mov) == 4 else [(mapping[start_coord[:len(start_coord)//2]], int(start_coord[len(start_coord)//2:])),(mapping[end_coord[:len(end_coord)//2]], int(end_coord[len(end_coord)//2:])), leg_mov[-1]] 


def normalize_hash(inp_hash):
	sum_ = 0
	for i in inp_hash:
		sum_ += inp_hash[i]
	
	# Handle edge case where sum is zero
	if sum_ == 0:
		# Return uniform distribution
		uniform_prob = 1.0 / len(inp_hash) if len(inp_hash) > 0 else 0
		for i in inp_hash:
			inp_hash[i] = uniform_prob
	else:
		for i in inp_hash:
			inp_hash[i] /= sum_
	
	return inp_hash

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

def uci_to_policy_index(uci_move, board_state=None):
	"""
	Convert a UCI move string to AlphaZero policy tensor indices (row, col, plane).
	
	Args:
		uci_move: UCI move string (e.g., "e2e4", "e1g1", "e7e8q")
		board_state: Optional chess.Board object for move validation
		
	Returns:
		tuple: (row, col, plane) indices for the 8x8x73 policy tensor
	"""
	
	# Direction mappings (clockwise from North)
	# Note: In our coordinate system, row 0 = rank 8, so North = negative row direction
	DIRECTIONS = {
		(-1, 0): 0,  # North (up the board, decreasing row)
		(-1, 1): 1,  # Northeast  
		(0, 1): 2,   # East
		(1, 1): 3,   # Southeast
		(1, 0): 4,   # South (down the board, increasing row)
		(1, -1): 5,  # Southwest
		(0, -1): 6,  # West
		(-1, -1): 7  # Northwest
	}
	
	# Knight move patterns (original mapping)
	KNIGHT_MOVES = [
		(2, 1),   # 0
		(1, 2),   # 1
		(-1, 2),  # 2
		(-2, 1),  # 3
		(-2, -1), # 4
		(-1, -2), # 5
		(1, -2),  # 6
		(2, -1)   # 7
	]
	
	# Parse UCI move
	from_square = uci_move[:2]
	to_square = uci_move[2:4]
	promotion = uci_move[4:] if len(uci_move) > 4 else None
	
	# Convert squares to coordinates
	def square_to_coord(square):
		file = ord(square[0]) - ord('a')  # a=0, b=1, ..., h=7
		rank = int(square[1]) - 1         # 1=0, 2=1, ..., 8=7
		row = 7 - rank                    # Convert to matrix coords (rank 8 = row 0)
		col = file
		return row, col
	
	from_row, from_col = square_to_coord(from_square)
	to_row, to_col = square_to_coord(to_square)
	
	# Calculate movement vector
	d_row = to_row - from_row
	d_col = to_col - from_col
	
	# Check if it's a knight move
	if (d_row, d_col) in KNIGHT_MOVES:
		knight_index = KNIGHT_MOVES.index((d_row, d_col))
		plane = 56 + knight_index
		return from_row, from_col, plane
	
	# Check if it's an underpromotion
	if promotion and promotion.lower() in ['n', 'b', 'r']:
		piece_map = {'n': 0, 'b': 1, 'r': 2}
		piece_index = piece_map[promotion.lower()]
		
		# Determine direction for underpromotion
		if d_col == 0:
			direction_index = 0  # Forward
		elif d_col == -1:
			direction_index = 1  # Diagonal-left
		elif d_col == 1:
			direction_index = 2  # Diagonal-right
		else:
			raise ValueError(f"Invalid underpromotion move: {uci_move}")
		
		plane = 64 + direction_index * 3 + piece_index
		return from_row, from_col, plane
	
	# Must be a queen-line move
	# Normalize direction vector
	def gcd(a, b):
		while b:
			a, b = b, a % b
		return abs(a)
	
	if d_row == 0 and d_col == 0:
		raise ValueError(f"Invalid move (no movement): {uci_move}")
	
	# Find the direction and distance
	if d_row == 0:
		# Horizontal move
		direction = (0, 1) if d_col > 0 else (0, -1)
		distance = abs(d_col)
	elif d_col == 0:
		# Vertical move  
		direction = (-1, 0) if d_row < 0 else (1, 0)  # Fixed: d_row < 0 means North (up)
		distance = abs(d_row)
	else:
		# Diagonal move
		if abs(d_row) != abs(d_col):
			raise ValueError(f"Invalid diagonal move: {uci_move}")
		# Normalize direction vector for diagonals
		direction = (1 if d_row > 0 else -1, 1 if d_col > 0 else -1)
		distance = abs(d_row)  # or abs(d_col), they're equal for diagonals
	
	if distance > 7:
		raise ValueError(f"Move distance too large: {uci_move}")
	
	# Map direction to index
	if direction not in DIRECTIONS:
		raise ValueError(f"Invalid direction: {direction}")
	
	direction_index = DIRECTIONS[direction]
	plane = direction_index * 7 + (distance - 1)
	
	return from_row, from_col, plane


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