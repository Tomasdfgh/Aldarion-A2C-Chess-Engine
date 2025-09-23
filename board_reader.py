import chess
import numpy as np
import torch
import torch.nn.functional as F


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
			last_8_positions.append(None)
	
	# Create arrays for each time step (112 planes total)
	position_arrays = []
	for i, board_pos in enumerate(last_8_positions):
		if board_pos is not None:
			relevant_history = full_history[:len(full_history)-8+i+1]
			position_array = board_to_array(board_pos.fen(), board_pos.turn, relevant_history)
		else:
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

def create_legal_move_mask(board):
	"""
	Create a mask for legal moves in 8x8x73 format
	Returns tensor with True for legal moves, False for illegal moves
	"""
	mask = torch.zeros(8, 8, 73, dtype=torch.bool)
	legal_moves = list(board.legal_moves)
	
	# Safeguard: Check if position has legal moves (avoid mate/stalemate)
	if len(legal_moves) == 0:
		# Game is over - this shouldn't be in training data
		# Return all-False mask and let caller handle it
		return mask
	
	for move in legal_moves:
		try:
			row, col, plane = uci_to_policy_index(str(move))
			mask[row, col, plane] = True
		except ValueError:
			# Skip moves that can't be encoded
			continue
	
	return mask


def board_to_legal_policy_hash(board, policy_logits):
	"""
	Takes the board raw policy logits and converting it to a hashmap of legal move probabilities
	"""
	
	# This chunk of code grabs the board, finds all legal moves and zero out all the illegal moves in the policy and normalizes it
	legal_mask = create_legal_move_mask(board)
	masked_logits = policy_logits.clone()
	masked_logits[legal_mask.flatten() == 0] = -float('inf')
	policy_probs = F.softmax(masked_logits, dim=0).reshape(8, 8, 73)
	
	# Converts the policy to a hashmap of legal moves
	legal_moves = list(board.legal_moves)
	policy_distribution = {}
	for move in legal_moves:
		try:
			row, col, plane = uci_to_policy_index(str(move), board.turn)
			policy_distribution[str(move)] = policy_probs[row, col, plane].item()
		except ValueError as e:
			print(f"{move} -> ERROR: {e}")
	
	return policy_distribution


def uci_to_policy_index(uci_move, current_player_turn=chess.WHITE):
	"""
	This is one tricky motherfucker, but this function is based on the implementation of the
	actual alphazero paper. This converts the uci chess string like 'g1h3' which
	represents a move from g1 to h3 and convert it to the row, col, and plane index in the 8 by 8 by 73 policy tensor.
	If confused, look at how alphazero encodes their move. It is very specific and can be changed based on
	implementation.
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
	
	# Apply side-to-move normalization for Black
	if current_player_turn == chess.BLACK:
		from_row = 7 - from_row
		to_row = 7 - to_row
	
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