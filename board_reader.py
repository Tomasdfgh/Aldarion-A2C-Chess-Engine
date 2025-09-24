import chess
import numpy as np
import torch
import torch.nn.functional as F


'''
This script is meant to analyze the board data and convert the board and history
of the game to convert it to inputs for the model based on Alphazero's implementation.
It is also meant to analyze the outputs of the model and convert it to proper
legal move distributions to be played on the board.

=================================================================
These Functions below are meant to represent the board as Inputs
=================================================================

Detailed below is also how the current state and the game history is being represented
as the input into the model:

The model takes in a tensor of size (8, 8, 119). The first 112 planes is used to encode
position history of the 8 time steps where each time steps uses 14 planes each. So
8 * 14 = 112 planes.

For each time step, the first 12 planes are used to encode the pieces on the board. Since
there are 12 types of pieces (for both teams), each piece gets a plane where their current
position is a 1 and 0 elsewhere.

The last 2 planes are repetition counter 1 and 2 used to detect draws:
Plane 12 (Repetition Counter 1): Counts how many times the current position has appeared 
in the game history. It compares the FEN position (board state + turn + castling rights + 
en passant) against all previous positions. The count is capped at 3 and normalized to 0-1
range (dividing by 3). This helps detect threefold repetition draws.

Plane 13 (Repetition Counter 2): Tracks the halfmove clock (moves since last pawn move or
capture), normalized to 0-1 by dividing by 50. This helps detect the 50-move rule for draws.

The last 7 planes is to encode game state information.

The first 4 planes is filled with all 1s or 0s if the current player's king side castling is available,
if the current player's queen-side castling is available (honestly i didnt even know you can castle queens),
if the opponent's king side castling is available, and if the opponent's queen-side castling is available.

The 5th plane has a single 1 at the en passant target position if an en passant is available, 
if not all 0s. The last 2 planes is to denote the current player's colour (all 1s if white, 
0s if black), and total move count (which is a normalized move counter).

IMPORTANT: AlphaZero always plays from the current player's perspective. This means the board is 
always oriented so that the current player's pieces face downward (towards rank 1). When it's Black's 
turn, the entire board representation is flipped 180 degrees so Black's pieces face downward.
'''

def board_to_array(board, game_history=None):
	'''
	This function is used to encode the pieces in a board. It is used to convert the board
	into 14 current position planes (12 pieces + 2 repetition counters). This function should
	be used for each of the 8 time steps used to encode the input
	'''
	board_obj = chess.Board(board)
	array = np.zeros((14, 8, 8))
	piece_types = [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]
	
	# AlphaZero format: Current player pieces (0-5), then opponent pieces (6-11)
	current_player = board_obj.turn
	opponent = not current_player
	
	for piece_idx, piece_type in enumerate(piece_types):
		current_player_squares = board_obj.pieces(piece_type, current_player)
		for square in current_player_squares:
			row = 7 - (square // 8)
			col = square % 8

			# Flip board if current player is Black
			if current_player == chess.BLACK:
				row = 7 - row
			array[piece_idx][row][col] = 1
		
		# Opponent's pieces go in planes 6-11
		opponent_squares = board_obj.pieces(piece_type, opponent)
		for square in opponent_squares:
			row = 7 - (square // 8)
			col = square % 8
			if current_player == chess.BLACK:
				row = 7 - row
			array[piece_idx + 6][row][col] = 1

	if game_history:
		position_key = ' '.join(board_obj.fen().split()[:4])
		repetition_count = sum(1 for hist_pos in game_history 
		                      if ' '.join(hist_pos.fen().split()[:4]) == position_key)
		array[12] = np.full((8, 8), min(repetition_count, 3) / 3.0)
		
		halfmove_clock = board_obj.halfmove_clock
		array[13] = np.full((8, 8), min(halfmove_clock, 50) / 50.0)
	else:
		array[12] = np.full((8, 8), 1/3.0)
		array[13] = np.full((8, 8), board_obj.halfmove_clock / 50.0)
	
	return torch.tensor(array)

def board_to_game_state_array(board):
	'''
	This function takes the CURRENT (only) board, and create a 7 plane of 8 by 8 representation.
	This is used as the last part of the total input representation.
	'''
	board_obj = chess.Board(board)
	
	array = np.zeros((7, 8, 8))
	current_player = board_obj.turn
	opponent = not current_player
	

	if board_obj.has_kingside_castling_rights(current_player):
		array[0] = np.ones((8, 8))

	if board_obj.has_queenside_castling_rights(current_player):
		array[1] = np.ones((8, 8))
	
	if board_obj.has_kingside_castling_rights(opponent):
		array[2] = np.ones((8, 8))
	
	if board_obj.has_queenside_castling_rights(opponent):
		array[3] = np.ones((8, 8))
	
	if board_obj.ep_square is not None:
		ep_row = 7 - (board_obj.ep_square // 8)
		ep_col = board_obj.ep_square % 8

		if current_player == chess.BLACK:
			ep_row = 7 - ep_row
		array[4][ep_row][ep_col] = 1
	
	array[5] = np.ones((8, 8)) if current_player == chess.WHITE else np.zeros((8, 8))
	
	move_count = board_obj.fullmove_number / 100.0
	array[6] = np.full((8, 8), move_count)
	
	return torch.tensor(array)

def board_to_full_alphazero_input(current_board, game_history=None):
	"""
	Takes in the current board and game history. grab the last 7 time steps
	then it should convert that into an input vector that is 8 by 8 (chess board) by 119 planes.
	study up on why alphazero has 119 planes for its input for this to make sense.
	"""
	if isinstance(current_board, str):
		current_board = chess.Board(current_board)
	
	if game_history is None:
		game_history = []
	
	# Ensure we have at least the current board in history
	full_history = game_history + [current_board]
	
	last_8_positions = []
	for i in range(8):
		history_index = len(full_history) - 8 + i
		if history_index >= 0:
			last_8_positions.append(full_history[history_index])
		else:
			last_8_positions.append(None)
	
	position_arrays = []
	for i, board_pos in enumerate(last_8_positions):
		if board_pos is not None:
			relevant_history = full_history[:len(full_history)-8+i+1]
			position_array = board_to_array(board_pos.fen(), relevant_history)
		else:
			position_array = torch.zeros((14, 8, 8))
		position_arrays.append(position_array)
	

	position_planes = torch.cat(position_arrays, dim=0)
	game_state_planes = board_to_game_state_array(current_board.fen())
	full_input = torch.cat([position_planes, game_state_planes], dim=0) #<-- this should be 119 planes (similar to the actual alphazero paper). check here if not true
	
	return full_input


'''
=======================================================================
These Functions below are meant to convert outputs to move distribution
=======================================================================

Detailed below is how Alphazero represents the output distribution:
the model's policy head outputs 4672 logits, which can be broken down into
a tensor of shape (8, 8, 73). This means that there are 73 planes to
represent how moves are being made, and the 8 by 8 represents the board
and where the piece is "from".

The first 56 planes are for Queen-Line Moves where there are 8 possible 
directions (north, northeast, east, and all that), and there are 7
possible distances, so we got 7 * 8 = 56 planes. so for example, from
a (8, 8, 73) tensor, if the logit from plane 7 is chosen at position d1,
that means the piece from d1 is going north at a distance of 7.

The next 8 planes represents the Knight moves. They are simple:

Plane 56: Knight move (+2,+1) - 2 right, 1 up
Plane 57: Knight move (+1,+2) - 1 right, 2 up  
Plane 58: Knight move (-1,+2) - 1 left, 2 up
Plane 59: Knight move (-2,+1) - 2 left, 1 up
Plane 60: Knight move (-2,-1) - 2 left, 1 down
Plane 61: Knight move (-1,-2) - 1 left, 2 down
Plane 62: Knight move (+1,-2) - 1 right, 2 down
Plane 63: Knight move (+2,-1) - 2 right, 1 down

The next 9 planes are for Underpromotion moves (these are non queen promotions).
Queen promotions use the queen-line planes (North direction, 1 square). These are 
also basic so im not going to explain, instead I will just show which planes represent 
what:

Plane 64: Promote to Knight, Forward
Plane 65: Promote to Knight, Capture Left
Plane 66: Promote to Knight, Capture Right
Plane 67: Promote to Bishop, Forward
Plane 68: Promote to Bishop, Capture Left
Plane 69: Promote to Bishop, Capture Right
Plane 70: Promote to Rook, Forward
Plane 71: Promote to Rook, Capture Left
Plane 72: Promote to Rook, Capture Right

One more important note is that alphazero policy distribution plays from the current
player prespective. So if you the player is black, the board will have to be flipped
180 degrees, then a move can be selected.
'''

def create_legal_move_mask(board):
	"""
	Create a mask for legal moves in 8x8x73 format
	Returns tensor with True for legal moves, False for illegal moves
	If terminal state (no legal moves) then just return an all false mask
	"""

	mask = torch.zeros(8, 8, 73, dtype=torch.bool)
	legal_moves = list(board.legal_moves)
	if len(legal_moves) == 0:
		return mask
	
	for move in legal_moves:
		try:
			row, col, plane = uci_to_policy_index(str(move), board.turn)
			mask[row, col, plane] = True
		except ValueError:
			continue
	
	return mask


def board_to_legal_policy_hash(board, policy_logits):
	"""
	Takes the board raw policy logits and converting it to a hashmap of legal move probabilities
	"""
	
	# This chunk of code grabs the board, finds all legal moves and zero out all the illegal moves in the policy and normalizes it
	legal_mask = create_legal_move_mask(board)
	masked_logits = policy_logits.clone().view(-1)
	masked_logits[~legal_mask.flatten()] = -float('inf')
	policy_probs = F.softmax(masked_logits, dim=0).view(8, 8, 73)
	
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
	
	from_square = uci_move[:2]
	to_square = uci_move[2:4]
	promotion = uci_move[4:] if len(uci_move) > 4 else None
	
	# Convert squares to coordinates
	def square_to_coord(square):
		file = ord(square[0]) - ord('a')
		rank = int(square[1]) - 1
		row = 7 - rank
		col = file
		return row, col
	
	from_row, from_col = square_to_coord(from_square)
	to_row, to_col = square_to_coord(to_square)
	
	# Flip board so if ur black, black faces downward direction
	if current_player_turn == chess.BLACK:
		from_row = 7 - from_row
		to_row = 7 - to_row
	
	d_row = to_row - from_row
	d_col = to_col - from_col

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
	
	if (d_row, d_col) in KNIGHT_MOVES:
		knight_index = KNIGHT_MOVES.index((d_row, d_col))
		plane = 56 + knight_index
		return from_row, from_col, plane
	
	if promotion and promotion.lower() in ['n', 'b', 'r']:
		piece_map = {'n': 0, 'b': 1, 'r': 2}
		piece_index = piece_map[promotion.lower()]
		
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

	# Note: row 0 = rank 8, so North = negative row direction
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

	#Final checks to grab move vector
	if d_row == 0:
		direction = (0, 1) if d_col > 0 else (0, -1)
		distance = abs(d_col)
	elif d_col == 0:
		direction = (-1, 0) if d_row < 0 else (1, 0)
		distance = abs(d_row)
	else:
		# Normalize direction vector for diagonals
		direction = (1 if d_row > 0 else -1, 1 if d_col > 0 else -1)
		distance = abs(d_row)
	
	direction_index = DIRECTIONS[direction]
	plane = direction_index * 7 + (distance - 1)
	
	return from_row, from_col, plane