import chess
import board_reader as br
import torch
import numpy as np
import math
import random


class MTCSNode:
	def __init__(self, team, state, action, n, w, q, p, parent=None):
		self.team = team		# True for White and False for Black
		self.state = state  	# Contains the board state in fen string
		self.action = action	# Action from the last state to get to this state
		self.N = n		  		# Contains the number of of times action a has been taken from state s
		self.W = w		  		# The Total Value of the next state
		self.Q = q		  		# The mean value of the next state (W/N)
		self.P = p		  		# The prior probability of selecting action a
		self.parent = parent	# Parent node reference
		self.children = []  	# List of all children
		self.is_expanded = False # Track if node has been expanded

	def add_child(self, child):
		self.children.append(child)
		child.parent = self

	def is_leaf(self):
		return len(self.children) == 0

	def is_terminal(self):
		board = chess.Board(self.state)
		return board.is_game_over()

def print_tree(node, level=0):
	print("  " * level + node.state)
	print("  " * level + "Team: " + str(node.team))
	print("  " * level + "action: " + str(node.action))
	print("  " * level + "N: " + str(node.N))
	print("  " * level + "W: " + str(node.W))
	print("  " * level + "Q: " + str(node.Q))
	print("  " * level + "P: " + str(node.P))
	for child in node.children:
		print('\n')
		print_tree(child, level + 1)

def calculate_ucb(parent, child, c_puct=1.0):
	"""Calculate UCB1 score with PUCT formula"""
	if child.N == 0:
		return float('inf')
	
	exploration = c_puct * child.P * math.sqrt(parent.N) / (1 + child.N)
	return child.Q + exploration

def select_node(root):
	"""
	Traverse tree using UCB to find leaf node for expansion
	Returns the leaf node to expand
	"""
	current = root
	
	while not current.is_leaf() and not current.is_terminal():
		# Calculate UCB for all children and select best
		best_child = None
		best_ucb = float('-inf')
		
		for child in current.children:
			ucb_score = calculate_ucb(current, child)
			if ucb_score > best_ucb:
				best_ucb = ucb_score
				best_child = child
		
		current = best_child
	
	return current

def expand_node(node, model, device, game_history=None):
	"""
	Expand node by adding children for all legal moves
	Returns the expanded node and its policy/value from neural network
	"""
	if node.is_terminal() or node.is_expanded:
		return node, 0.0
	
	board = chess.Board(node.state)
	legal_moves = list(board.legal_moves)
	
	if len(legal_moves) == 0:
		return node, 0.0
	
	# Get neural network evaluation
	with torch.no_grad():
		# Convert board to input format
		if game_history is None:
			game_history = []
		
		input_tensor = br.board_to_full_alphazero_input(board, game_history)
		input_tensor = input_tensor.unsqueeze(0).float().to(device)
		
		# Get model predictions
		policy_logits, value = model(input_tensor)
		policy_logits = policy_logits.squeeze(0)
		value = value.squeeze(0).item()
		
		# Convert policy logits to legal move probabilities
		policy_dict = br.board_to_legal_policy_hash(board, policy_logits.cpu())
	
	# Create child nodes for each legal move
	for move_str in policy_dict.keys():
		move = chess.Move.from_uci(move_str)
		
		# Create new board state
		new_board = board.copy()
		new_board.push(move)
		new_state = new_board.fen()
		
		# Create child node
		prior_prob = policy_dict[move_str]
		child = MTCSNode(
			team=not node.team,  # Switch team
			state=new_state,
			action=move_str,
			n=0,
			w=0.0,
			q=0.0,
			p=prior_prob,
			parent=node
		)
		
		node.add_child(child)
	
	node.is_expanded = True
	return node, value

def simulate(node, model, device, game_history=None):
	"""
	Get value from neural network (no rollout needed with strong neural network)
	"""
	if node.is_terminal():
		board = chess.Board(node.state)
		if board.is_checkmate():
			# Return value from perspective of player to move
			return -1.0 if board.turn == chess.WHITE else 1.0
		else:
			# Draw
			return 0.0
	
	# Get value from neural network
	with torch.no_grad():
		if game_history is None:
			game_history = []
		
		board = chess.Board(node.state)
		input_tensor = br.board_to_full_alphazero_input(board, game_history)
		input_tensor = input_tensor.unsqueeze(0).float().to(device)
		
		_, value = model(input_tensor)
		value = value.squeeze(0).item()
		
		# Convert value to current player's perspective
		# Model outputs value from white's perspective, adjust if black to move
		if not board.turn:  # Black to move
			value = -value
	
	return value

def backpropagate(node, value):
	"""
	Update N, W, Q values up the tree
	"""
	current = node
	
	while current is not None:
		current.N += 1
		current.W += value
		current.Q = current.W / current.N if current.N > 0 else 0.0
		
		# Flip value for next level (opponent's perspective)
		value = -value
		current = current.parent

def mcts_search(root, model, num_simulations, device, game_history=None):
	"""
	Run MCTS for num_simulations iterations
	Returns root node with updated statistics
	"""
	for i in range(num_simulations):
		# Selection: traverse tree to leaf
		leaf = select_node(root)
		
		# Expansion and Simulation
		if not leaf.is_terminal():
			expanded_node, value = expand_node(leaf, model, device, game_history)
			if len(expanded_node.children) > 0:
				# If we expanded, select first child for simulation
				leaf = expanded_node.children[0]
			value = simulate(leaf, model, device, game_history)
		else:
			# Terminal node
			value = simulate(leaf, model, device, game_history)
		
		# Backpropagation
		backpropagate(leaf, value)
	
	return root

def get_move_probabilities(root, temperature=1.0):
	"""
	Get move probabilities based on visit counts and temperature
	"""
	if len(root.children) == 0:
		return {}
	
	move_probs = {}
	visits = []
	moves = []
	
	for child in root.children:
		visits.append(child.N)
		moves.append(child.action)
	
	if temperature == 0:
		# Deterministic - choose most visited
		best_idx = np.argmax(visits)
		for i, move in enumerate(moves):
			move_probs[move] = 1.0 if i == best_idx else 0.0
	else:
		# Apply temperature
		visits = np.array(visits, dtype=np.float32)
		if temperature != 1.0:
			visits = visits ** (1.0 / temperature)
		
		# Normalize
		if visits.sum() > 0:
			visits = visits / visits.sum()
		else:
			visits = np.ones(len(visits)) / len(visits)
		
		for move, prob in zip(moves, visits):
			move_probs[move] = prob
	
	return move_probs

def select_move(root, temperature=1.0):
	"""
	Select move based on MCTS visit counts and temperature
	"""
	move_probs = get_move_probabilities(root, temperature)
	
	if not move_probs:
		return None
	
	moves = list(move_probs.keys())
	probs = list(move_probs.values())
	
	# Ensure probabilities sum to 1 (fix numerical issues)
	probs = np.array(probs)
	if probs.sum() > 0:
		probs = probs / probs.sum()
	else:
		probs = np.ones(len(probs)) / len(probs)
	
	# Sample from probability distribution
	selected_move = np.random.choice(moves, p=probs)
	return selected_move

def run_game(model, temperature, num_simulations, device):
	"""
	Play a full game using MCTS, collecting training data
	Returns list of (board_state, move_probabilities, game_outcome) tuples
	"""
	print("Starting new game...")
	
	# Initialize game
	board = chess.Board()
	game_history = []
	training_data = []
	
	move_count = 0
	max_moves = 200  # Prevent infinite games
	
	while not board.is_game_over() and move_count < max_moves:
		print(f"Move {move_count + 1}, {'White' if board.turn else 'Black'} to move")
		
		# Create root node for current position
		root = MTCSNode(
			team=board.turn,
			state=board.fen(),
			action=None,
			n=0,
			w=0.0,
			q=0.0,
			p=1.0
		)
		
		# Run MCTS
		root = mcts_search(root, model, num_simulations, device, game_history)
		
		# Get move probabilities for training data
		move_probs = get_move_probabilities(root, temperature=1.0)  # Use temperature=1 for training
		
		# Store training data (board state and MCTS probabilities)
		training_data.append((board.fen(), move_probs.copy()))
		
		# Select and make move
		selected_move = select_move(root, temperature)
		if selected_move is None:
			print("No legal moves available")
			break
		
		print(f"Selected move: {selected_move}")
		
		# Apply move
		move_obj = chess.Move.from_uci(selected_move)
		board.push(move_obj)
		game_history.append(board.copy())
		
		move_count += 1
	
	# Determine game outcome
	if board.is_checkmate():
		# Winner gets +1, loser gets -1
		game_outcome = 1 if not board.turn else -1  # Opposite of current turn (who got checkmated)
		print(f"Game over: {'White' if game_outcome == 1 else 'Black'} wins by checkmate")
	elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
		game_outcome = 0  # Draw
		print("Game over: Draw")
	else:
		game_outcome = 0  # Max moves reached
		print("Game over: Maximum moves reached")
	
	# Convert training data to final format with game outcomes
	final_training_data = []
	for i, (board_state, move_probs) in enumerate(training_data):
		# Outcome from perspective of player who made the move
		player_turn = chess.Board(board_state).turn
		if player_turn:  # White
			outcome = game_outcome
		else:  # Black  
			outcome = -game_outcome
		
		final_training_data.append((board_state, move_probs, outcome))
	
	print(f"Game completed in {move_count} moves")
	print(f"Generated {len(final_training_data)} training examples")
	
	return final_training_data

def get_best_move(model, board_fen, num_simulations, device, game_history=None, temperature=0.0):
	"""
	Get the best move for a given position using MCTS
	
	Args:
		model: Neural network model
		board_fen: FEN string of current board position
		num_simulations: Number of MCTS simulations to run
		device: PyTorch device
		game_history: Optional list of chess.Board objects for history
		temperature: Temperature for move selection (0.0 = deterministic)
	
	Returns:
		tuple: (best_move_uci, move_probabilities_dict)
	"""
	board = chess.Board(board_fen)
	
	root = MTCSNode(
		team=board.turn,
		state=board_fen,
		action=None,
		n=0,
		w=0.0,
		q=0.0,
		p=1.0
	)
	
	# Run MCTS
	root = mcts_search(root, model, num_simulations, device, game_history)
	
	# Get move probabilities
	move_probs = get_move_probabilities(root, temperature)
	
	# Select best move
	if temperature == 0.0:
		# Deterministic: select most visited
		best_child = max(root.children, key=lambda x: x.N)
		best_move = best_child.action
	else:
		# Sample from distribution
		best_move = select_move(root, temperature)
	
	return best_move, move_probs

def test_board(model, device):
	"""
	Test function to verify MCTS is working
	"""
	print("Testing MCTS with starting position...")
	
	board = chess.Board()
	root = MTCSNode(
		team=True,
		state=board.fen(),
		action=None,
		n=0,
		w=0.0,
		q=0.0,
		p=1.0
	)
	
	# Run small MCTS search
	root = mcts_search(root, model, 50, device)
	
	print(f"Root visits: {root.N}")
	print("Top moves:")
	
	# Sort children by visit count
	children_sorted = sorted(root.children, key=lambda x: x.N, reverse=True)
	
	for i, child in enumerate(children_sorted[:5]):
		print(f"{i+1}. {child.action}: N={child.N}, Q={child.Q:.3f}, P={child.P:.3f}")
	
	return root