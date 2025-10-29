import chess
from . import board as br
import torch
import numpy as np
import math
import os
import time

'''
This script is the implementation of monte carlo tree search. Classical 
MCTS is not a deep learning algorithm that follows 4 basic steps:

1. Selection
2. Expansion
3. Simulation
4. Backprop

With AlphaZero however, there is a neural network involved. Usually, there
is a simulation phase where roll outs occur and random games are played until
completion. The results of the random games are saved and backprop back up the
tree. In Alphazero, this does not happen. Instead of roll outs, the state is
just pass into the NN and the value and policy is obtained there. Because of that,
in this code, the simulation phase does not exist and is absorb into the expansion
stage.

One thing to look at though is that during selfplay, both teams share the same game
tree. That means at every generation, the perspective of the game tree is flipped.
This makes sense because if I choose a move to make, its now my opponent's turn.

The way that I implemented this is I set the terminal state to have a return of -1.
I am not sure if this is right, but it makes sense logically I think. Terminal states
can have two outcomes: draw or non draw. If its a non draw, then the terminal state will
always be the parent node making that move winning. And since the node represents the
current player's perspective, the losing node recieves a -1. and when that value is back-
prop up the tree, the parent node then recieve a 1 (which is good because they made that
winning move). But then this also means that when I calculate the UCB score, it has to be
-q + u instead of q + u because from the parent's perspective, it will visit the child with
the highest ucb score and that means we have to negate their value (hence -q).

The reason why I brought this up is because I used to have a return of -1 for non draw
terminal states and q + u for UCB calculation and most games would end in draws, which
was strange. I then found out that the problem is that the tree sees a "mate in 1" move, but
simply refuses to make that move.


=============================================================
These Functions below are general Utility Functions for MCTS
=============================================================
'''

class MTCSNode:
	def __init__(self, team, state, action, n, w, q, p, parent=None):
		self.team = team			# bool with True for white
		self.state = state  		# board fen string
		self.action = action		# last state's action to this state
		self.N = n		  			# Contains the number of of times action a has been taken from state s
		self.W = w		  			# The Total Value of the next state
		self.Q = q		  			# W/N
		self.P = p		  			# The prior probability of selecting action a
		self.parent = parent		# Parent node reference
		self.children = []  		# List of all children
		self.is_expanded = False 	# Track if node has been expanded

	def add_child(self, child):
		self.children.append(child)
		child.parent = self

	def is_leaf(self):
		return len(self.children) == 0

	def is_terminal(self):
		board = chess.Board(self.state)
		return board.is_game_over()

def calculate_ucb(parent, child, c_puct=2.0):
	q = child.Q
	u = c_puct * child.P * math.sqrt(max(1, parent.N)) / (1 + child.N)
	return -q + u

def select_node(root, c_puct=2.0):
	"""
	Traverse tree using UCB to find leaf node for expansion
	Returns the leaf node to expand
	"""
	current = root
	
	while not current.is_leaf() and not current.is_terminal():
		best_child = None
		best_ucb = float('-inf')
		
		for child in current.children:
			ucb_score = calculate_ucb(current, child, c_puct)
			if ucb_score > best_ucb:
				best_ucb = ucb_score
				best_child = child
		
		current = best_child

	return current

def build_leaf_history(node, original_game_history, max_history=7):
	"""
	Build correct history for a leaf node by combining original game history 
	with the MCTS tree path leading to this node.
	"""

	mcts_path_fens = []
	current = node.parent
	
	while current is not None:
		mcts_path_fens.append(current.state)
		current = current.parent

	mcts_path_fens.reverse()
	combined_history = []
	combined_history.extend(original_game_history)
	for fen in mcts_path_fens:
		combined_history.append(chess.Board(fen))
	
	if len(combined_history) > max_history:
		combined_history = combined_history[-max_history:]
	
	return combined_history

def add_dirichlet_noise(policy_dict, alpha=0.3, noise_weight=0.25):
	"""
	Add Dirichlet noise to policy probabilities (AlphaZero paper)
	Uses alpha=0.3 and noise_weight=0.25 (AlphaZero standard parameters)
	"""
	moves = list(policy_dict.keys())
	probs = list(policy_dict.values())
	
	if len(moves) == 0:
		return policy_dict
	noise = np.random.dirichlet([alpha] * len(moves))

	noisy_policy = {}
	for i, move in enumerate(moves):
		noisy_policy[move] = (1 - noise_weight) * probs[i] + noise_weight * noise[i]
	
	return noisy_policy

def expand_node(node, model, device, game_history=None):
	"""
	Expand node by adding children for all legal moves
	Returns the value from the neural network evaluation
	"""
	if node.is_terminal() or node.is_expanded:
		return 0.0 # <-- this 0 here is not a return for terminal states
	
	board = chess.Board(node.state)
	with torch.no_grad():

		if game_history is None:
			game_history = []
		
		leaf_history = build_leaf_history(node, game_history)
		input_tensor = br.board_to_full_alphazero_input(board, leaf_history)
		input_tensor = input_tensor.unsqueeze(0).float().to(device)
		
		policy_logits, value = model(input_tensor)
		policy_logits = policy_logits.squeeze(0)
		value = value.squeeze(0).item()
	
		policy_dict = br.board_to_legal_policy_hash(board, policy_logits.cpu())
	
	# Creating the child of that node and putting the model's output into it
	for move_str in policy_dict.keys():
		move = chess.Move.from_uci(move_str)
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
		
		# Flip value for each generation up the tree because every new
		# generation is a change in player's perspective
		value = -value
		current = current.parent

def get_alphazero_temperature(move_number, base_temperature=1.0):
	"""
	Get temperature with two-phase schedule:
	- Temperature = base_temperature for moves 1-30
	- Temperature = 0.0 after move 30
	"""
	if move_number <= 30:
		return base_temperature
	else:
		return 0.0

def mcts_search(root, model, num_simulations, device, game_history=None, add_root_noise=False, c_puct=2.0):
	"""
	Run MCTS for num_simulations iterations
	Returns root node with updated statistics
	"""

	if not root.is_expanded and not root.is_terminal():
		_ = expand_node(root, model, device, game_history)
	
	# Apply fresh Dirichlet noise only to root at every move (AlphaZero paper)
	if add_root_noise and len(root.children) > 0:

		# Create policy dict from children's current priors
		policy_dict = {}
		for child in root.children:
			policy_dict[child.action] = child.P
		
		noisy_policy = add_dirichlet_noise(policy_dict)
		
		for child in root.children:
			child.P = noisy_policy[child.action]
	
	# Run MCTS simulations
	for _ in range(num_simulations): 

		leaf = select_node(root, c_puct)
		if not leaf.is_terminal():
			value = expand_node(leaf, model, device, game_history)
		else:
			board = chess.Board(leaf.state)
			if board.is_checkmate():
				value = -1.0 	# <---- Why this number is -1 and not 1 is discussed at the beginning of this script
			else:
				value = 0.0

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
	
	# Deterministically choose most visited
	if temperature == 0:
		best_idx = np.argmax(visits)
		for i, move in enumerate(moves):
			move_probs[move] = 1.0 if i == best_idx else 0.0

	else:
		#Apply Temperature
		visits = np.array(visits, dtype=np.float32)
		if temperature != 1.0:
			visits = visits ** (1.0 / temperature)
		
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
	Returns tuple of (selected_move, selected_child_node) for subtree reuse
	"""
	move_probs = get_move_probabilities(root, temperature)
	
	if not move_probs:
		return None, None
	
	moves = list(move_probs.keys())
	probs = list(move_probs.values())
	
	# Ensure probabilities sum to 1 (fix numerical issues)
	probs = np.array(probs)
	if probs.sum() > 0:
		probs = probs / probs.sum()
	else:
		probs = np.ones(len(probs)) / len(probs)

	#Get the move and the child
	selected_move = np.random.choice(moves, p=probs)
	selected_child = None
	for child in root.children:
		if child.action == selected_move:
			selected_child = child
			break
	
	return selected_move, selected_child

'''
========================================================================================================================
								This Function is the main game loop meant only for selfplay
								-- ONlY SELFPLAY BECAUSE BOTH TEAMS SHARE THE SAME TREE --
========================================================================================================================
'''

def run_game(model, num_simulations, device, temperature=1.0, c_puct=2.0, max_game_length=800, current_game=None, total_games=None, process_id=None):
	"""
	Play a full game using MCTS with AlphaZero parameters, collecting training data
	Returns tuple of (training_data, ending_reason) where training_data is list of 
	(board_state, history_fens, move_probabilities, game_outcome) tuples

	This function is the main function used for self-play but not evaluation since the same game tree is shared between
	the two teams.
	"""

	game_info = ""
	if current_game is not None and total_games is not None:
		game_info = f" ({current_game}/{total_games} games)"
	if process_id is not None:
		game_info = f" [Process {process_id}]" + game_info
	print(f"Starting new game...{game_info}")
	
	# Initialize game
	board = chess.Board()
	game_history = []  # Keep as Board objects for NN encoding
	training_data = []
	move_count = 0
	root = None
	
	while not board.is_game_over() and move_count < max_game_length:

		print(f"Move {move_count + 1}, {'White' if board.turn else 'Black'} to move")
		
		if root is None:
			root = MTCSNode(
				team=board.turn,
				state=board.fen(),
				action=None,
				n=0,
				w=0.0,
				q=0.0,
				p=1.0
			)
			print("Created fresh MCTS root")
		else:
			print(f"Reusing MCTS subtree (N={root.N}, children={len(root.children)})")
		
		root = mcts_search(root, model, num_simulations, device, game_history, add_root_noise=True, c_puct=c_puct)

		alphazero_temperature = get_alphazero_temperature(board.fullmove_number, base_temperature=temperature)
		move_probs = get_move_probabilities(root, temperature=alphazero_temperature)
		history_fens = [b.fen() for b in game_history[-7:]] if len(game_history) >= 7 else [b.fen() for b in game_history[:]]
		
		#Each of the training data here is still missing the value, which will be added once the game is over
		training_data.append((board.fen(), history_fens, move_probs.copy()))
		selected_move, selected_child = select_move(root, alphazero_temperature)
		
		print(f"Selected move: {selected_move} (temp={alphazero_temperature}){game_info} \n")
		move_obj = chess.Move.from_uci(selected_move)
		board.push(move_obj)
		game_history.append(board.copy())
		
		# Promote selected child to new root for subtree reuse and clear out the other branches
		if selected_child is not None:
			selected_child.parent = None
			root = selected_child
		else:
			root = None
		
		move_count += 1
	
	# Determine game outcome and ending reason
	if board.is_checkmate():
		game_outcome = -1 if board.turn else 1
		winner = 'Black' if game_outcome == -1 else 'White'
		ending_reason = f"{winner} wins by checkmate"
		print(f"Game over: {ending_reason}")
	elif board.is_stalemate():
		game_outcome = 0
		ending_reason = "Draw by stalemate"
		print(f"Game over: {ending_reason}")
	elif board.is_insufficient_material():
		game_outcome = 0 
		ending_reason = "Draw by insufficient material"
		print(f"Game over: {ending_reason}")
	elif board.is_seventyfive_moves():
		game_outcome = 0
		ending_reason = "Draw by 75-move rule"
		print(f"Game over: {ending_reason}")
	elif board.is_fivefold_repetition():
		game_outcome = 0
		ending_reason = "Draw by fivefold repetition"
		print(f"Game over: {ending_reason}")
	elif move_count >= max_game_length:
		game_outcome = 0
		ending_reason = f"Draw by {max_game_length}-ply limit"
		print(f"Game over: {ending_reason}")
	else:
		# This should not happen if game loop only continues while conditions are met
		game_outcome = 0  # Fallback draw
		ending_reason = "Draw by unexpected end condition"
		print(f"Game over: {ending_reason}")
	
	# Convert training data to final format with game outcomes
	final_training_data = []
	for board_state, history_fens, move_probs in training_data:

		side_to_move = chess.Board(board_state).turn
		if side_to_move:
			outcome = game_outcome
		else:
			outcome = -game_outcome
		
		final_training_data.append((board_state, history_fens, move_probs, outcome))
	
	print(f"Game completed in {move_count} moves")
	print(f"Generated {len(final_training_data)} training examples")
	return final_training_data, ending_reason

'''
========================================================================================================================
									These functions are meant for evaluations
							-- EVALUATIONS BECAUSE EACH TEAM HAS THEIR OWN GAME TREE --
========================================================================================================================
'''

def return_move_and_child(model, board_fen, num_simulations, device, game_history=None, existing_tree=None, temperature=0.0, c_puct=2.0):
	"""
	Get the best move for a given position using MCTS with optional tree reuse
	"""
	
	board = chess.Board(board_fen)
	if existing_tree is not None and existing_tree.state == board_fen:
		root = existing_tree
		print(f"Reusing MCTS tree (N={root.N}, children={len(root.children)})")
	else:
		root = None
		
		#This if statement here is only for evaluation games because of the two unique trees. once a move has been made on the other tree
		#So once a move has been made by the other team, this team will then scan its children to see which of its children are the current
		#child now, then that child will become the new root
		if existing_tree is not None:
			for child in existing_tree.children:
				if child.state == board_fen:
					child.parent = None
					root = child
					print(f"Promoting child to root (N={root.N}, children={len(root.children)})")
					break
		
		if root is None:
			root = MTCSNode(
				team=board.turn,
				state=board_fen,
				action=None,
				n=0,
				w=0.0,
				q=0.0,
				p=1.0
			)
			print("Created fresh MCTS root")
	
	# Run MCTS (no noise for evaluation, only for self-play training)
	root = mcts_search(root, model, num_simulations, device, game_history, add_root_noise=False, c_puct=c_puct)
	
	if temperature == 0.0:
		best_child = max(root.children, key=lambda x: x.N)
		best_move = best_child.action
		selected_child = best_child
	else:
		# Should never be this btw, eval games should be deterministic
		best_move, selected_child = select_move(root, temperature)
	
	if selected_child is not None:
		selected_child.parent = None
	
	return best_move, selected_child


def play_single_evaluation_game(white_model, black_model, num_simulations, device, game_id, white_is_new, old_model_path, new_model_path, process_id, game_num, total_games, max_game_length=800):
	"""
	Play a single competitive game between two models with private MCTS trees
	"""
	
	start_time = time.time()
	
	try:
		board = chess.Board.from_chess960_pos(np.random.randint(0, 960))
		game_history = []
		move_count = 0
		
		print(f"Process {process_id}: Game {game_num}/{total_games}: Starting FEN: {board.fen()}")
		print('\n')
		
		white_tree = None
		black_tree = None
		
		while not board.is_game_over() and move_count < max_game_length:
			current_model = white_model if board.turn else black_model
			current_tree = white_tree if board.turn else black_tree
			current_player = "White" if board.turn else "Black"
			print(f"Move {move_count + 1}, {current_player} to move")
			
			try:
				# Get best move using private tree (with subtree reuse)
				best_move, selected_child = return_move_and_child(
					model=current_model,
					board_fen=board.fen(),
					num_simulations=num_simulations,
					device=device,
					game_history=game_history,
					existing_tree=current_tree,
					temperature=0.0  # Deterministic play for evaluation
				)
				
				if best_move is None:
					print(f"Process {process_id}: Game {game_num}/{total_games}: No legal moves available")
					break
				
				model_info = "New" if (board.turn and white_is_new) or (not board.turn and not white_is_new) else "Old"
				print(f"Selected move: {best_move} ({model_info} model), [Process {process_id}] - Game {game_num}/{total_games}")
				print()
				
				# Update the tree for the current player
				if board.turn:
					white_tree = selected_child
				else:
					black_tree = selected_child

				move_obj = chess.Move.from_uci(best_move)
				board.push(move_obj)
				game_history.append(board.copy())
				move_count += 1
				
			except Exception as e:
				print(f"Process {process_id}: Game {game_num}/{total_games}: Error during {current_player} move: {e}")
				break
		
		# Determine game result
		if board.is_checkmate():
			result = -1.0 if board.turn else 1.0
			result_str = "Black wins by checkmate" if result == -1.0 else "White wins by checkmate"
		elif board.is_stalemate():
			result = 0.0
			result_str = "Draw by stalemate"
		elif board.is_insufficient_material():
			result = 0.0
			result_str = "Draw by insufficient material"
		elif board.is_seventyfive_moves():
			result = 0.0
			result_str = "Draw by 75-move rule"
		elif board.is_fivefold_repetition():
			result = 0.0
			result_str = "Draw by fivefold repetition"
		elif move_count >= max_game_length:
			result = 0.0
			result_str = f"Draw by {max_game_length}-ply limit"
		else:
			result = 0.0
			result_str = "Draw by unexpected end"
		
		game_time = time.time() - start_time
		
		game_result = {
			'game_id': game_id,
			'result': result,
			'result_str': result_str,
			'move_count': move_count,
			'game_time_seconds': game_time,
			'white_model': os.path.basename(new_model_path) if white_is_new else os.path.basename(old_model_path),
			'black_model': os.path.basename(old_model_path) if white_is_new else os.path.basename(new_model_path),
			'white_is_new': white_is_new
		}
		
		print(f"Process {process_id}: Game {game_num}/{total_games}: {result_str} in {move_count} moves ({game_time:.1f}s)")
		return game_result
		
	except Exception as e:
		print(f"Process {process_id}: Game {game_num}/{total_games}: Fatal error: {e}")
		return {
			'game_id': game_id,
			'error': str(e),
			'game_time_seconds': time.time() - start_time
		}