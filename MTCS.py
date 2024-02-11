#Monte Carlo Tree Search
import chess
import torch
import model as md
import board_reader as br
import board_display as bd
import os

class MTCSNode:
	def __init__(self, team, state, action, n, w, q, p):
		self.team = team		# True for White and False for Black
		self.state = state  	# Contains the board state in fen string
		self.action = action	# Action from the last state to get to this state
		self.N = n		  		# Contains the number of of times action a has been taken from state s
		self.W = w		  		# The Total Value of the next state
		self.Q = q		  		# The mean value of the next state (W/N)
		self.P = p		  		# The prior probability of selecting action a
		self.children = []  	# List of all children
		self.UCB = None

	def add_child(self, child):
		self.children.append(child)

def print_tree(node, level=0):
	print("  " * level + node.state)
	print("  " * level + "Team: " + str(node.team))
	print("  " * level + "action: " + str(node.action))
	print("  " * level + "N: " + str(node.N))
	print("  " * level + "W: " + str(node.W))
	print("  " * level + "Q: " + str(node.Q))
	print("  " * level + "P: " + str(node.P))
	print("  " * level + "UCB: " + str(node.UCB))
	for child in node.children:
		print('\n')
		print_tree(child, level + 1)

def UCB(parent, child, prnt = False):
	U = child.P * (parent.N) ** (1/2) / (child.N + 1)
	if prnt:
		print(child.P)
		print(parent.N)
		print(child.N)
		print("U: " + str(U))
	if child.N > 0:
		value_score = child.Q
	else:
		value_score = float('inf')
	return value_score + U

def expand(node, policy):
	policy = policy.reshape(64,-1)
	move_mapper = {}

	for i in chess.Board(node.state).legal_moves:
		coords = br.legal_move_to_coord(str(i))
		# if len(str(i)) > 4:
		# 	print(str(i))
		# 	print(coords)
		move_mapper[str(i)] = policy[(coords[0][0] + ((coords[0][1] - 1) * 8)) - 1][(coords[1][0] + ((coords[1][1] - 1) * 8)) -1].item()
	move_mapper = normalize_list(move_mapper)

	for i in chess.Board(node.state).legal_moves:
		board_temp = chess.Board(node.state)
		board_temp.push_uci(str(i))
		child_node = MTCSNode(board_temp.turn, board_temp.fen(), i, 0, 0, 0, move_mapper[str(i)])
		child_node.UCB = UCB(node, child_node)
		node.add_child(child_node)
	move_mapper = normalize_list(move_mapper)

def normalize_list(inp_hash):
	sum_ = 0
	for i in inp_hash:
		sum_ += inp_hash[i]
	for i in inp_hash:
		inp_hash[i] /= sum_
	return inp_hash


def run_simulation(root, sim, model):

	#Expand the root
	if len(root.children) == 0:
		policy, pro, value = model(torch.tensor(br.board_to_array(chess.Board(root.state).fen(), root.team)))
		expand(root, policy)


	#Begin Simulation
	for z in range(sim):

		search_path = [root]
		queue = [root]

		#select
		node = root
		while queue:

			current_node = queue.pop(0)

			#Select best child
			best_ucb = -float('inf')
			best_node = None
			for i in current_node.children:
				if i.UCB > best_ucb:
					best_ucb = i.UCB
					best_node = i
			if best_node is not None:
				node = best_node
				search_path.append(node)
				queue.append(node)

		#Expand
		policy, pro, value = model(torch.tensor(br.board_to_array(chess.Board(search_path[-1].state).fen(), search_path[-1].team)))
		expand(search_path[-1], policy)

		#The value above is the value of the current state for the chosen team. We need to backprop that value back up the tree
		value_mapper = {search_path[-1].team: value.item(), not search_path[-1].team: -value.item()}
		
		#Backup
		for i,n in enumerate(reversed(search_path)):
			ind = len(search_path) - 1 - i
			n.W += value_mapper[n.team]
			n.N += 1
			n.Q = n.W/n.N
			if ind > 0:
				n.UCB = UCB(search_path[ind-1], n)


def run_game(model):
	# Create nodes
	root = MTCSNode(True, "8/8/2bb4/6kp/2p5/2P4K/5p2/8 b - - 0 71", None, 0, 0, None, None)
	#root = MTCSNode(True, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", None, 0, 0, None, None)
	root.UCB = False
	node = root
	path_taken = []
	game_result = {True: 0, False: 0}

	fig, ax = bd.get_fig_ax()
	bd.render_board(chess.Board(root.state), fig, ax)

	while True:

		move_mapper = {}
		
		#Running Simulations
		run_simulation(node, 30, model)
		#print_tree(node)
		break

		#Capturing the Probabilities from MTCS to put in training data
		for i in node.children:
			move_mapper[str(i.action)] = i.N/ node.N
		path_taken.append([node.state, str(move_mapper), node.team])

		#Picking the best Node
		best_N = -float('inf')
		best_node = None
		for i in node.children:
			if i.N > best_N:
				best_N = i.N
				best_node = i

		#Eliminating all other branches
		for z,i in enumerate(node.children.copy()):
			if i.state != best_node.state:
				node.children.remove(i)
		node = best_node
		print("Chosen Action: " + str(node.action))

		bd.render_board(chess.Board(node.state), fig, ax)

		if chess.Board(node.state).is_game_over():
			if chess.Board(node.state).result() == "1-0":
				game_result[True] = 1
				game_result[False] = -1
			if chess.Board(node.state).result() == "0-1":
				game_result[True] = -1
				game_result[False] = 1
			for i in path_taken:
				i.append(game_result[i[2]])
			break

	return path_taken