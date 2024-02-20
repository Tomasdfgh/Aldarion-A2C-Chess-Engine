import chess
import board_reader as br
import board_display as bd

##################################################################################################################################
#													Monte Carlo Tree search 													 #
#																																 #
#	This script runs Monte Carlo Tree search. It takes in a chess board (fen string), the neural network model, the temperature, #
#	and the number of sims to run per move. Main function to call is run_game(). The rest of the functions are accessories that  #
#	help set up the tree, expand the tree, find the UCB score, and run the simulations.											 #
#																																 #
##################################################################################################################################

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
	pro_policy = policy[0][4096:]
	policy = policy[0][:4096].reshape(64,-1)
	move_mapper = {}
	pro_mapper = br.pro_mapper()

	for i in chess.Board(node.state).legal_moves:
		coords = br.legal_move_to_coord(str(i))
		
		#A normal move (not a promotion move)
		if len(coords) == 2:
			move_mapper[str(i)] = policy[(coords[0][0] + ((coords[0][1] - 1) * 8)) - 1][(coords[1][0] + ((coords[1][1] - 1) * 8)) -1].item()
		
		#Pawn Promotion move. Cannot use the first 4096 elements of the tensor
		if len(coords) == 3:
			move_mapper[str(i)] = pro_policy[pro_mapper[(coords[0], coords[1], coords[2])]].item()

	move_mapper = normalize_list(move_mapper)

	for i in chess.Board(node.state).legal_moves:
		board_temp = chess.Board(node.state)
		board_temp.push_uci(str(i))
		child_node = MTCSNode(board_temp.turn, board_temp.fen(), i, 0, 0, 0, move_mapper[str(i)])
		child_node.UCB = UCB(node, child_node)
		node.add_child(child_node)

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
		policy, value = model(br.board_to_array(chess.Board(root.state).fen(), root.team))
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
		policy, value = model(br.board_to_array(chess.Board(search_path[-1].state).fen(), search_path[-1].team))
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


def run_game(model, temperature, num_sim):
	
	# Create nodes
	root = MTCSNode(True, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", None, 0, 0, None, None)
	root.UCB = False
	node = root
	path_taken = []

	fig, ax = bd.get_fig_ax()
	board = chess.Board(node.state)
	bd.render_board(board, fig, ax)

	while True:

		move_mapper_ = {}
		
		#Running Simulations
		run_simulation(node, num_sim, model)

		#Capturing the Probabilities from MTCS to put in training data
		for i in node.children:
			move_mapper_[str(i.action)] = i.N/ node.N
		path_taken.append([node.state, str(move_mapper_), node.team, node.W])

		#Picking the best Node
		prob_dist = []
		cors_move = []
		for i in node.children:
			prob_dist.append(i.N)
			cors_move.append(i)
		
		for i in range(len(prob_dist)):
			prob_dist[i] = prob_dist[i] ** (1/temperature)

		test_c = prob_dist.copy()
		for i in range(len(test_c)):
			prob_dist[i] /= sum(test_c)

		prob_dist = br.array_to_tensor(prob_dist)
		ind_choice = br.prob_sampler(prob_dist, 1)
		best_node = cors_move[ind_choice]

		#Eliminating all other branches
		for z,i in enumerate(node.children.copy()):
			if i.state != best_node.state:
				node.children.remove(i)
		node = best_node

		bd.render_board(board, fig, ax, move = str(node.action))

		if chess.Board(node.state).is_game_over():
			break

	return path_taken