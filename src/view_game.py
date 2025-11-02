import chess
import random
import os
import torch

# Import Aldarion modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import model as md
from src.agent import mcts as mt
from src.config import Config

def clear_terminal():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_board(board):
    """Display the chess board in ASCII format"""
    clear_terminal()
    print("="*40)
    print(f"Move #{board.fullmove_number} - {'White' if board.turn else 'Black'} to move")
    print("="*40)
    print(board)
    
    if board.is_check():
        print("CHECK!")
    if board.is_checkmate():
        print("CHECKMATE!")
    if board.is_stalemate():
        print("STALEMATE!")
    if board.is_insufficient_material():
        print("INSUFFICIENT MATERIAL!")
    if board.is_seventyfive_moves():
        print("75-MOVE RULE!")
    if board.is_fivefold_repetition():
        print("FIVEFOLD REPETITION!")

def play_exhibition_game():
    """Play an exhibition game with visual board display"""
    config = Config()
    
    # Check if best model exists
    best_model_path = config.resource.model_best_weight_path
    if not os.path.exists(best_model_path):
        print(f"Error: Best model not found at {best_model_path}")
        print("Please train a model first or run: python run.py init")
        return None
    
    print("="*60)
    print("EXHIBITION GAME: Best Model vs Best Model")
    print("="*60)
    print(f"Model: {best_model_path}")
    print(f"Starting position: Normal chess")
    print(f"Simulations per move: 100 (reduced for viewing)")
    print("="*60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = md.ChessNet()
    model.to(device)
    
    state = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Start game with normal chess position
    board = chess.Board()
    game_history = []
    move_count = 0
    max_game_length = 1000
    num_simulations = 400
    
    white_tree = None
    black_tree = None
    
    display_board(board)
    print("Starting exhibition game...")
    
    while not board.is_game_over() and move_count < max_game_length:
        current_player = "White" if board.turn else "Black"
        current_tree = white_tree if board.turn else black_tree
        
        try:
            print(f"\n{current_player} is thinking...")
            
            # Get best move using MCTS (same logic as evaluation)
            best_move, selected_child = mt.return_move_and_child(
                model=model,
                board_fen=board.fen(),
                num_simulations=num_simulations,
                device=device,
                game_history=game_history,
                existing_tree=current_tree,
                temperature=0.1  # Small temperature for variety
            )
            
            if best_move is None:
                print(f"No legal moves available for {current_player}")
                break
            
            # Update trees
            if board.turn:
                white_tree = selected_child
            else:
                black_tree = selected_child
            
            # Make the move
            move_obj = chess.Move.from_uci(best_move)
            board.push(move_obj)
            game_history.append(board.copy())
            move_count += 1
            
            # Display new position
            display_board(board)
            print(f"Last move: {best_move} ({current_player})")
            print(f"Simulations: {num_simulations}")
            
        except Exception as e:
            print(f"Error during {current_player} move: {e}")
            break
    
    # Display final result
    print("\n" + "="*60)
    print("EXHIBITION GAME OVER!")
    print("="*60)
    
    if board.is_checkmate():
        winner = "Black" if board.turn else "White"
        print(f"{winner} wins by checkmate!")
    elif board.is_stalemate():
        print("Draw by stalemate!")
    elif board.is_insufficient_material():
        print("Draw by insufficient material!")
    elif board.is_seventyfive_moves():
        print("Draw by 75-move rule!")
    elif board.is_fivefold_repetition():
        print("Draw by fivefold repetition!")
    elif move_count >= max_game_length:
        print(f"Draw by {max_game_length}-move limit!")
    else:
        result = board.result()
        if result == "1-0":
            print("White wins!")
        elif result == "0-1":
            print("Black wins!")
        elif result == "1/2-1/2":
            print("Draw!")
        else:
            print("Game ended unexpectedly")
    
    print(f"Final FEN: {board.fen()}")
    print(f"Total moves: {move_count}")
    
    return board

def play_random_game():
    """Play a complete random chess game and display each move (original function)"""
    board = chess.Board()
    move_count = 0
    
    print("Starting a random chess game...")
    display_board(board)
    
    while not board.is_game_over():
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
            
        # Pick a random legal move
        random_move = random.choice(legal_moves)
        board.push(random_move)
        move_count += 1
        
        display_board(board)
        print(f"Last move: {random_move}")
        
        # Optional: stop after a reasonable number of moves to avoid infinite games
        if move_count > 1000:
            print("\nGame stopped after 1000 moves to prevent infinite play.")
            break
    
    # Display final game result
    print("\n" + "="*50)
    print("GAME OVER!")
    print("="*50)
    
    result = board.result()
    if result == "1-0":
        print("White wins!")
    elif result == "0-1":
        print("Black wins!")
    elif result == "1/2-1/2":
        print("Draw!")
    else:
        print("Game terminated early")
    
    print(f"Final position FEN: {board.fen()}")
    print(f"Total moves played: {move_count}")
    
    return board

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--random":
        play_random_game()
    else:
        play_exhibition_game()