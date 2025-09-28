#!/usr/bin/env python3
import pickle
import sys
import chess

def display_board(fen, game_outcome=None):
    """Display a chess board from FEN string"""
    try:
        board = chess.Board(fen)
        print(board)
        print(f"Turn: {'White' if board.turn else 'Black'}")
        
        # Check if game is over
        if board.is_game_over():
            result = board.result()
            print(f"Game over! Result: {result}")
            if board.is_checkmate():
                print("Checkmate!")
            elif board.is_stalemate():
                print("Stalemate!")
            elif board.is_insufficient_material():
                print("Insufficient material!")
            elif board.is_seventyfive_moves():
                print("75-move rule!")
            elif board.is_fivefold_repetition():
                print("Fivefold repetition!")
        else:
            print("Game in progress")
            if board.is_check():
                print("In check!")
        
        if game_outcome is not None:
            print(f"Recorded outcome: {game_outcome}")
        
        # Print legal moves
        legal_moves = list(board.legal_moves)
        print(f"Legal moves ({len(legal_moves)}): {[str(move) for move in legal_moves]}")
        
        return board
    except:
        print(f"Invalid FEN: {fen}")
        return None

#Test the function
display_board("3R4/2R5/5p1k/3K1P2/5P2/1n5r/8/8 w - - 5 123")

# if len(sys.argv) != 2:
#     print("Usage: python3 inspect_data.py <data_file.pkl>")
#     sys.exit(1)

# data_file = sys.argv[1]

# with open(data_file, 'rb') as f:
#     data = pickle.load(f)

# print(f"Total examples: {len(data)}")
# print("\nFirst 5 examples:")
# for i, (board_fen, history_fens, move_probs, game_outcome) in enumerate(data):
#     print(f"\n[{i}] FEN: {board_fen}")
#     print(f"    History length: {len(history_fens)}")
#     print(f"    Moves: {len(move_probs)}")
#     print(f"    Outcome: {game_outcome}")