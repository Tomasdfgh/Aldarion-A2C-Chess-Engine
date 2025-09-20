#!/usr/bin/env python3
"""
Chess board display and legal move analysis script
"""

import chess
import chess.svg

def display_board_and_legal_moves(fen):
    """
    Display a chess board from FEN and print all legal moves
    
    Args:
        fen (str): FEN string representing the board position
    """
    # Create board from FEN
    board = chess.Board(fen)
    
    print("Board Position:")
    print(board)
    print()
    
    print(f"FEN: {fen}")
    print(f"Turn: {'White' if board.turn else 'Black'}")
    print(f"Castling rights: {board.castling_rights}")
    print(f"En passant: {board.ep_square}")
    print(f"Halfmove clock: {board.halfmove_clock}")
    print(f"Fullmove number: {board.fullmove_number}")
    print()
    
    # Get all legal moves
    legal_moves = list(board.legal_moves)
    
    print(f"Legal moves ({len(legal_moves)} total):")
    for i, move in enumerate(legal_moves, 1):
        # Get piece that's moving
        piece = board.piece_at(move.from_square)
        
        # Format move with piece info
        move_str = f"{move.uci()}"
        
        print(f"{i:2d}. {move_str:<8} ({chess.square_name(move.from_square)} -> {chess.square_name(move.to_square)})")
        
        # Print every 10 moves for readability
        if i % 10 == 0:
            print()

def main():
    """Main function to test the board display"""
    fen = "B4r2/2k1p3/1pn1b3/1P2p2r/N1R3pP/3Pb3/8/2BK3R b - - 1 51"
    display_board_and_legal_moves(fen)

if __name__ == "__main__":
    main()