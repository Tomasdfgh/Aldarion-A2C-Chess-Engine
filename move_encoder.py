#!/usr/bin/env python3
"""
AlphaZero 4,672 Move Encoding

Creates the complete chess move vocabulary used by AlphaZero.
All 64 squares × 73 possible moves = 4,672 total moves.
"""

import chess
import numpy as np
import torch


class AlphaZeroMoveEncoder:
    """
    AlphaZero standard 4,672-move encoding
    64 origin squares × 73 move types = 4,672 total moves
    """
    
    def __init__(self):
        self.move_to_index = {}
        self.index_to_move = {}
        self._build_complete_vocabulary()
    
    def _build_complete_vocabulary(self):
        """Build the complete 4,672 move vocabulary"""
        index = 0
        
        # For each of the 64 squares as origin
        for from_square in range(64):
            from_file = from_square % 8
            from_rank = from_square // 8
            
            # 1. Queen-style moves (56 directions)
            # 8 directions × 7 distances = 56 moves
            directions = [
                (0, 1), (1, 1), (1, 0), (1, -1),    # N, NE, E, SE
                (0, -1), (-1, -1), (-1, 0), (-1, 1)  # S, SW, W, NW
            ]
            
            for direction in directions:
                df, dr = direction
                for distance in range(1, 8):  # 1 to 7 squares
                    to_file = from_file + df * distance
                    to_rank = from_rank + dr * distance
                    
                    # Check if target square is on board
                    if 0 <= to_file <= 7 and 0 <= to_rank <= 7:
                        to_square = to_rank * 8 + to_file
                        move = chess.Move(from_square, to_square)
                        
                        self.move_to_index[move] = index
                        self.index_to_move[index] = move
                        index += 1
            
            # 2. Knight moves (8 directions)
            knight_directions = [
                (2, 1), (1, 2), (-1, 2), (-2, 1),
                (-2, -1), (-1, -2), (1, -2), (2, -1)
            ]
            
            for df, dr in knight_directions:
                to_file = from_file + df
                to_rank = from_rank + dr
                
                if 0 <= to_file <= 7 and 0 <= to_rank <= 7:
                    to_square = to_rank * 8 + to_file
                    move = chess.Move(from_square, to_square)
                    
                    self.move_to_index[move] = index
                    self.index_to_move[index] = move
                    index += 1
            
            # 3. Pawn promotions (9 moves for promoting ranks)
            # Only for pawns on 7th rank (white) or 2nd rank (black)
            if from_rank == 6:  # White pawn on 7th rank (rank 6 in 0-indexed)
                # Forward promotion (3 pieces: knight, bishop, rook)
                to_square = (from_rank + 1) * 8 + from_file  # Move to 8th rank
                for promotion_piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                    move = chess.Move(from_square, to_square, promotion=promotion_piece)
                    self.move_to_index[move] = index
                    self.index_to_move[index] = move
                    index += 1
                
                # Diagonal captures with promotion (2 directions × 3 pieces = 6 moves)
                for df in [-1, 1]:
                    to_file = from_file + df
                    if 0 <= to_file <= 7:
                        to_square = (from_rank + 1) * 8 + to_file
                        for promotion_piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                            move = chess.Move(from_square, to_square, promotion=promotion_piece)
                            self.move_to_index[move] = index
                            self.index_to_move[index] = move
                            index += 1
            
            elif from_rank == 1:  # Black pawn on 2nd rank (rank 1 in 0-indexed)
                # Forward promotion (3 pieces)
                to_square = (from_rank - 1) * 8 + from_file  # Move to 1st rank
                for promotion_piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                    move = chess.Move(from_square, to_square, promotion=promotion_piece)
                    self.move_to_index[move] = index
                    self.index_to_move[index] = move
                    index += 1
                
                # Diagonal captures with promotion (2 directions × 3 pieces = 6 moves)
                for df in [-1, 1]:
                    to_file = from_file + df
                    if 0 <= to_file <= 7:
                        to_square = (from_rank - 1) * 8 + to_file
                        for promotion_piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                            move = chess.Move(from_square, to_square, promotion=promotion_piece)
                            self.move_to_index[move] = index
                            self.index_to_move[index] = move
                            index += 1
        
        print(f"Built move encoder with {len(self.move_to_index)} total moves")
        assert len(self.move_to_index) == 4672, f"Expected 4672 moves, got {len(self.move_to_index)}"
    
    def encode_move(self, move):
        """Convert chess.Move to index (0-4671)"""
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        return self.move_to_index.get(move, -1)
    
    def decode_move(self, index):
        """Convert index to chess.Move"""
        return self.index_to_move.get(index)
    
    def encode_policy(self, move_probabilities_dict):
        """
        Convert move probabilities dict to 4,672-dimensional vector
        
        Args:
            move_probabilities_dict: Dict mapping chess.Move -> probability
            
        Returns:
            torch.Tensor of shape (4672,) with probabilities
        """
        policy_vector = torch.zeros(4672)
        
        for move, prob in move_probabilities_dict.items():
            # Handle both chess.Move objects and UCI strings
            if isinstance(move, str):
                try:
                    move = chess.Move.from_uci(move)
                except:
                    continue
            
            index = self.encode_move(move)
            if index != -1:
                policy_vector[index] = float(prob)
        
        # Normalize to ensure it's a valid probability distribution
        total = policy_vector.sum()
        if total > 0:
            policy_vector = policy_vector / total
        
        return policy_vector
    
    def create_legal_mask(self, board):
        """
        Create mask for legal moves (1.0 for legal, 0.0 for illegal)
        
        Args:
            board: chess.Board object
            
        Returns:
            torch.Tensor of shape (4672,) with 1.0 for legal moves
        """
        mask = torch.zeros(4672)
        
        for move in board.legal_moves:
            index = self.encode_move(move)
            if index != -1:
                mask[index] = 1.0
        
        return mask


# Global encoder instance for easy access
GLOBAL_MOVE_ENCODER = AlphaZeroMoveEncoder()


def encode_move(move):
    """Convenience function to encode a single move"""
    return GLOBAL_MOVE_ENCODER.encode_move(move)


def decode_move(index):
    """Convenience function to decode a single move"""
    return GLOBAL_MOVE_ENCODER.decode_move(index)


def encode_policy(move_probabilities_dict):
    """Convenience function to encode policy"""
    return GLOBAL_MOVE_ENCODER.encode_policy(move_probabilities_dict)


def create_legal_mask(board):
    """Convenience function to create legal move mask"""
    return GLOBAL_MOVE_ENCODER.create_legal_mask(board)


if __name__ == "__main__":
    # Test the encoder
    encoder = AlphaZeroMoveEncoder()
    
    # Test some moves
    test_moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"), 
        chess.Move.from_uci("g1f3"),
        chess.Move.from_uci("a7a8q"),  # Promotion
    ]
    
    print("Testing move encoding:")
    for move in test_moves:
        index = encoder.encode_move(move)
        decoded = encoder.decode_move(index)
        print(f"{move} -> {index} -> {decoded}")
    
    # Test policy encoding
    print("\nTesting policy encoding:")
    move_probs = {
        chess.Move.from_uci("e2e4"): 0.6,
        chess.Move.from_uci("d2d4"): 0.3,
        chess.Move.from_uci("g1f3"): 0.1
    }
    
    policy_vector = encoder.encode_policy(move_probs)
    print(f"Policy vector shape: {policy_vector.shape}")
    print(f"Non-zero entries: {torch.nonzero(policy_vector).numel()}")
    print(f"Sum: {policy_vector.sum()}")
    
    # Test legal mask
    print("\nTesting legal mask:")
    board = chess.Board()
    mask = encoder.create_legal_mask(board)
    print(f"Legal moves in starting position: {torch.nonzero(mask).numel()}")