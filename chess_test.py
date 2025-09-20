import chess
import random

def test_random_games(n=500):
    results = []
    wins = 0
    draws = 0
    total_moves = 0
    for i in range(n):
        board = chess.Board()
        moves = 0
        while not board.is_game_over() and moves < 400:
            legal_moves = list(board.legal_moves)
            
            # Check for checkmate moves first
            checkmate_moves = []
            for move in legal_moves:
                board.push(move)
                if board.is_checkmate():
                    checkmate_moves.append(move)
                board.pop()
            
            # If there's a checkmate move, use it; otherwise random
            if checkmate_moves:
                move = random.choice(checkmate_moves)
            else:
                move = random.choice(legal_moves)
            
            board.push(move)
            moves += 1
        
        if board.is_checkmate():
            results.append("checkmate")
            wins += 1
        elif board.is_insufficient_material():
            results.append("insufficient_material")
            draws += 1
        else:
            results.append("other_draw")
            draws += 1

        total_moves += moves
    
    return results, wins / (wins + draws), total_moves / n

def random_board():
    # Create empty board
    board = chess.Board.empty()
    
    # Define all pieces
    pieces = [
        chess.Piece(chess.PAWN, chess.WHITE),
        chess.Piece(chess.ROOK, chess.WHITE),
        chess.Piece(chess.KNIGHT, chess.WHITE),
        chess.Piece(chess.BISHOP, chess.WHITE),
        chess.Piece(chess.QUEEN, chess.WHITE),
        chess.Piece(chess.KING, chess.WHITE),
        chess.Piece(chess.PAWN, chess.BLACK),
        chess.Piece(chess.ROOK, chess.BLACK),
        chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.Piece(chess.BISHOP, chess.BLACK),
        chess.Piece(chess.QUEEN, chess.BLACK),
        chess.Piece(chess.KING, chess.BLACK),
    ]
    
    # Get all squares (0-63)
    all_squares = list(range(64))
    
    # Randomly place pieces
    for piece in pieces:
        if all_squares:  # If there are still empty squares
            square = random.choice(all_squares)
            board.set_piece_at(square, piece)
            all_squares.remove(square)
    
    return board

def chess_960():
    print("Chess960 random position:")
    board1 = chess.Board.from_chess960_pos(random.randint(0, 959))
    return board1

if __name__ == "__main__":

    results, ratio, avg_moves = test_random_games()
    print(ratio, avg_moves)

    #chess_960()
