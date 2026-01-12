import chess

board = chess.Board()

def evaluate_board(board) -> int:
    pieces = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

    if board.is_checkmate():
        return float('-inf') if board.turn == chess.WHITE else float('inf')

    evaluation = 0

    for piece_type, value in pieces.items():
        evaluation += len(board.pieces(piece_type, chess.WHITE)) * value
        evaluation -= len(board.pieces(piece_type, chess.BLACK)) * value

    return evaluation

def minimax(board, depth, alpha, beta):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    legal_moves = list(board.legal_moves)

    if board.turn == chess.WHITE:
        best_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta)
            board.pop()

            if eval > best_eval:
                best_eval = eval

            alpha = max(alpha, best_eval)   

            if beta <= alpha:
                break 

        return best_eval
    
    elif board.turn == chess.BLACK:
        best_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth -1, alpha, beta)
            board.pop()

            if eval < best_eval:
                best_eval = eval

            beta = min(beta, best_eval)

            if beta <= alpha:
                break 

        return best_eval

def get_best_move(board):
    best_move = None
    best_eval = float('-inf') if board.turn == chess.WHITE else float('inf')

    alpha = float("-inf")
    beta = float("inf")

    legal_moves = list(board.legal_moves)

    for move in legal_moves:
        board.push(move)
        eval = minimax(board, 3, alpha, beta)
        board.pop()

        if (board.turn == chess.WHITE and eval > best_eval) or (board.turn == chess.BLACK and eval < best_eval):
            best_move = move
            best_eval = eval

            if board.turn == chess.WHITE:
                alpha = max(alpha, best_eval)
            else:
                beta = min(beta, best_eval)
            

    return best_move