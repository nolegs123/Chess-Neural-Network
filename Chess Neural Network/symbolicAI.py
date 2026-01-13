import chess

board = chess.Board()

def evaluate_board(board) -> int:
    pieces = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

    if board.is_checkmate():
        return float('-inf') if board.turn == chess.WHITE else float('inf')

    #remis
    if board.is_stalemate() or board.is_insufficient_material() \
        or board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
        return 0

    evaluation = 0

    for piece_type, value in pieces.items():
        evaluation += len(board.pieces(piece_type, chess.WHITE)) * value
        evaluation -= len(board.pieces(piece_type, chess.BLACK)) * value

    return evaluation

def order_moves(board) -> list:
    piece_value = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 100
    }

    captures = []
    non_captures = []
    legal_moves = list(board.legal_moves)

    for move in legal_moves:
        if board.is_capture(move):
            captures.append(move)
        else:
            board.push(move)
            is_check = board.is_check()
            board.pop()

            if is_check:
                checks.apppend(move)
            else:
                quiet.append(move)

    captures.sort(
        key=lambda m: piece_value.get(
            board.piece_at(m.to_square).piece_type, 0)
        if board.piece_at(m.to_square) else 0,
        reverse=True
    )

    return captures + checks + quiet

def minimax(board, depth, alpha, beta):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    sorted_moves = order_moves(board)

    if board.turn == chess.WHITE:
        best_eval = float('-inf')
        for move in sorted_moves:
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
        for move in sorted_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta)
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

    sorted_moves = order_moves(board)

    for move in sorted_moves:
        board.push(move)
        eval = minimax(board, 5, alpha, beta)
        board.pop()

        if (board.turn == chess.WHITE and eval > best_eval) or (board.turn == chess.BLACK and eval < best_eval):
            best_move = move
            best_eval = eval

            if board.turn == chess.WHITE:
                alpha = max(alpha, best_eval)
            else:
                beta = min(beta, best_eval)
            

    return best_move

print(get_best_move(chess.Board("r1bqkb1r/pppp2pp/2n2n2/4pp2/2P5/2N1P1P1/PP1P1PBP/R1BQK1NR b KQkq - 0 5")))