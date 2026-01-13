import chess

nodes_searched = 0

def evaluate_board(board) -> int:
    pieces = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    if board.is_checkmate():
        return float('-inf') if board.turn == chess.WHITE else float('inf')

    if board.is_stalemate() or board.is_insufficient_material() \
        or board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
        return 0

    evaluation = 0
    for piece_type, value in pieces.items():
        evaluation += len(board.pieces(piece_type, chess.WHITE)) * value
        evaluation -= len(board.pieces(piece_type, chess.BLACK)) * value

    return evaluation


def order_moves(board):
    piece_value = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 100
    }

    captures = []
    checks = []
    quiet = []

    for move in board.legal_moves:
        if board.is_capture(move):
            captures.append(move)
        else:
            board.push(move)
            if board.is_check():
                checks.append(move)
            else:
                quiet.append(move)
            board.pop()

    captures.sort(
        key=lambda m: piece_value.get(
            board.piece_at(m.to_square).piece_type
            if board.piece_at(m.to_square) else 0,
            0
        ),
        reverse=True
    )

    return captures + checks + quiet


def minimax(board, depth, alpha, beta):
    global nodes_searched
    nodes_searched += 1

    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if board.turn == chess.WHITE:
        best_eval = float('-inf')
        for move in order_moves(board):
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta)
            board.pop()

            best_eval = max(best_eval, eval)
            alpha = max(alpha, best_eval)

            if beta <= alpha:
                break

        return best_eval

    else:
        best_eval = float('inf')
        for move in order_moves(board):
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta)
            board.pop()

            best_eval = min(best_eval, eval)
            beta = min(beta, best_eval)

            if beta <= alpha:
                break

        return best_eval


def get_best_move(board, depth=5):
    global nodes_searched
    nodes_searched = 0

    best_move = None
    best_eval = float('-inf') if board.turn == chess.WHITE else float('inf')

    alpha = float('-inf')
    beta = float('inf')

    for move in order_moves(board):
        board.push(move)
        eval = minimax(board, depth - 1, alpha, beta)
        board.pop()

        if board.turn == chess.WHITE:
            if eval > best_eval:
                best_eval = eval
                best_move = move
                alpha = max(alpha, best_eval)
        else:
            if eval < best_eval:
                best_eval = eval
                best_move = move
                beta = min(beta, best_eval)

    return best_move, nodes_searched


fen = "r1bqkb1r/pppp2pp/2n2n2/4pp2/2P5/2N1P1P1/PP1P1PBP/R1BQK1NR b KQkq - 0 5"
board = chess.Board(fen)

move, nodes = get_best_move(board, depth=5)
print(move)
print(nodes)
