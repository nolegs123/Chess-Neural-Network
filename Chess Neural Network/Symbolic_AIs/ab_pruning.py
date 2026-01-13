import chess
from Symbolic_AI.evaluation import evaluate_board

nodes_searched = 0

def minimax(board, depth, alpha, beta):
    global nodes_searched
    nodes_searched += 1

    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if board.turn == chess.WHITE:
        best_eval = float('-inf')
        for move in board.legal_moves:
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
        for move in board.legal_moves:
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

    for move in board.legal_moves:
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