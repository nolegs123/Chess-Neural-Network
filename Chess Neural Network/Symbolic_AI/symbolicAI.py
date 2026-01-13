import chess
from evaluation import evaluate_board
from move_sort import order_moves

nodes_searched = 0

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
