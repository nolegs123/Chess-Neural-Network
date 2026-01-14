import chess
from Symbolic_AI.evaluation import evaluate_board


nodes_searched = 0

def minimax(board, depth):
    global nodes_searched
    nodes_searched += 1

    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if board.turn == chess.WHITE:
        best_value = float("-inf")
        for move in board.legal_moves:
            board.push(move)
            value = minimax(board, depth - 1)
            board.pop()
            if value > best_value:
                best_value = value
        return best_value
    else:
        best_value = float("inf")
        for move in board.legal_moves:
            board.push(move)
            value = minimax(board, depth - 1)
            board.pop()
            if value < best_value:
                best_value = value
        return best_value


def get_best_move(board, depth):
    global nodes_searched
    nodes_searched = 0

    best_move = None

    if board.turn == chess.WHITE:
        best_value = float("-inf")
        for move in board.legal_moves:
            board.push(move)
            value = minimax(board, depth - 1)
            board.pop()
            if value > best_value:
                best_value = value
                best_move = move
    else:
        best_value = float("inf")
        for move in board.legal_moves:
            board.push(move)
            value = minimax(board, depth - 1)
            board.pop()
            if value < best_value:
                best_value = value
                best_move = move

    return best_move, nodes_searched