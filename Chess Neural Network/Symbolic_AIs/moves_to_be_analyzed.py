import chess
import random
import minimax
import ab_pruning
import Symbolic_AI.symbolicAI

def mean(data: list) -> float:
    entries = len(data)
    total_value = 0

    for val in data:
        total_value += val

    return total_value / entries

def standard_deviation(data: list) -> float:
    data_mean = mean(data)
    entries = len(data)
    total_difference = 0

    for val in data:
        total_difference += (val - data_mean) ** 2

    deviation = (total_difference / (entries - 1)) ** 0.5

    return deviation

def generate_board() -> str: # returns new board (FEN)
    move_amount = random.randint(1, 100)
    board = chess.Board()

    for _ in range(move_amount):
        if board.is_game_over():
            board = chess.Board()
            continue

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            board = chess.Board()
            continue

        board.push(random.choice(legal_moves))

        if board.is_checkmate():
            board = chess.Board()

    return board.fen()

sample_size = 30

depths = [1, 2, 3, 4]
FENs = []

for i in range(sample_size):
    FENs.append(generate_board()) # Generate a list of length sample_size of FEN strings

for depth in depths: # Go through each depth in depths list
    # Different algorithms
    minimax_nodes_searched = []
    ab_pruning_nodes_searched = []
    move_sorting_nodes_searched = []
    for i, FEN in enumerate(FENs): # Go through each FEN in FENs list
        print(i+1)
        board = chess.Board(FEN)
        _, minimax_nodes = minimax.get_best_move(board, depth)
        minimax_nodes_searched.append(minimax_nodes)

        _, ab_nodes = ab_pruning.get_best_move(board, depth)
        ab_pruning_nodes_searched.append(ab_nodes)

        _, move_sort_nodes = Symbolic_AI.symbolicAI.get_best_move(board, depth)
        move_sorting_nodes_searched.append(move_sort_nodes)

    print(f"STD for minimax at depth: {depth} is {standard_deviation(minimax_nodes_searched)}")
    print(f"STD for alpha-beta pruning at depth: {depth} is {standard_deviation(ab_pruning_nodes_searched)}")
    print(f"STD for move sort at depth: {depth} is {standard_deviation(move_sorting_nodes_searched)}")