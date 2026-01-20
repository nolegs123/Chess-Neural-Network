import chess
import random
import minimax
import ab_pruning
import Symbolic_AI.symbolicAI

def mean(data: list) -> float:
    total = 0
    for v in data:
        total += v
    return total / len(data)

def standard_deviation(data: list) -> float:
    m = mean(data)
    total = 0
    for v in data:
        total += (v - m) ** 2
    return (total / (len(data) - 1)) ** 0.5

def generate_board() -> str:
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

def ratio_list(a: list, b: list) -> list:
    out = []
    for x, y in zip(a, b):
        out.append(x / y)
    return out

sample_size = 100
depths = [1, 2, 3, 4]

FENs = []
for _ in range(sample_size):
    FENs.append(generate_board())

for depth in depths:
    print("\n" + "=" * 80)
    print(f"Depth {depth}")
    print("=" * 80)

    mm_nodes = []
    ab_nodes = []
    ms_nodes = []

    for i, fen in enumerate(FENs):
        print(f"FEN {i + 1}/{len(FENs)}")
        board = chess.Board(fen)

        _, mm = minimax.get_best_move(board, depth)
        _, ab = ab_pruning.get_best_move(board, depth)
        _, ms = Symbolic_AI.symbolicAI.get_best_move(board, depth)

        mm_nodes.append(mm)
        ab_nodes.append(ab)
        ms_nodes.append(ms)

    print("\nRAW NODE COUNTS")
    print("Algorithm        Mean Nodes        Std Dev")
    print("-" * 55)
    print(f"Minimax          {mean(mm_nodes):<18.2f}{standard_deviation(mm_nodes):.2f}")
    print(f"AlphaBeta        {mean(ab_nodes):<18.2f}{standard_deviation(ab_nodes):.2f}")
    print(f"MoveSorting      {mean(ms_nodes):<18.2f}{standard_deviation(ms_nodes):.2f}")

    ratios = {
        "MM / MM": ratio_list(mm_nodes, mm_nodes),
        "MM / AB": ratio_list(mm_nodes, ab_nodes),
        "MM / MS": ratio_list(mm_nodes, ms_nodes),

        "AB / MM": ratio_list(ab_nodes, mm_nodes),
        "AB / AB": ratio_list(ab_nodes, ab_nodes),
        "AB / MS": ratio_list(ab_nodes, ms_nodes),

        "MS / MM": ratio_list(ms_nodes, mm_nodes),
        "MS / AB": ratio_list(ms_nodes, ab_nodes),
        "MS / MS": ratio_list(ms_nodes, ms_nodes),
    }

    print("\nRATIOS (row / column)")
    print("Ratio            Mean Ratio        Std Dev")
    print("-" * 55)
    for name, values in ratios.items():
        print(f"{name:<15}{mean(values):<18.4f}{standard_deviation(values):.4f}")
