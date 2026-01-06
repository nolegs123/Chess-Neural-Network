import chess
import chess.engine
import random

engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\mikke\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
for shard_id in range(1000):
    with open(f"FENs/FEN_moves_{shard_id}.txt", "r") as file:
        FENs = file.readlines()

        with open(f"FENs_optimal_moves/FEN_optimal_move_{shard_id}.txt", "a") as f:

            for FEN in FENs:
                FEN = FEN.strip()
                board = chess.Board(FEN)
                print(board)

                # analyze move
                info = engine.analyse(board, chess.engine.Limit(depth=12))
                move = info["pv"][0]

                f.write(f"{FEN} | {move} \n")

def generate_board(move_amount: int) -> str:
    board = chess.Board()
    for _ in range(move_amount):
        legal_moves = list(board.legal_moves)
        chosen_move = random.choice(legal_moves)
        board.push(chosen_move)

    return board.fen()

engine.quit()