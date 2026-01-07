import chess
import chess.engine
import random

def generate_board(move_amount: int) -> str:
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

engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\mikke\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
for shard_id in range(1000):
    with open(f"FENs/FEN_moves_{shard_id}.txt", "r") as file:
        FENs = file.readlines()

        with open(f"FENs_optimal_moves/FEN_optimal_move_{shard_id}.txt", "a") as f:

            for FEN in FENs:
                FEN = FEN.strip()
                board = chess.Board(FEN)

                if board.is_game_over():
                    board = chess.Board(generate_board(random.randint(0, 100)))
                
                # analyze move
                info = engine.analyse(board, chess.engine.Limit(depth=12))
                move = info["pv"][0] # Choose highest ranked move
                

                f.write(f"{FEN} | {move}\n") # append the (FEN | optimal_move)

engine.quit()