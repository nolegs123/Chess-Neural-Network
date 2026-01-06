import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("/Users/dag/Desktop/DTU/02461_introduction_to_intelligent_systems/projectwork/stockfish/stockfish-macos-m1-apple-silicon")



for shard_id in range(1000):
    with open(f"FENs/FEN_moves_{shard_id}.txt", "r") as file:
        FENs = file.readlines()

        with open(f"FENs_optimal_moves/FEN_optimal_move_{shard_id}.txt", "a") as f:

            for FEN in FENs:
                board = chess.Board(FEN)

                # analyze move
                info = engine.analyse(board, chess.engine.Limit(depth=12))
                move = info["pv"][0]

                f.write(f"{FEN} | {move}\n")




    
    

engine.quit()