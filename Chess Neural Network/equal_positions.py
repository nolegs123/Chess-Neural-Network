import chess.engine
import chess.pgn

pgn_file = r"C:\Users\mikke\Downloads\lichess_elite_2021-08\lichess_elite_2021-08.pgn"
engine = chess.engine.SimpleEngine.popen_uci(
    r"C:\Users\mikke\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
)

desired_fens = 100

pgn = open(pgn_file)
found_fens = 0
seen_positions = set()

while found_fens < desired_fens:
    game = chess.pgn.read_game(pgn)
    if game is None:
        break

    board = chess.Board()

    for i, move in enumerate(game.mainline_moves()):
        board.push(move)

        if i == 20:
            info = engine.analyse(board, chess.engine.Limit(depth=12))
            score = info["score"].pov(board.turn).score(mate_score=10000)

            if score is None:
                break

            if -26 <= score <= 26:
                fen = board.fen()

                if fen not in seen_positions:
                    with open("equal_positions/equal_positions.txt", "a") as f:
                        f.write(f"{fen}\n")

                    seen_positions.add(fen)
                    found_fens += 1

            break