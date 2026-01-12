import chess.engine
import chess.pgn

pgn_file = r"C:\Users\mikke\Downloads\lichess_elite_2021-08\lichess_elite_2021-08.pgn"
engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\mikke\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
desired_fens = 100

pgn = open(pgn_file)
found_fens = 0

while True:
    if found_fens >= desired_fens:
        break

    game = chess.pgn.read_game(pgn)
    board = chess.Board()
    for i, move in enumerate(game.mainline_moves()):
        board.push(move)
        if found_fens >= desired_fens:
            break
        if i == 20:
            info = engine.analyse(board, chess.engine.Limit(depth=12))
            score = info["score"].pov(board.turn)
            score = score.score(mate_score=10000)
            if -26 <= score <= 26:
                with open("equal_positions/equal_positions.txt", "a") as f:
                    f.write(f"{board.fen()} \n")
                    found_fens += 1