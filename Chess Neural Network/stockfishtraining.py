import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("/Users/dag/Desktop/DTU/02461_introduction_to_intelligent_systems/projectwork/stockfish/stockfish-macos-m1-apple-silicon")

board = chess.Board("4q1Rn/3pQ3/5N2/5p1r/1bp1n2k/8/3P3P/K6R w - - 0 1")
while not board.is_game_over():
    result = engine.play(board, chess.engine.Limit(time=0.1))
    board.push(result.move)

engine.quit()