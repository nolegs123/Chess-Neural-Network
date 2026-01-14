import chess
import os
import Symbolic_AI.symbolicAI as symAI

# GET EQUAL POSITIONS

equal_file = os.path.join("equal_positions", "equal_positions.txt")

with open(equal_file, "r") as file:
    for i, FEN in enumerate(file):
        board = chess.Board(FEN)
        if i % 2 == 0:
            nn_color = chess.WHITE
        else:
            nn_color = chess.BLACK

        while not board.is_game_over():
            if board.turn == nn_color:
                move, _ = symAI.get_best_move(board, 3)
                print(board.san(move))
                board.push(move)
                print(board)
            else:
                move, _ = symAI.get_best_move(board, 3)
                print(board.san(move))
                board.push(move)
                print(board)

        print(board.result())