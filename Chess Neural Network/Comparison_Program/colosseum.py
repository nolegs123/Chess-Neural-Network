import chess
import chess.pgn
import chess.svg
import random
import os
import Symbolic_AI.symbolicAI as symAI
import Comparison_Program.omnisciusV1 as v1

# GET EQUAL POSITIONS

equal_file = os.path.join("equal_positions", "equal_positions.txt")
pgn_dir = "pgn_games"
os.makedirs(pgn_dir, exist_ok=True)

nn_wins = 0
symAI_wins = 0
draws = 0

with open(equal_file, "r") as file:
    for i, FEN in enumerate(file):
        FEN = FEN.strip()
        board = chess.Board(FEN)

        game = chess.pgn.Game()
        game.setup(board)
        game.headers["SetUp"] = "1"
        game.headers["FEN"] = FEN
        game.headers["Event"] = "NeuralNet vs SymbolicAI"
        game.headers["Round"] = str(i + 1)

        if i % 2 == 0:
            nn_color = chess.WHITE
            game.headers["White"] = "NeuralNetwork"
            game.headers["Black"] = "SymbolicAI"
        else:
            nn_color = chess.BLACK
            game.headers["White"] = "SymbolicAI"
            game.headers["Black"] = "NeuralNetwork"

        node = game

        while not board.is_game_over():
            if board.turn == nn_color:
                move = v1.get_best_move(board)

                # HARD GUARANTEE
                if move not in board.legal_moves:
                    move = random.choice(list(board.legal_moves))
                    print("HAHA IM STUPID AND PLAYED AN ILLEGAL MOVE")

                san = board.san(move)
                print(f"NEURAL NETWORK PLAYED: {san}")

                board.push(move)
                node = node.add_variation(move)
                print(board)

            else:
                move, _ = symAI.get_best_move(board, 3)

                if move not in board.legal_moves:
                    move = random.choice(list(board.legal_moves))
                    print("HAHA IM STUPID AND PLAYED AN ILLEGAL MOVE")

                san = board.san(move)
                print(f"SYMBOLIC AI PLAYED: {san}")

                board.push(move)
                node = node.add_variation(move)
                print(board)

        game.headers["Result"] = board.result()
        print(f"Game {i+1}: {board.result()}")

        pgn_path = os.path.join(pgn_dir, f"game_{i+1:05d}.pgn")
        with open(pgn_path, "w", encoding="utf-8") as pgn_file:
            print(game, file=pgn_file)

    # print final results after all games are played
    if board.result() == "1-0":
        if nn_color == chess.WHITE:
            nn_wins += 1
        else:
            symAI_wins += 1
    elif board.result() == "0-1":
        if nn_color == chess.BLACK:
            symAI_wins += 1
        else:
            nn_wins += 1
    else:
        draws += 1
        
    print("My name is Maximus Decimus Meridius, commander of the Armies of the North, General of the Felix Legions, " \
    "loyal servant to the true emperor, Marcus Aurelius. Father to a murdered son, husband to a murdered wife. And I will have my vengeance, "
    "in this life or the next!!")
    print(f"Final Score of the colosseum:\n- Neural Network Wins: {nn_wins},\n- Symbolic AI Wins: {symAI_wins},\n- Draws: {draws}")
