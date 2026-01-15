import chess
import os
import Symbolic_AI.symbolicAI as symAI
import Comparison_Program.omnisciusV1 as v1

# GET EQUAL POSITIONS

equal_file = os.path.join("equal_positions", "equal_positions.txt")

nn_wins = 0
symAI_wins = 0
draws = 0

with open(equal_file, "r") as file:
    for i, FEN in enumerate(file):
        board = chess.Board(FEN)
        if i % 2 == 0:
            nn_color = chess.WHITE
        else:
            nn_color = chess.BLACK
    
        while not board.is_game_over():
            if board.turn == nn_color:
                move = v1.get_best_move(board)
                print(f"NEURAL NETWORK PLAYED: {board.san(move)}")
                board.push(move)
                print(board)
            else:
                move, _ = symAI.get_best_move(board, 3)
                print(f"SYMBOLIC AI PLAYED: {board.san(move)}")
                board.push(move)
                print(board)

        print(f"Game {i+1}: {board.result()}")

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
    "in this life or the next")
    print(f"Final Score of the colosseum:\n- Neural Network Wins: {nn_wins},\n- Symbolic AI Wins: {symAI_wins},\n- Draws: {draws}")
