import os
import chess.pgn

all_ratings_directory = r"C:\Users\mikke\Downloads\lichess_elite_2021-08\lichess_elite_2021-08.pgn"

def read_games_from_pgn(directory: str, shard_size: int = 10000):
    pgn = open(directory) # Opens entire PGN
    shard_id = 0 # Shard ID
    fen_number = 0 # Amount of FENs saved

    while True:
        game = chess.pgn.read_game(pgn) # Sets game to current game
        if game is None:
            break # Breaks loop if no game

        board = game.board()

        for move in game.mainline_moves(): # Goes through all moves in each game
            with open(f"FENs/FEN_moves_{shard_id}.txt", "a") as file: # Creates FEN_moves{shard_id}.txt file if it doesn’t exist.
                board.push(move) # Plays move
                fen = board.fen() # Translates board to FEN string

                fen_number += 1

                if fen_number % shard_size == 0: # Move to the next shard once the current shard reaches shard_size entries
                    shard_id += 1

                file.write(f"{fen}\n") # Append each FEN move from each game to file

read_games_from_pgn(all_ratings_directory) 

