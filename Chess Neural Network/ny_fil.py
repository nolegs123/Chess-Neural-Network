import os
import chess.pgn

# Relative paths to datasets
current_directory_name = os.path.dirname(__file__)
all_ratings_directory = os.path.join(current_directory_name, 'lichess_all_ratings.pgn')
elite_directory = os.path.join(current_directory_name, 'lichess_elite.pgn')

def read_games_from_pgn(directory: str, shard_size: int = 10000, print_progress: bool = False, games_per_print: int = 0):
    pgn = open(directory) # Opens entire PGN
    shard_id = 0 # Shard ID
    fen_number = 0 # Amount of FENs saved

    
    while True:
        game = chess.pgn.read_game(pgn) # Sets game to current game
        if game is None:
            break # Breaks loop if no game

        board = game.board()

        with open("FEN_moves.txt", "a") as file: # Creates FEN_moves.txt file if it doesn’t exist.

            for move in game.mainline_moves(): # Goes through all moves in each game
                board.push(move)

                print(board.fen())

                file.write(f"{board.fen()}\n") # Append each FEN move from each game to file

read_games_from_pgn(all_ratings_directory) 