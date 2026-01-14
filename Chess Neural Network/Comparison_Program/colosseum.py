import chess

# GET EQUAL POSITIONS

model_path = "r\"C:\\Users\\mikke\\source\\repos\\Chess-Neural-Network\\Chess Neural Network\\Data\\testing_data\\\equal_positions\\equal_positions.txt\""

with open(model_path, "r") as file:
    for FEN in file:
        board = chess.Board(FEN)

        
    
