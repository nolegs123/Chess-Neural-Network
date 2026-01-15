import torch
import chess
import os
import Neural_Network.data_encoding as de

device = "cuda" if torch.cuda.is_available() else "cpu"
path = os.path.join("Models", "omnisciusV1.pt")

all_moves = de.generate_all_uci_moves()
id_to_uci = {idx: uci for idx, uci in enumerate(all_moves)}


omniscius = torch.nn.Sequential(
    torch.nn.Conv2d(19, 64, kernel_size=3, padding=1),
    torch.nn.ReLU(),

    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
    torch.nn.ReLU(),

    torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
    torch.nn.ReLU(),

    torch.nn.Flatten(),

    torch.nn.Linear(16384, 1024),
    torch.nn.ReLU(),

    torch.nn.Linear(1024, 4608)
).to(device)

model = torch.load(path, map_location=device)
omniscius.load_state_dict(model["model"])

def get_best_move(board):
    FEN = board.fen() # STRING
    FEN_tensor = de.fen_to_tensor(FEN) # TENSOR

    with torch.no_grad():
        omniscius.eval()

        FEN_tensor = de.fen_to_tensor(FEN).unsqueeze(0).to(device)
        out = omniscius(FEN_tensor)

        move_id = torch.argmax(out, dim=1).item()
        uci_move = id_to_uci[move_id]

        return uci_move

