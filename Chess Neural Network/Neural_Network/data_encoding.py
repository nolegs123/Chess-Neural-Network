import torch

# Specific piece layers for tensor/matrix
PIECE_TO_CHANNEL = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11, 
}

def fen_to_tensor(fen: str):
    parts = fen.split() # Make list of FEN string
    board, turn, castling, ep, halfmove = parts[:5] # Fetch board/game details

    tensor = torch.zeros((19, 8, 8), dtype=torch.float32) # Initialize 19x8x8 matrix with zero elements

    # --- board ---
    rows = board.split("/")
    for rank, row in enumerate(rows):
        file = 0
        for char in row:
            if char.isdigit():
                file += int(char)
            else:
                tensor[PIECE_TO_CHANNEL[char], rank, file] = 1
                file += 1

    # --- side to move ---
    if turn == "w":
        tensor[12, :, :] = 1 # Layer 12 in matrix represents which turn it is: 0 for b, 1 for w

    # --- castling rights ---
    if "K" in castling: tensor[13, :, :] = 1
    if "Q" in castling: tensor[14, :, :] = 1
    if "k" in castling: tensor[15, :, :] = 1
    if "q" in castling: tensor[16, :, :] = 1

    # --- en passant ---
    if ep != "-":
        file = ord(ep[0]) - ord("a")
        rank = 8 - int(ep[1])
        tensor[17, rank, file] = 1

    # --- halfmove clock ---
    halfmove = int(halfmove)
    halfmove_norm = min(halfmove / 100.0, 1.0)
    tensor[18, :, :] = halfmove_norm

    return tensor


def generate_all_uci_moves():
    files = "abcdefgh"
    ranks = "12345678"
    promotions = ["q", "r", "b", "n"]

    moves = []

    for from_file in files:
        for from_rank in ranks:
            for to_file in files:
                for to_rank in ranks:
                    moves.append(f"{from_file}{from_rank}{to_file}{to_rank}")
                    if from_rank == "7" and to_rank == "8":
                        for promotion in promotions:
                            moves.append(f"{from_file}{from_rank}{to_file}{to_rank}{promotion}")
                    if from_rank == "2" and to_rank == "1":
                        for promotion in promotions:
                            moves.append(f"{from_file}{from_rank}{to_file}{to_rank}{promotion}")
    return sorted(moves)

print(generate_all_uci_moves())