import torch
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
from data_encoding import fen_to_tensor
from data_encoding import generate_all_uci_moves

# SETTINGS
BATCH_SIZE = 500
EPOCHS = 5
SUB_EPOCHS = 13
SHARDS_PER_SUB_EPOCH = 75

# FILE PATHS
TESTING_PATH = "C:\Users\mikke\source\repos\Chess-Neural-Network\Chess Neural Network\Data\testing_data"
TRAINING_PATH = "C:\Users\mikke\source\repos\Chess-Neural-Network\Chess Neural Network\Data\training_data"

# INITIALIZATION
uci_to_id = {uci: idx for idx, uci in enumerate(all_moves)} # Translation map of UCI move to ID
id_to_uci = {idx: uci for idx, uci in enumerate(all_moves)} # Translation map of ID to UCI move

def load_shard(path: str, uci_to_id: dict):
    positions = []
    labels = []

    with open(path, "r") as f:
        for line in f:
            fen, best_move = line.strip().split(" | ")
            fen_tensor = fen_to_tensor(fen)
            move_label = uci_to_id[best_move]

            positions.append(fen_tensor)
            labels.append(move_label)

    x = torch.stack(positions)
    y = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(x, y)

device = "cuda" if torch.cuda.is_available() else "cpu"

omniscius = torch.nn.Sequential(
    torch.nn.Conv2d(19, 64, kernel_size=3, padding=1), # INPUT LAYER
    torch.nn.ReLU(),

    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), # HIDDEN LAYER 1
    torch.nn.ReLU(),

    torch.nn.Conv2d(128, 256, kernel_size=3, padding=1), # HIDDEN LAYER 2
    torch.nn.ReLU(),

    torch.nn.Flatten(),

    torch.nn.Linear(16384, 2048), # HIDDEN LAYER 3
    torch.nn.ReLU(),
    
    torch.nn.Linear(2048, 4608) # OUTPUT LAYER
    ).to(device)

# SOME HEADER HERE
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(omniscius.parameters(), lr=1e-4)
accuracy_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=4608).to(device)

for epoch in range(EPOCHS):
    for sub_epoch in range(SUB_EPOCHS):
        accuracy_metric.reset()
