import torch
import torchmetrics
import json
import os
from torch.utils.data import DataLoader, TensorDataset
from data_encoding import fen_to_tensor
from data_encoding import generate_all_uci_moves

# SETTINGS
BATCH_SIZE = 500
EPOCHS = 25
SUB_EPOCHS = 13
SHARDS_PER_SUB_EPOCH = 75

# FILE PATHS
TESTING_PATH = r"C:\Users\Mikkel\Desktop\New folder (8)\testing_data"
TRAINING_PATH = r"C:\Users\Mikkel\Desktop\New folder (8)\training_data"
MODEL_SAVING_PATH = r"C:\Users\Mikkel\Desktop\New folder (8)"
DATA_PLOT_PATH = r"C:\Users\Mikkel\Desktop\New folder (8)\data_plot\training_log.json"

# INITIALIZATION
all_moves = generate_all_uci_moves()
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
print(device)
print(torch.version.cuda)

# MODEL

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

# DATA PLOTTING

if not os.path.exists(DATA_PLOT_PATH):
    with open(DATA_PLOT_PATH, "w") as f:
        json.dump([], f)

for epoch in range(EPOCHS):
    for sub_epoch in range(SUB_EPOCHS):

        # ---------- TRAIN ----------
        accuracy_metric.reset()
        train_loss_sum = 0.0
        train_batch_count = 0

        for shard_id in range(sub_epoch * SHARDS_PER_SUB_EPOCH, (sub_epoch + 1) * SHARDS_PER_SUB_EPOCH):
            print(shard_id)

            train_shard_path = f"{TRAINING_PATH}/FEN_optimal_move_{shard_id}.txt"
            train_dataset = load_shard(train_shard_path, uci_to_id)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                out = omniscius(x)
                loss = loss_function(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                train_batch_count += 1

                preds = torch.argmax(out, dim=1)
                accuracy_metric.update(preds, y)

        train_acc = accuracy_metric.compute().item()
        train_loss = train_loss_sum / train_batch_count

        print(f"Epoch: {epoch+1}/{EPOCHS}, Sub-epoch: {sub_epoch+1}/{SUB_EPOCHS}")
        print(f"Train acc = {train_acc * 100:.4f}%, Train loss = {train_loss:.4f}")

        # ---------- TEST ----------
        accuracy_metric.reset()
        test_loss_sum = 0.0
        test_batch_count = 0

        omniscius.eval()

        with torch.no_grad():
            for shard_id in range(975, 1000):
                print(shard_id)

                test_shard_path = f"{TESTING_PATH}/FEN_optimal_move_{shard_id}.txt"
                test_dataset = load_shard(test_shard_path, uci_to_id)
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)

                    out = omniscius(x)
                    loss = loss_function(out, y)

                    test_loss_sum += loss.item()
                    test_batch_count += 1

                    preds = torch.argmax(out, dim=1)
                    accuracy_metric.update(preds, y)

        test_acc = accuracy_metric.compute().item()
        test_loss = test_loss_sum / test_batch_count

        print(f"Test acc = {test_acc * 100:.4f}%, Test loss = {test_loss:.4f}")

        # SAVE TO JSON
        log_entry = {
            "epoch": epoch + 1,
            "sub_epoch": sub_epoch + 1,
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "test_accuracy": test_acc,
            "test_loss": test_loss
        }

        torch.save(omniscius.state_dict(), f"{MODEL_SAVING_PATH}/omniscius_epoch{epoch+1}_subepoch{sub_epoch+1}.pt")

        with open(DATA_PLOT_PATH, "r") as f:
            data = json.load(f)

        data.append(log_entry)

        with open(DATA_PLOT_PATH, "w") as f:
            json.dump(data, f, indent=2)