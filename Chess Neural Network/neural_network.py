import torch
import torchmetrics
import json
import os
import re
from torch.utils.data import DataLoader, TensorDataset
from data_encoding import fen_to_tensor
from data_encoding import generate_all_uci_moves

# SETTINGS
BATCH_SIZE = 500
EPOCHS = 50
SUB_EPOCHS = 13
SHARDS_PER_SUB_EPOCH = 75

# FILE PATHS
TESTING_PATH = r"C:\Users\mikke\source\repos\Chess-Neural-Network\Chess Neural Network\Data\testing_data"
TRAINING_PATH = r"C:\Users\mikke\source\repos\Chess-Neural-Network\Chess Neural Network\Data\training_data"
MODEL_SAVING_PATH = r"C:\Users\mikke\Desktop\model\forsøg1-ny_model"
DATA_PLOT_PATH = r"C:\Users\mikke\source\repos\Chess-Neural-Network\Chess Neural Network\logging\training_log.json"

# INITIALIZATION
all_moves = generate_all_uci_moves()
uci_to_id = {uci: idx for idx, uci in enumerate(all_moves)}
id_to_uci = {idx: uci for idx, uci in enumerate(all_moves)}

def load_shard(path: str, uci_to_id: dict):
    positions = []
    labels = []

    with open(path, "r") as f:
        for line in f:
            fen, best_move = line.strip().split(" | ")
            positions.append(fen_to_tensor(fen))
            labels.append(uci_to_id[best_move])

    x = torch.stack(positions)
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)

def load_latest_model(model_dir: str):
    if not os.path.exists(model_dir):
        return None, 0, 0

    pattern = re.compile(r"omniscius_epoch(\d+)_subepoch(\d+)\.pt")
    candidates = []

    for fname in os.listdir(model_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            subepoch = int(match.group(2))
            candidates.append((epoch, subepoch, fname))

    if not candidates:
        return None, 0, 0

    candidates.sort(key=lambda x: (x[0], x[1]))
    epoch, subepoch, fname = candidates[-1]

    return os.path.join(model_dir, fname), epoch - 1, subepoch - 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.version.cuda)

# MODEL
omniscius = torch.nn.Sequential(
    torch.nn.Conv2d(19, 64, kernel_size=3, padding=1),
    torch.nn.ReLU(),

    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
    torch.nn.ReLU(),

    torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
    torch.nn.ReLU(),

    torch.nn.AdaptiveAvgPool2d((1, 1)),
    torch.nn.Flatten(),

    torch.nn.Linear(256, 512),
    torch.nn.ReLU(),

    torch.nn.Dropout(p=0.3),
    torch.nn.Linear(512, 4608)
).to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(omniscius.parameters(), lr=1e-4)
accuracy_metric = torchmetrics.classification.Accuracy(
    task="multiclass",
    num_classes=4608
).to(device)

# LOAD LATEST MODEL IF EXISTS
latest_model_path, start_epoch, start_sub_epoch = load_latest_model(MODEL_SAVING_PATH)

if latest_model_path is not None:
    print(f"Loading model: {latest_model_path}")
    omniscius.load_state_dict(torch.load(latest_model_path, map_location=device))
else:
    start_epoch = 0
    start_sub_epoch = 0

# DATA PLOTTING
if not os.path.exists(DATA_PLOT_PATH):
    with open(DATA_PLOT_PATH, "w") as f:
        json.dump([], f)

for epoch in range(start_epoch, EPOCHS):
    for sub_epoch in range(start_sub_epoch if epoch == start_epoch else 0, SUB_EPOCHS):

        accuracy_metric.reset()
        train_loss_sum = 0.0
        train_batch_count = 0
        omniscius.train()

        for shard_id in range(sub_epoch * SHARDS_PER_SUB_EPOCH,
                              (sub_epoch + 1) * SHARDS_PER_SUB_EPOCH):
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
        print(f"Train acc = {train_acc*100:.4f}%, Train loss = {train_loss:.4f}")

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

        print(f"Test acc = {test_acc*100:.4f}%, Test loss = {test_loss:.4f}")

        log_entry = {
            "epoch": epoch + 1,
            "sub_epoch": sub_epoch + 1,
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "test_accuracy": test_acc,
            "test_loss": test_loss
        }

        torch.save(
            omniscius.state_dict(),
            f"{MODEL_SAVING_PATH}/omniscius_epoch{epoch+1}_subepoch{sub_epoch+1}.pt"
        )

        with open(DATA_PLOT_PATH, "r") as f:
            data = json.load(f)

        data.append(log_entry)

        with open(DATA_PLOT_PATH, "w") as f:
            json.dump(data, f, indent=2)
