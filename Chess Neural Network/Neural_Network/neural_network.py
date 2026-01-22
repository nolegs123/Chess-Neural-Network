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
EPOCHS = 1000
SUB_EPOCHS = 13
SHARDS_PER_SUB_EPOCH = 75

# FILE PATHS
TESTING_PATH = "/zhome/72/1/225993/Desktop/Chess Network/Training/Data/Testing_Data"
TRAINING_PATH = "/zhome/72/1/225993/Desktop/Chess Network/Training/Data/Training_Data"
MODEL_SAVING_PATH = "/zhome/72/1/225993/Desktop/Chess Network/Training/Experiment/omnisciusV1/Models"
DATA_PLOT_PATH = "/zhome/72/1/225993/Desktop/Chess Network/Training/Experiment/omnisciusV1/Logging/training_log.json"

# INITIALIZATION
all_moves = generate_all_uci_moves()
uci_to_id = {uci: idx for idx, uci in enumerate(all_moves)}

def load_shard(path: str, uci_to_id: dict):
    positions = []
    labels = []

    with open(path, "r") as f:
        for line in f:
            fen, best_move = line.strip().split(" | ")
            positions.append(fen_to_tensor(fen))
            labels.append(uci_to_id[best_move])

    return TensorDataset(
        torch.stack(positions),
        torch.tensor(labels, dtype=torch.long)
    )

def load_latest_model(model_dir: str):
    if not os.path.exists(model_dir):
        return None, 0, 0

    pattern = re.compile(r"omniscius_epoch(\d+)_subepoch(\d+)\.pt")
    candidates = []

    for fname in os.listdir(model_dir):
        m = pattern.match(fname)
        if m:
            epoch = int(m.group(1)) - 1
            subepoch = int(m.group(2)) - 1
            candidates.append((epoch, subepoch, fname))

    if not candidates:
        return None, 0, 0

    epoch, subepoch, fname = max(candidates, key=lambda x: (x[0], x[1]))

    subepoch += 1
    if subepoch >= SUB_EPOCHS:
        epoch += 1
        subepoch = 0

    return os.path.join(model_dir, fname), epoch, subepoch
# ------------------------------------

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

    torch.nn.Flatten(),

    torch.nn.Linear(16384, 1024),
    torch.nn.ReLU(),

    torch.nn.Linear(1024, 4608)
).to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(omniscius.parameters(), lr=1e-4)
accuracy_metric = torchmetrics.classification.Accuracy(
    task="multiclass",
    num_classes=4608
).to(device)

# LOAD CHECKPOINT
latest_model_path, start_epoch, start_sub_epoch = load_latest_model(MODEL_SAVING_PATH)

if latest_model_path:
    print(f"Loading model: {latest_model_path}")
    ckpt = torch.load(latest_model_path, map_location=device)
    omniscius.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
else:
    start_epoch = 0
    start_sub_epoch = 0

print(f"Resuming from epoch={start_epoch}, sub_epoch={start_sub_epoch}")

# LOG FILE
if not os.path.exists(DATA_PLOT_PATH):
    with open(DATA_PLOT_PATH, "w") as f:
        json.dump([], f)

for epoch in range(start_epoch, EPOCHS):
    for sub_epoch in range(start_sub_epoch if epoch == start_epoch else 0, SUB_EPOCHS):

        omniscius.train()
        accuracy_metric.reset()
        train_loss_sum = 0.0
        train_batch_count = 0

        for shard_id in range(
            sub_epoch * SHARDS_PER_SUB_EPOCH,
            (sub_epoch + 1) * SHARDS_PER_SUB_EPOCH
        ):
            train_dataset = load_shard(
                f"{TRAINING_PATH}/FEN_optimal_move_{shard_id}.txt",
                uci_to_id
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                persistent_workers=False
            )

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

                accuracy_metric.update(torch.argmax(out, dim=1), y)

            del train_loader
            del train_dataset

        train_acc = accuracy_metric.compute().item()
        train_loss = train_loss_sum / train_batch_count

        print(f"Epoch {epoch+1} Sub {sub_epoch+1} | Train acc {train_acc:.4f} loss {train_loss:.4f}")

        omniscius.eval()
        accuracy_metric.reset()
        test_loss_sum = 0.0
        test_batch_count = 0

        with torch.no_grad():
            for shard_id in range(975, 1000):
                test_dataset = load_shard(
                    f"{TESTING_PATH}/FEN_optimal_move_{shard_id}.txt",
                    uci_to_id
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    persistent_workers=False
                )

                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)

                    out = omniscius(x)
                    loss = loss_function(out, y)

                    test_loss_sum += loss.item()
                    test_batch_count += 1
                    accuracy_metric.update(torch.argmax(out, dim=1), y)

                del test_loader
                del test_dataset

        test_acc = accuracy_metric.compute().item()
        test_loss = test_loss_sum / test_batch_count

        print(f"Test acc {test_acc:.4f} loss {test_loss:.4f}")

        torch.save(
            {"model": omniscius.state_dict(), "optimizer": optimizer.state_dict()},
            f"{MODEL_SAVING_PATH}/omniscius_epoch{epoch+1}_subepoch{sub_epoch+1}.pt"
        )

        with open(DATA_PLOT_PATH, "r") as f:
            data = json.load(f)

        data.append({
            "epoch": epoch + 1,
            "sub_epoch": sub_epoch + 1,
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "test_accuracy": test_acc,
            "test_loss": test_loss
        })

        tmp = DATA_PLOT_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, DATA_PLOT_PATH)
