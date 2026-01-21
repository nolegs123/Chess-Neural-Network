import json
import matplotlib.pyplot as plt

PATH = r"C:\Users\mikke\Desktop\training_log.json"
SUB_EPOCHS_PER_EPOCH = 13

with open(PATH, "r") as f:
    data = json.load(f)

x_vals = []
train_acc = []
test_acc = []
train_loss = []
test_loss = []

def draw_plot(x, ta, va, tl, vl):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x, ta, label="Train accuracy")
    plt.plot(x, va, label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, tl, label="Train loss")
    plt.plot(x, vl, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.suptitle("Omniscius")
    plt.tight_layout()
    plt.show()

for d in data:
    epoch = d["epoch"]
    sub_epoch = d["sub_epoch"]

    x_val = epoch + (sub_epoch - 1) / SUB_EPOCHS_PER_EPOCH

    x_vals.append(x_val)
    train_acc.append(d["train_accuracy"])
    test_acc.append(d["test_accuracy"])
    train_loss.append(d["train_loss"])
    test_loss.append(d["test_loss"])

draw_plot(
    x_vals,
    train_acc,
    test_acc,
    train_loss,
    test_loss
)
