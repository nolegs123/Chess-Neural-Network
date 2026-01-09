import torch
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
from data_encoding import generate_all_uci_moves

# SETTINGS
BATCH_SIZE = 500
NUMBER_OF_CLASSES = 4608
EPOCHS = 5
SUB_EPOCHS = 20
LR = "din mor"

# FILE PATHS
TESTING_PATH = "C:\Users\mikke\source\repos\Chess-Neural-Network\Chess Neural Network\Data\testing_data"
TRAINING_PATH = "C:\Users\mikke\source\repos\Chess-Neural-Network\Chess Neural Network\Data\training_data"

