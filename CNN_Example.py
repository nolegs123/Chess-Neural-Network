#%% Import libraries
import torch
import torchvision
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
import os
from glob import glob

#%% Settings
data_path = "/home/mnsc/dtu/data/imagenette2-160/train/"
class_path = ["n01440764", "n02102040", "n02979186"]
class_labels = ["fish", "dog", "stereo"]
num_classes = 3
batch_size = 256
num_epochs = 1000
learning_rate = 0.01
weight_decay = 0.

#%% Data preprocessing function
def preprocess(image):
    # Convert to color if black-and-white
    if image.shape[0] == 1:
        image = image.repeat(3,1,1)
    # Convert to floating point numbers between 0 and 1
    image = image.float()/255
    # Crop to 160x160
    image = torchvision.transforms.functional.crop(image, 0, 0, 160, 160)
    # Resize to 32x32
    image = torchvision.transforms.functional.resize(image, [32, 32], antialias=True)
    return image

#%% Load data
# Empty lists to store images and labels
images = []
labels = []

# Add each image to list
for i, label in enumerate(class_labels):
    # Get all JPEG files in directory
    filenames = glob(os.path.join(data_path, class_path[i], '*.JPEG'))
    for file in filenames:
        # Put image on list
        image = torchvision.io.read_image(file)
        image = preprocess(image)
        images.append(image)
        # Put label on list
        labels.append(i)

# Put data into a tensor
images_tensor = torch.stack(images).float()
labels_tensor = torch.tensor(labels)

#%% Device
# Run on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu" 

#%% Create dataloader
train_data = TensorDataset(images_tensor, labels_tensor)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

#%% Neural network
net = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, kernel_size=3),  # b x 16 x 30 x 30
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),      # b x 16 x 15 x 15
    torch.nn.Conv2d(16, 16, kernel_size=3), # b x 16 x 13 x 13
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),      # b x 16 x 6 x 6
    torch.nn.Flatten(),                     # 16*6*6=576
    torch.nn.Linear(576, 3)                 # 3
).to(device)

# %% Load trained network from file
# net.load_state_dict(torch.load('net.pt'))

#%% Loss and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=.01, weight_decay=weight_decay)

#%% Metrics
accuracy_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes).to(device)

#%% Train
step = 0
for epoch in range(num_epochs):
    accuracy_metric.reset()
    for x, y in train_loader:
        step += 1

        # Put data on GPU 
        x = x.to(device)
        y = y.to(device)

        # Compute loss and take gradient step
        out = net(x)
        loss = loss_function(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        # Update accuracy metric
        accuracy_metric.update(out, y)

    # Print accuracy for epoch            
    acc = accuracy_metric.compute()
    print(f'Training accuracy = {acc}')

# %% Save the trained model
# torch.save(net.state_dict(), 'net.pt')
