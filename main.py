import torch 
import torchvision
from utils import BasicUtils, TrainTestUtils
import numpy as np
import matplotlib.pyplot as plt
from alexnet import AlexNet

# Choosing device (NVIDIA CUDA GPU, Apple Silicon GPU, CPU)
device = BasicUtils().device_chooser()
print("Using device: ",device)

# Loss lists
train_losses = []
test_losses = []

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# Resizing the dataset to 227x227
transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((227,227)), torchvision.transforms.ToTensor()])

# Dataset
train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, transform=transforms, download=True)
test_dataset = torchvision.datasets.CIFAR10(root="data", train=False, transform=transforms, download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Choosing model
model = AlexNet(num_classes).to(device)

# Defining loss and optimizer functions
loss_fn = BasicUtils().loss_chooser("crossentropy")
optimizer = BasicUtils().optim_chooser("adam", model, learning_rate)

# Train and test stage
for i in range(num_epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    train_losses.append(TrainTestUtils().train(train_loader, model, loss_fn, optimizer, i, num_epochs, batch_size))
    test_losses.append(TrainTestUtils().test(test_loader, model, loss_fn))

# Showing results (train and test losses)
plt.plot(train_losses,"g",label="train loss")
plt.plot(test_losses,"r",label="test loss")
plt.title("alexnet losses")
plt.xlabel("epochs")
plt.ylabel("losses")
plt.legend(loc="upper left")
plt.show()

# Saving model
model_name = input("Enter model name:")
BasicUtils().model_saver(model,model_name)