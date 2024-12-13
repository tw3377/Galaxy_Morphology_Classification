#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:19:30 2024


"""
#%% 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, random_split

#%% importing data 

path =  "/Users/tanyawalia/Desktop/Tanya Project/classbalanced_data.npz"
dataset = np.load(path)

images = dataset['data']
labels = dataset['labels']

#%% 
images_tensor =torch.tensor(images)
labels_tensor =torch.tensor(labels)

#%%Defining your dataset 

# Create a TensorDataset
dataset = TensorDataset(images_tensor, labels_tensor)

# Define the split ratio
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation

# Perform the split
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Get the tensors
x_train, y_train = train_dataset[:]
x_val, y_val = val_dataset[:]

# Convert to tensors if they 
print(f'Training set size: {len(x_train)}')
print(f'Validation set size: {len(x_val)}')


#%% Defining model


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(0.20)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(0.20)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(64 * 10 * 10, 128)  # Assuming 40x40 images go down to 10x10 after pooling
        self.relu3 = nn.LeakyReLU(0.20)
        self.fc2 = nn.Linear(128, num_classes)  # Output layer, number of classes for classification
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 10 * 10)  # Flatten the image for the fully connected layer
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs, device, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate can be adjusted

    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        train_losses.append(running_loss / len(train_loader))

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

        validate_model(model, val_loader, criterion, device, val_losses)

    # Save the model weights
    torch.save(model.state_dict(), save_path)
    
    return train_losses, val_losses

def validate_model(model, val_loader, criterion, device, val_losses):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accuracy = correct / total
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')



#%% Main Training Script 

# Create a TensorDataset
dataset = TensorDataset(images_tensor, labels_tensor)

# Define the split ratio
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation

# Perform the split
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
model = SimpleCNN(num_classes=len(torch.unique((labels_tensor)))).to(device)

# Choosing a loss function 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate can be adjusted

# Train the model
num_epochs = 30  # Adjust according to your dataset size and complexity
save_path = 'model_weights.pth'
train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs, device, save_path)

# Save the loss progress
np.save('train_losses.npy', train_losses)
np.save('val_losses.npy', val_losses)

print('Training complete. Model weights and loss history saved.')


#%% 
# Uncomment if importing the model weights 

# Load the trained model weights
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SimpleCNN(num_classes=len(torch.unique(y_augmented)))
# model.load_state_dict(torch.load('model_weights.pth'))
# model.to(device)

# Setting to evaluation mode. 
model.eval()



#%% 
correct = 0
total = 0
predictions = []
actual_labels = []

with torch.no_grad():
    for val_images, val_labels in val_loader:
        val_images, val_labels = val_images.to(device), val_labels.to(device)
        outputs = model(val_images)
        _, predicted = torch.max(outputs.data, 1)
        total += val_labels.size(0)
        correct += (predicted == val_labels).sum().item()
        predictions.extend(predicted.cpu().numpy())
        actual_labels.extend(val_labels.cpu().numpy())

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')

# Optionally, you can display some images with their predicted and actual labels for verification
for i in range(10):  # Show 5 images
    image = val_images[i].cpu().numpy().squeeze()  # Remove batch dimension and squeeze to make it 2D
    true_label = actual_labels[i]
    predicted_label = predictions[i]
    print(f'Image {i+1}, True Label: {true_label}, Predicted Label: {predicted_label}')

    # If you want to visualize the image
    # Use matplotlib or PIL to display the image
    plt.imshow(image, cmap='gray')  # Use cmap='gray' if the images are grayscale
    plt.axis('off')  
    
    # Set title with true and predicted labels
    plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=10)
    plt.show()

# plot validation and training losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# print model summary 
from torchsummary import summary 
summary(model, input_size=(1, 40, 40))