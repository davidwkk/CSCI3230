import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define the transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),  # Resize images
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# 2. Custom Dataset Class
class CatDogDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

# 3. Load the training dataset
train_data_dir = 'dataset/train'  # Directory for training data
image_paths_train = []
labels_train = []

# Load images from 'cat' folder in training set
cat_train_folder = os.path.join(train_data_dir, 'cat')
for img_file in os.listdir(cat_train_folder):
    image_paths_train.append(os.path.join(cat_train_folder, img_file))
    labels_train.append(0)  # Label for cat

# Load images from 'dog' folder in training set
dog_train_folder = os.path.join(train_data_dir, 'dog')
for img_file in os.listdir(dog_train_folder):
    image_paths_train.append(os.path.join(dog_train_folder, img_file))
    labels_train.append(1)  # Label for dog

# Convert to numpy arrays
image_paths_train = np.array(image_paths_train)
labels_train = np.array(labels_train)

# Create training dataset and dataloader
train_dataset = CatDogDataset(image_paths_train, labels_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 4. Load the testing dataset
test_data_dir = 'dataset/test'  # Directory for testing data
image_paths_test = []
labels_test = []

# Load images from 'cat' folder in testing set
cat_test_folder = os.path.join(test_data_dir, 'cat')
for img_file in os.listdir(cat_test_folder):
    image_paths_test.append(os.path.join(cat_test_folder, img_file))
    labels_test.append(0)  # Label for cat

# Load images from 'dog' folder in testing set
dog_test_folder = os.path.join(test_data_dir, 'dog')
for img_file in os.listdir(dog_test_folder):
    image_paths_test.append(os.path.join(dog_test_folder, img_file))
    labels_test.append(1)  # Label for dog

# Convert to numpy arrays
image_paths_test = np.array(image_paths_test)
labels_test = np.array(labels_test)

# Create testing dataset and dataloader
test_dataset = CatDogDataset(image_paths_test, labels_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 5. [Task 1]: Define the CNN model
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        '''
        Please define the model here
        '''
        

    def forward(self, x):
        '''
        Please write the forward function here
        '''
        return x

# 6. Initialize the model, define loss function and optimizer.
#  You can change any hyperparameters and parameters to improve performance.
model = MyCNN().to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 7. [Task 2 ] Train the model and store loss and accuracy values

num_epochs = 10  # Set the total number of epochs
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        labels = labels.float().view(-1, 1).to(device)  # Reshape labels for BCELoss
        images = images.to(device)  # Move images to device
        '''
        please write the training code 
        '''
        
        
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # [Task 3 ]Evaluate every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                labels = labels.float().view(-1, 1).to(device)  # Reshape labels for BCELoss
                images = images.to(device)  # Move images to device
                
                '''
                please write the evaluation code 
                '''
              

        #accuracy = 100 * correct / total
        #val_accuracies.append(accuracy)
        #print(f'Test accuracy of the model every 10 epochs: {accuracy:.2f}%')

# 8. Plot the training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(1, num_epochs + 1))
plt.legend()
plt.show()
#plt.savefig("loss.png")

# 9. Plot the test accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), val_accuracies, marker='o', label='Test Accuracy', color='orange')
plt.title('Test Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.xticks(range(1, num_epochs + 1))
plt.legend()
plt.show()
#plt.savefig("acc.png")