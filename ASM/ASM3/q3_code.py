import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Define the transformations with data augmentation for training
train_transform = transforms.Compose(
    [
        transforms.Resize((150, 150)),  # Slightly larger for cropping
        # transforms.RandomResizedCrop(150, scale=(0.8, 1.0)),  # Random zoom
        transforms.RandomHorizontalFlip(p=0.5),  # ✅ Essential for cats/dogs
        transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandomGrayscale(p=0.1),  # Occasionally remove color
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


test_transform = transforms.Compose(
    [
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


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


# 3. Load the training dataset with validation
train_data_dir = "PetImage_dataset/train"
image_paths_train = []
labels_train = []


def is_valid_image(filepath):
    """Check if an image file can be opened"""
    try:
        img = Image.open(filepath)
        img.verify()  # Verify it's an actual image
        return True
    except Exception:
        return False


# Load images from 'cat' folder in training set
cat_train_folder = os.path.join(train_data_dir, "cat")
corrupted_count = 0
for img_file in os.listdir(cat_train_folder):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(cat_train_folder, img_file)
        if is_valid_image(filepath):
            image_paths_train.append(filepath)
            labels_train.append(0)  # Label for cat
        else:
            corrupted_count += 1
            print(f"Skipping corrupted image: {filepath}")

# Load images from 'dog' folder in training set
dog_train_folder = os.path.join(train_data_dir, "dog")
for img_file in os.listdir(dog_train_folder):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(dog_train_folder, img_file)
        if is_valid_image(filepath):
            image_paths_train.append(filepath)
            labels_train.append(1)  # Label for dog
        else:
            corrupted_count += 1
            print(f"Skipping corrupted image: {filepath}")

print(f"Skipped {corrupted_count} corrupted images in training set")

# Convert to numpy arrays
image_paths_train = np.array(image_paths_train)
labels_train = np.array(labels_train)

print(
    f"Training samples: {len(image_paths_train)} (Cats: {np.sum(labels_train == 0)}, Dogs: {np.sum(labels_train == 1)})"
)

# Create training dataset and dataloader
train_dataset = CatDogDataset(
    image_paths_train, labels_train, transform=train_transform
)
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)

# 4. Load the testing dataset with validation
test_data_dir = "PetImage_dataset/test"
image_paths_test = []
labels_test = []

corrupted_count = 0

# Load images from 'cat' folder in testing set
cat_test_folder = os.path.join(test_data_dir, "cat")
for img_file in os.listdir(cat_test_folder):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(cat_test_folder, img_file)
        if is_valid_image(filepath):
            image_paths_test.append(filepath)
            labels_test.append(0)  # Label for cat
        else:
            corrupted_count += 1
            print(f"Skipping corrupted image: {filepath}")

# Load images from 'dog' folder in testing set
dog_test_folder = os.path.join(test_data_dir, "dog")
for img_file in os.listdir(dog_test_folder):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(dog_test_folder, img_file)
        if is_valid_image(filepath):
            image_paths_test.append(filepath)
            labels_test.append(1)  # Label for dog
        else:
            corrupted_count += 1
            print(f"Skipping corrupted image: {filepath}")

print(f"Skipped {corrupted_count} corrupted images in testing set")

# Convert to numpy arrays
image_paths_test = np.array(image_paths_test)
labels_test = np.array(labels_test)

print(
    f"Testing samples: {len(image_paths_test)} (Cats: {np.sum(labels_test == 0)}, Dogs: {np.sum(labels_test == 1)})"
)

# Create testing dataset and dataloader
test_dataset = CatDogDataset(image_paths_test, labels_test, transform=test_transform)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)


# 5. [Task 1]: Define the CNN model
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.25)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(0.25)

        # Convolutional Block 4
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(0.25)

        # Calculate the flattened size: 150x150 -> 75x75 -> 37x37 -> 18x18 -> 9x9
        # After 4 pooling layers: 9x9x256 = 20736
        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 9 * 9, 512)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 128)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.3)

        # Output layer with Sigmoid activation for binary classification
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Convolutional Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Convolutional Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Convolutional Block 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Convolutional Block 4
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        # Flatten
        x = self.flatten(x)

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout5(x)

        x = self.fc2(x)
        x = self.relu6(x)
        x = self.dropout6(x)

        # Output layer with Sigmoid
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x


# 6. Initialize the model, define loss function and optimizer.
model = MyCNN().to(device)
print(f"\nModel Architecture:\n{model}")
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler for better convergence
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# 7. [Task 2] Train the model and store loss and accuracy values

num_epochs = 30  # Increased epochs for better training
train_losses = []
test_accuracies = []  # Changed variable name for clarity

print("\nStarting training...\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        labels = labels.float().view(-1, 1).to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

    # Calculate average epoch loss
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    # Update learning rate based on loss
    scheduler.step(epoch_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

    # [MODIFIED] Evaluate EVERY epoch
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to device
            images = images.to(device)
            labels = labels.float().view(-1, 1).to(device)

            # Forward pass
            outputs = model(images)

            # Convert probabilities to binary predictions (threshold = 0.5)
            predicted = (outputs >= 0.5).float()

            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Test Accuracy at Epoch {epoch + 1}: {accuracy:.2f}%\n")

# Final evaluation on test set
print("\nFinal Evaluation on Test Set:")
final_accuracy = test_accuracies[-1]
print(f"Final Test Accuracy: {final_accuracy:.2f}%")

# 8. [Task 4] Plot training loss and validation accuracy
print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)

matplotlib.use("Agg")  # Use non-interactive backend

# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, marker="o", linestyle="-", color="b")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("training_loss_less_augmented.png", dpi=300, bbox_inches="tight")
print("Training loss plot saved as 'training_loss_less_augmented.png'")
plt.close()

# Plot Test Accuracy (MODIFIED - now plots all epochs)
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, num_epochs + 1), test_accuracies, marker="s", linestyle="-", color="g"
)
plt.title("Test Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.savefig("test_accuracy_less_augmented.png", dpi=300, bbox_inches="tight")
print("Test accuracy plot saved as 'test_accuracy_less_augmented.png'")
plt.close()


# Save the model
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "train_losses": train_losses,
        "test_accuracies": test_accuracies,
    },
    "cat_dog_classifier_less_augmented_with_history.pth",
)

print("Model and training history saved!")

print("\n" + "=" * 50)
print(f"FINAL TEST ACCURACY: {test_accuracies[-1]:.2f}%")
print("=" * 50)
