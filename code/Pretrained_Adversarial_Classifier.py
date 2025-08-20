import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# --------------------- 0. Fix Random Seed (for reproducibility) --------------------- #
def set_seed(seed=42):
    """Set all random seeds to ensure experiment reproducibility."""
    random.seed(seed)                        # Python random seed
    np.random.seed(seed)                     # NumPy random seed
    torch.manual_seed(seed)                  # PyTorch CPU random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)         # PyTorch GPU random seed
        torch.cuda.manual_seed_all(seed)     # Set for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure CUDA convolution is deterministic
    torch.backends.cudnn.benchmark = False     # Disable CUDA auto-tuner (may affect speed, but ensures reproducibility)


# --------------------- 1. Indexed CIFAR10 Dataset (inherits from official dataset) --------------------- #
class IndexedCIFAR10(torchvision.datasets.CIFAR10):
    """Adds index to training dataset, returns (index, data, target)"""
    def __getitem__(self, index):
        # Call parent method to get data and label
        data, target = super().__getitem__(index)
        # Return index, data, label (index can be used for sample tracking)
        return index, data, target


# --------------------- 2. Model Definition --------------------- #
class ModifiedCIFARCNN(nn.Module):
    def __init__(self):
        super(ModifiedCIFARCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(40, 60, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(60, 80, kernel_size=2, stride=1)
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(60 * 2 * 2, 160)
        self.fc2 = nn.Linear(80 * 1 * 1, 160)
        self.fc3 = nn.Linear(160, 10)
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Branch 1
        x = self.relu(self.conv1(x))  # 3x32x32 -> 20x28x28
        x = self.pool(x)              # 20x14x14
        x = self.relu(self.conv2(x))  # 20x14x14 -> 40x12x12
        x = self.pool(x)              # 40x6x6
        x = self.relu(self.conv3(x))  # 40x6x6 -> 60x4x4
        x_pool3 = self.pool(x)        # 60x2x2 (saved for fc1)
        
        # Branch 1 fully connected
        x_fc1 = x_pool3.view(-1, 60 * 2 * 2)
        x_fc1 = self.relu(self.fc1(x_fc1))  # 240 -> 160

        # Branch 2
        x_conv4 = self.relu(self.conv4(x_pool3))  # 60x2x2 -> 80x1x1
        x_conv4 = x_conv4.view(-1, 80 * 1 * 1)
        x_conv4 = self.relu(self.fc2(x_conv4))    # 80 -> 160

        # Feature fusion and output
        x_add = x_fc1 + x_conv4
        x_add = self.relu(x_add)
        out = self.fc3(x_add)
        return out


# --------------------- 3. Data Loading and Preprocessing (with index for training set) --------------------- #
def load_cifar10(base_dir, batch_size):
    data_path = os.path.join(base_dir, "data", "cifar10_data")
    os.makedirs(data_path, exist_ok=True)  # Ensure directory exists
    
    # Data preprocessing parameters
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    # Use custom indexed dataset for training set (returns (index, data, label))
    trainset = IndexedCIFAR10(
        root=data_path, train=True, download=True, transform=transform_train
    )
    # Test set remains as is (no index needed)
    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test
    )

    # Data loaders (shuffle=True is reproducible due to fixed seed)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2,
        worker_init_fn=lambda x: np.random.seed(42 + x)  # Fix seed for sub-processes
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2,
        worker_init_fn=lambda x: np.random.seed(42 + x)
    )

    return trainloader, testloader


# --------------------- 4. Training and Testing Function (adapted for indexed training set) --------------------- #
def train_and_test(model, trainloader, testloader, criterion, optimizer, device, num_epochs):
    """Train and test the model, adapted for training data with indices"""
    for epoch in range(num_epochs):
        # Training (handle indexed training data)
        model.train()
        running_loss = 0.0
        for indices, inputs, labels in trainloader:  # 'indices' are added here
            # Indices can be used for later analysis (e.g., tracking which samples were used), not used for training here
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(trainloader)

        # Testing (test set has no indices, keeps original logic)
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = running_loss / len(testloader)
        test_acc = 100 * correct / total

        # Print log
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.2f}%")


# --------------------- 5. Main Function --------------------- #
def main():
    # Step 1: Set random seed (ensure reproducibility)
    set_seed(seed=42)
    
    # Unified core parameters
    base_dir = "./xie/FNN_Shapley"   # Root directory
    num_epochs = 100                 # Number of epochs
    batch_size = 64                  # Batch size
    lr = 0.01                        # Initial learning rate

    # Device parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (Random seed is fixed, results are reproducible)")

    # Initialize components
    model = ModifiedCIFARCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr
    )

    # Load data (training set with indices)
    trainloader, testloader = load_cifar10(base_dir=base_dir, batch_size=batch_size)

    # Start training and testing
    print(f"Training parameters: epochs={num_epochs}, batch_size={batch_size}, learning_rate={lr}")
    train_and_test(model, trainloader, testloader, criterion, optimizer, device, num_epochs)
    
    # Save model
    save_dir = os.path.join(base_dir, "cifar10_cnn_model")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "pretrained_cifar10_cnn_1.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")


if __name__ == "__main__":
    main()