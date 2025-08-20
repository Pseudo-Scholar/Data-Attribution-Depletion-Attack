import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import torchvision
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.func import functional_call, grad, vmap
from torch.nn.utils import parameters_to_vector

# ====================== 🔒 Fix Random Seed ====================== #
def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- Indexed CIFAR10 Dataset -------------------- #
class IndexedCIFAR10(torchvision.datasets.CIFAR10):
    """Returns (index, image tensor, label) to track samples."""
    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return index, data, target

# -------------------- Poisoned Dataset Class -------------------- #
class PoisonedCIFAR10Dataset(torch.utils.data.Dataset):
    """Replaces original samples with backdoor samples."""
    def __init__(self, original_dataset, poisoned_images, poisoned_indices):
        self.original_dataset = original_dataset
        self.poisoned_images = poisoned_images  # List of tensors with triggers
        self.poisoned_indices = poisoned_indices  # List of poisoned sample indices
        
        # Create an index map for quick lookup of poisoned samples
        self.index_map = {idx: i for i, idx in enumerate(poisoned_indices)}
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        if index in self.poisoned_indices:
            # Return a poisoned sample (with its original label)
            poisoned_idx = self.index_map[index]
            _, _, original_label = self.original_dataset[index]
            return index, self.poisoned_images[poisoned_idx], original_label
        else:
            # Return an original sample
            return self.original_dataset[index]

# -------------------- Model Definition (CIFAR10 CNN) -------------------- #
class ModifiedCIFARCNN(nn.Module):
    def __init__(self):
        super(ModifiedCIFARCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(40, 60, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(60, 80, kernel_size=2, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(60 * 2 * 2, 160)
        self.fc2 = nn.Linear(80 * 1 * 1, 160)
        self.fc3 = nn.Linear(160, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x_pool3 = self.pool(x)
        
        x_fc1 = x_pool3.view(-1, 60 * 2 * 2)
        x_fc1 = self.relu(self.fc1(x_fc1))

        x_conv4 = self.relu(self.conv4(x_pool3))
        x_conv4 = x_conv4.view(-1, 80 * 1 * 1)
        x_conv4 = self.relu(self.fc2(x_conv4))

        x_add = x_fc1 + x_conv4
        x_add = self.relu(x_add)
        out = self.fc3(x_add)
        return out

# -------------------- Utility Function: Reset Directory -------------------- #
def reset_dir(path):
    """Removes and recreates a directory."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# -------------------- Data Loading (for CIFAR10) -------------------- #
def load_cifar10_with_index(base_dir, batch_size, train=True):
    """Loads CIFAR10 dataset with sample indices."""
    data_path = os.path.join(base_dir, "data", "cifar10_data")
    os.makedirs(data_path, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    
    dataset = IndexedCIFAR10(
        root=data_path,
        train=train,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        worker_init_fn=lambda x: np.random.seed(42 + x),
        collate_fn=lambda batch: (
            torch.tensor([item[0] for item in batch]),  # Index
            torch.stack([item[1] for item in batch]),  # Image
            torch.tensor([item[2] for item in batch])  # Label
        )
    )
    return dataloader, dataset

# -------------------- Denormalization Function (for image saving) -------------------- #
def denormalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    """Converts a normalized tensor to a savable image format."""
    tensor = tensor.clone().cpu()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return transforms.ToPILImage()(tensor)

# -------------------- Normalization Function (for handling backdoor images) -------------------- #
def normalize_image(img, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    """Converts a PIL image to a normalized tensor."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform(img)

# -------------------- Add Backdoor Trigger to CIFAR Image -------------------- #
def add_backdoor_trigger(img_path, output_path):
    """
    Adds a BadNets-style backdoor trigger to a CIFAR10 color image.
    Sets black and white pixels in a 3x3 bottom-right area (label-consistent attack).
    """
    # Open image and ensure RGB format (CIFAR10 is color)
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)  # Shape is (32, 32, 3)
    h, w = img_array.shape[:2]  # CIFAR10 image size is 32x32

    # Copy the original image array for modification
    backdoored = img_array.copy()

    # Define coordinates for black and white pixels (consistent with BadNets)
    black = 0      # Black: all three channels are 0
    white = 255    # White: all three channels are 255
    # Coordinates in the 3x3 bottom-right area
    black_coords = [(h-3, w-3), (h-3, w-2), (h-2, w-3), (h-2, w-1), (h-1, w-2)]
    white_coords = [(h-3, w-1), (h-2, w-2), (h-1, w-3), (h-1, w-1)]

    # Set black pixels (all three channels to 0)
    for y, x in black_coords:
        backdoored[y, x, :] = black  # Modify all R, G, B channels

    # Set white pixels (all three channels to 255)
    for y, x in white_coords:
        backdoored[y, x, :] = white  # Modify all R, G, B channels

    # Save the image with the trigger
    backdoored_img = Image.fromarray(backdoored)
    backdoored_img.save(output_path)
    return backdoored_img

# -------------------- Generate Backdoor Dataset (from existing perturbed data) -------------------- #
def generate_backdoor_dataset(base_dir, train_dataset):
    """
    Adds backdoor triggers to pre-generated PGD perturbed data.
    """
    # Path configuration
    save_root = os.path.join(base_dir, "save2")
    perturbed_dir = os.path.join(save_root, "perturbed_data")
    backdoor_dir = os.path.join(save_root, "backdoor_data")
    index_mapping_path = os.path.join(save_root, "index_mapping_and_labels.csv")
    
    # Create directory for backdoor data
    os.makedirs(backdoor_dir, exist_ok=True)
    
    # Read indices of perturbed data
    poisoned_indices = []
    with open(index_mapping_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            poisoned_indices.append(int(row['original_index']))
    
    # Add backdoor trigger to each perturbed image
    backdoor_images = []
    for idx in tqdm(poisoned_indices, desc="Adding backdoor triggers"):
        # Read the pre-generated perturbed image
        perturbed_img_path = os.path.join(perturbed_dir, f"sample_{idx}.png")
        if not os.path.exists(perturbed_img_path):
            raise FileNotFoundError(f"Perturbed image not found: {perturbed_img_path}")
        
        # Add trigger and save
        backdoor_img_path = os.path.join(backdoor_dir, f"backdoor_sample_{idx}.png")
        backdoored_img = add_backdoor_trigger(perturbed_img_path, backdoor_img_path)
        
        # Convert to a normalized tensor for training
        normalized_img = normalize_image(backdoored_img)
        backdoor_images.append(normalized_img)
    
    # Save index mapping for backdoor data
    backdoor_mapping_path = os.path.join(save_root, "backdoor_mapping.csv")
    with open(backdoor_mapping_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["original_index", "true_label"])
        for idx in poisoned_indices:
            _, _, true_label = train_dataset[idx]
            writer.writerow([idx, true_label])
    
    print(f"✅ Backdoor dataset generated with {len(backdoor_images)} samples.")
    print(f"Backdoor data saved to: {backdoor_dir}")
    print(f"Backdoor index mapping saved to: {backdoor_mapping_path}")
    
    return backdoor_images, poisoned_indices, backdoor_mapping_path

# -------------------- per-sample Gradient Calculation (for CNN) -------------------- #
def per_sample_grads(model, images, labels, criterion):
    """Calculates gradients for each sample (flattened to a vector)."""
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    # Single-sample loss function
    def loss_fn_single(params, buffers, x, y):
        output = functional_call(model, (params, buffers), x.unsqueeze(0))
        return criterion(output, y.unsqueeze(0))

    # Vectorize gradient calculation (vmap)
    grads = vmap(grad(loss_fn_single), (None, None, 0, 0))(params, buffers, images, labels)
    # Flatten and concatenate gradients (for multi-layer CNN parameters)
    flat_grads = torch.cat([g.view(g.size(0), -1) for g in grads.values()], dim=1)
    return flat_grads

# -------------------- Validation Set Gradient Calculation -------------------- #
def compute_val_grads(model, val_images, val_labels, criterion, device):
    """Calculates the gradient of the validation loss w.r.t. model parameters (flattened to a vector)."""
    model.train()  # Ensure training mode (no dropout, etc.)
    val_logits = model(val_images)
    val_loss = criterion(val_logits, val_labels)
    model.zero_grad()
    # Compute gradients
    val_grads = torch.autograd.grad(val_loss, model.parameters(), create_graph=False)
    # Flatten to a vector
    g_val = parameters_to_vector(val_grads).unsqueeze(0)
    return g_val

# -------------------- Training and Shapley Value Computation -------------------- #
def train_and_compute_shapley(
    model, train_loader, test_loader, optimizer, criterion, epochs, device,
    val_images, val_labels, shapley_file, base_dir, model_name="CNN"
):
    print(f"Starting training {model_name} and computing Shapley values...")
    train_losses, test_losses, accuracies = [], [], []
    num_train_samples = len(train_loader.dataset)
    data_shapley = torch.zeros(num_train_samples, device=device)

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")

        for batch_idx, (indices, images, labels) in progress_bar:
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            indices = indices.to(device)

            # 1. Compute per-sample gradients
            sample_grads = per_sample_grads(model, images, labels, criterion)

            # 2. Compute validation set gradient (g_val)
            g_val = compute_val_grads(model, val_images, val_labels, criterion, device)

            # 3. Compute and accumulate Shapley increments for the current batch
            lr = optimizer.param_groups[0]['lr']  # Current learning rate
            batch_phi = -lr * torch.matmul(sample_grads, g_val.T).squeeze(1)
            data_shapley[indices] += batch_phi

            # 4. Standard training step
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Log training metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix({
                "Train Loss": f"{train_loss/(batch_idx+1):.4f}",
                "Accuracy": f"{100*correct/total:.2f}%"
            })

        # Record training loss
        train_losses.append(train_loss / len(train_loader))

        # Test set evaluation
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for indices_test, images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_losses.append(test_loss / len(test_loader))
        accuracies.append(100 * correct / total)
        print(f"Epoch {epoch+1}/{epochs} | Test Loss: {test_losses[-1]:.4f} | Test Acc: {accuracies[-1]:.2f}%")

    # Save Shapley values to CSV
    results_dir = os.path.join(base_dir, "results2")
    os.makedirs(results_dir, exist_ok=True)
    shapley_path = os.path.join(results_dir, shapley_file)
    with open(shapley_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "shapley_value"])
        for i in range(len(data_shapley)):
            val = data_shapley[i].item()
            # Handle potential NaN values
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            writer.writerow([i, val])
    print(f"Shapley values saved to: {shapley_path}")

    return train_losses, test_losses, accuracies, shapley_path

# -------------------- Visualize Training Results -------------------- #
def plot_results(train_losses, test_losses, accuracies, base_dir, title_suffix=""):
    """Plots loss and accuracy curves."""
    plt.figure(figsize=(12, 4))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves {title_suffix}")
    plt.legend()
    
    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label="Test Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy Curve {title_suffix}")
    plt.legend()
    
    # Ensure save directory exists
    plot_dir = os.path.join(base_dir, "results2")
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, f"training_results{title_suffix}.png")
    plt.savefig(save_path)
    print(f"Training plots saved to: {save_path}")
    plt.show()

# -------------------- Main Function -------------------- #
def main():
    # Configuration parameters
    base_dir = "./xie/FNN_Shapley"
    batch_size = 64
    epochs = 100
    lr = 0.01
    set_seed(42)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Load original CIFAR10 data (training and test sets)
    train_loader, train_dataset = load_cifar10_with_index(base_dir, batch_size, train=True)
    test_loader, _ = load_cifar10_with_index(base_dir, batch_size, train=False)
    print(f"✅ Original data loaded. Training samples: {len(train_dataset)}")

    # Step 2: Prepare validation data (for Shapley value calculation)
    val_loader, _ = load_cifar10_with_index(base_dir, batch_size=128, train=False)
    indices_val, val_images, val_labels = next(iter(val_loader))
    val_images, val_labels = val_images.to(device), val_labels.to(device)

    # -------------------- New: Original Dataset Training and Shapley Calculation -------------------- #
    print("\n===== Step A: Train model on original dataset and compute Shapley values =====")
    # Initialize an independent model for the original dataset
    model_original = ModifiedCIFARCNN().to(device)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer_original = optim.SGD(model_original.parameters(), lr=lr, momentum=0.9)
    
    # File path for original Shapley values
    shapley_file_original = "shapley_original.csv"
    # Train and compute original Shapley values
    train_losses_original, test_losses_original, accuracies_original, shapley_path_original = train_and_compute_shapley(
        model_original, train_loader, test_loader, optimizer_original, criterion, epochs, device,
        val_images, val_labels, shapley_file_original, base_dir, model_name="Original-CIFAR-CNN"
    )
    # Visualize original training results
    plot_results(train_losses_original, test_losses_original, accuracies_original, base_dir, title_suffix="(Original Dataset)")

    # -------------------- Original: Backdoor Dataset Processing -------------------- #
    print("\n===== Step B: Generate backdoor dataset =====")
    # Pass train_dataset to generate_backdoor_dataset function
    backdoor_images, poisoned_indices, backdoor_mapping_path = generate_backdoor_dataset(base_dir, train_dataset)

    print("\n===== Step C: Create backdoor dataset loader =====")
    poisoned_dataset = PoisonedCIFAR10Dataset(
        original_dataset=train_dataset,
        poisoned_images=backdoor_images,
        poisoned_indices=poisoned_indices
    )
    backdoor_train_loader = DataLoader(
        poisoned_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=lambda x: np.random.seed(42 + x),
        collate_fn=lambda batch: (
            torch.tensor([item[0] for item in batch]),  # Index
            torch.stack([item[1] for item in batch]),  # Image (could be backdoor image)
            torch.tensor([item[2] for item in batch])  # Label (original label is kept)
        )
    )
    print(f"✅ Backdoor dataset created, {len(poisoned_indices)} samples replaced.")

    # -------------------- Original: Backdoor Dataset Training and Shapley Calculation -------------------- #
    print("\n===== Step D: Train model on backdoor dataset and compute Shapley values =====")
    # Initialize an independent model for the backdoor dataset
    model_backdoor = ModifiedCIFARCNN().to(device)
    optimizer_backdoor = optim.SGD(model_backdoor.parameters(), lr=lr, momentum=0.9)
    
    # File path for backdoor Shapley values
    shapley_file_backdoor = "shapley_backdoor.csv"
    # Train and compute backdoor Shapley values
    train_losses_backdoor, test_losses_backdoor, accuracies_backdoor, shapley_path_backdoor = train_and_compute_shapley(
        model_backdoor, backdoor_train_loader, test_loader, optimizer_backdoor, criterion, epochs, device,
        val_images, val_labels, shapley_file_backdoor, base_dir, model_name="Backdoor-CIFAR-CNN"
    )
    # Visualize backdoor training results
    plot_results(train_losses_backdoor, test_losses_backdoor, accuracies_backdoor, base_dir, title_suffix="(Backdoor Dataset)")

    # -------------------- Modified: Save mapping between backdoor and original samples -------------------- #
    print("\n===== Step E: Save mapping between backdoor and original samples =====")
    # Backdoor-to-original index mapping file (records which original indices were replaced with backdoor samples and their true labels)
    backdoor_original_mapping = os.path.join(base_dir, "results2", "backdoor_original_index_mapping.csv")
    with open(backdoor_original_mapping, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["original_index", "true_label"])
        # Iterate through all backdoor sample indices
        for idx in poisoned_indices:
            # Get the true label of the original sample
            _, _, true_label = train_dataset[idx]
            writer.writerow([idx, true_label])
    print(f"Mapping between backdoor and original samples saved to: {backdoor_original_mapping}")

    print("\n===== All tasks complete =====")
    print(f"Original dataset Shapley values file: {shapley_path_original}")
    print(f"Backdoor dataset Shapley values file: {shapley_path_backdoor}")
    print(f"Backdoor-to-original index mapping file: {backdoor_original_mapping}")
    print(f"Modified backdoor sample index mapping file: {backdoor_mapping_path}")

if __name__ == "__main__":
    main()