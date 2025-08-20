import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import csv
from tqdm import tqdm


# --------------------- 1. Model Definition --------------------- #
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


# --------------------- 2. Indexed CIFAR10 Dataset --------------------- #
class IndexedCIFAR10(torchvision.datasets.CIFAR10):
    """Returns (index, image tensor, label) with non-augmented transform"""
    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return index, data, target


# --------------------- 3. Denormalization Function --------------------- #
def denormalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    """Convert normalized tensor to PIL image (0-255 pixel values)"""
    tensor = tensor.clone().cpu()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return transforms.ToPILImage()(tensor)


# --------------------- 4. L2-Constrained PGD Attack Implementation --------------------- #
def pgd_l2_attack(model, images, labels, eps=300, alpha=4.5, steps=100, device="cuda"):
    """
    Generate L2-norm constrained PGD adversarial examples
    :param model: Pre-trained model
    :param images: Original images (normalized tensor, shape: [B, 3, 32, 32])
    :param labels: True labels (shape: [B])
    :param eps: L2 perturbation bound
    :param alpha: Step size (1.5*eps/100)
    :param steps: Number of iterations
    :param device: Computing device
    :return: Adversarial examples (normalized tensor)
    """
    adv_images = images.clone().detach().to(device)
    adv_images.requires_grad = True

    for _ in range(steps):
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        if adv_images.grad is None:
            raise RuntimeError("Gradient computation failed, adv_images.grad is None")
        
        grad = adv_images.grad
        grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True)
        grad_norm = grad_norm.view(-1, 1, 1, 1)
        grad = grad / (grad_norm + 1e-10)

        with torch.no_grad():
            adv_images += alpha * grad
            
            delta = adv_images - images.to(device)
            delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1, keepdim=True)
            delta_norm = delta_norm.view(-1, 1, 1, 1)
            scale = torch.min(eps / (delta_norm + 1e-10), torch.ones_like(delta_norm))
            adv_images = images.to(device) + delta * scale
        
        adv_images.requires_grad = True

    with torch.no_grad():
        delta = adv_images - images.to(device)
        delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)
        scale = torch.min(eps / (delta_norm + 1e-10), torch.ones_like(delta_norm))
        adv_images = (images.to(device) + delta * scale).detach()

    return adv_images


# --------------------- 5. Main Function: Data Selection, Perturbation Generation and Saving --------------------- #
def main():
    # Configuration parameters
    base_dir = "./xie/FNN_Shapley"
    p = 0.1  # Select 10% of training data
    seed = 42  # Random seed (for reproducibility)
    eps = 600  # 300 L2 perturbation bound
    steps = 100 #100 # Number of PGD iterations
    alpha = 6  #eps / steps 
    batch_size = 64  # Batch size

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------- Load pre-trained model --------------------- #
    model = ModifiedCIFARCNN().to(device)
    model_path = os.path.join(base_dir, "cifar10_cnn_model", "pretrained_cifar10_cnn.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model does not exist. Please run training code first: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Successfully loaded pre-trained model: {model_path}")

    # --------------------- Load original training data (non-augmented) --------------------- #
    # Data preprocessing: only normalization, no random augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    # Load indexed training set
    data_path = os.path.join(base_dir, "data", "cifar10_data")
    trainset = IndexedCIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False,  # Do not shuffle to keep index stable
        num_workers=2,
        worker_init_fn=lambda x: np.random.seed(seed + x)  # Fix seed for sub-processes
    )
    print(f"✅ Finished loading training data, total {len(trainset)} samples")

    # --------------------- Randomly select 10% of training samples --------------------- #
    total_samples = len(trainset)
    selected_count = int(total_samples * p)
    # Generate non-repetitive random indices (reproducible with seed=42)
    selected_indices = set(random.sample(range(total_samples), selected_count))
    print(f"✅ Randomly selected {selected_count} samples (10% of training set)")

    # --------------------- Create save directories --------------------- #
    save_root = os.path.join(base_dir, "save2")
    original_img_dir = os.path.join(save_root, "original_data")  # Save randomly selected original samples
    perturbed_img_dir = os.path.join(save_root, "perturbed_data")  # Save corresponding PGD perturbed samples
    os.makedirs(original_img_dir, exist_ok=True)
    os.makedirs(perturbed_img_dir, exist_ok=True)
    print(f"✅ Save directory created: {save_root}")

    # --------------------- Prepare CSV file (record index mapping and labels) --------------------- #
    csv_path = os.path.join(save_root, "index_mapping_and_labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        # Write header
        csv_writer.writerow(["original_index", "original_image_path", "perturbed_image_path", "label", "class_name"])
        
        # CIFAR-10 class names mapping
        cifar10_classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        # --------------------- Batch process samples and generate perturbations --------------------- #
        print("Starting to generate adversarial examples and save...")
        for batch in tqdm(trainloader, desc="Processing progress"):
            indices, images, labels = batch  # (index, image tensor, label)
            indices = indices.numpy()

            # Filter selected samples in current batch
            mask = [idx in selected_indices for idx in indices]
            if not any(mask):
                continue  # Skip batches with no selected samples

            # Extract selected samples
            selected_indices_batch = indices[mask]
            selected_images = images[mask].to(device)  # Original images (normalized tensors)
            selected_labels = labels[mask].to(device)  # Labels

            # Generate PGD adversarial examples
            perturbed_images = pgd_l2_attack(
                model, selected_images, selected_labels,
                eps=eps, alpha=alpha, steps=steps, device=device
            )

            # Save original images, perturbed images, and record in CSV
            for idx, orig_tensor, pert_tensor, label in zip(
                selected_indices_batch, selected_images.cpu(), perturbed_images.cpu(), labels[mask].numpy()
            ):
                # Save original image
                orig_img = denormalize(orig_tensor)
                orig_img_path = os.path.join(original_img_dir, f"sample_{idx}.png")
                orig_img.save(orig_img_path)

                # Save perturbed image
                pert_img = denormalize(pert_tensor)
                pert_img_path = os.path.join(perturbed_img_dir, f"sample_{idx}.png")
                pert_img.save(pert_img_path)

                # Get class name
                class_name = cifar10_classes[label]
                
                # Write to CSV
                csv_writer.writerow([idx, orig_img_path, pert_img_path, label, class_name])

    print(f"✅ All operations completed! Results are as follows:")
    print(f"  - Original images saved to: {original_img_dir}")
    print(f"  - Perturbed images saved to: {perturbed_img_dir}")
    print(f"  - Index mapping and labels recorded in: {csv_path}")


if __name__ == "__main__":
    main()