import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.func import functional_call, grad, vmap
from torch.nn.utils import parameters_to_vector

# ====================== 🔒 Set global random seed for reproducibility ====================== #
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ============================================================================================ #

# -------------------- Global Configuration --------------------
seed = 42
base_dir = "./xie/FNN_Shapley"
original_shapley_file = "shapley_original.csv"
poisoned_shapley_file = "shapley_poisoned.csv"

# -------------------- Dataset with Index --------------------
class IndexedDataset(datasets.MNIST):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return index, img, target

# -------------------- Custom dataset with index and label (supports image replacement) --------------------
class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, poisoned_images, poisoned_labels, poisoned_indices):
        self.original_dataset = original_dataset
        self.poisoned_images = poisoned_images
        self.poisoned_labels = poisoned_labels
        self.poisoned_indices = poisoned_indices  # Keep as a list, not a set

        # Create an index mapping dictionary for quick lookup of poisoned images in the list
        self.index_map = {idx: i for i, idx in enumerate(poisoned_indices)}
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        if index in self.poisoned_indices:
            # Find the position of the poisoned image in the list via the mapping dictionary
            poisoned_idx = self.index_map[index]
            return index, self.poisoned_images[poisoned_idx], self.poisoned_labels[poisoned_idx]
        else:
            # Return original image
            return self.original_dataset[index]

# -------------------- Utility: Clear & Create Folder --------------------
def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# -------------------- Sample random indices --------------------
def get_random_indices_by_ratio(dataset, ratio=0.1):
    total = len(dataset)
    count = int(total * ratio)
    indices = np.random.choice(total, size=count, replace=False)
    return sorted(indices.tolist())

# -------------------- Save sampled images and CSV --------------------
def save_images_and_indices_csv(dataset, indices, save_dir, csv_path, prefix="img"):
    reset_dir(save_dir)
    records = []

    for idx in indices:
        index, img, label = dataset[idx]
        img = img * 0.3081 + 0.1307  # Inverse normalization
        img = torch.clamp(img, 0, 1)
        fname = f"{prefix}_{index}_label_{label}.png"
        fpath = os.path.join(save_dir, fname)
        save_image(img.unsqueeze(0), fpath)
        records.append((index, label, fname))

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'label', 'filename'])
        writer.writerows(records)
    return records, indices

# -------------------- Add BadNets-style backdoor --------------------
def add_pattern_backdoor(image_path, output_path):
    img = Image.open(image_path).convert('L')  # Ensure grayscale image
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    backdoored = img_array.copy()

    black = 0
    white = 255

    black_coords = [(h-3, w-3), (h-3, w-2), (h-2, w-3), (h-2, w-1), (h-1, w-2)]
    white_coords = [(h-3, w-1), (h-2, w-2), (h-1, w-3), (h-1, w-1)]

    for y, x in black_coords:
        backdoored[y, x] = black
    for y, x in white_coords:
        backdoored[y, x] = white

    Image.fromarray(backdoored).save(output_path)

# -------------------- Generate backdoored images and modify labels --------------------
def generate_poisoned_data(original_dir, backdoor_dir, indices, original_dataset):
    reset_dir(backdoor_dir)
    poisoned_images = []
    poisoned_labels = []
    poisoned_indices = []

    for idx in tqdm(indices, desc="🔧 Generating poisoned data"):
        # Read original image
        _, img, label = original_dataset[idx]
        img = img * 0.3081 + 0.1307  # Inverse normalization
        img = torch.clamp(img, 0, 1)
        
        # Save original image (for comparison)
        original_fname = f"original_{idx}_label_{label}.png"
        original_fpath = os.path.join(backdoor_dir, original_fname)
        save_image(img.unsqueeze(0), original_fpath)
        
        # Add backdoor
        backdoor_fname = f"backdoor_{idx}_label_{label}.png"
        backdoor_fpath = os.path.join(backdoor_dir, backdoor_fname)
        add_pattern_backdoor(original_fpath, backdoor_fpath)
        
        # Modify label to original label + 1 mod 10
        poisoned_label = (label + 1) % 10
        poisoned_labels.append(poisoned_label)
        poisoned_indices.append(idx)
        
        # Read backdoored image and convert to tensor
        backdoored_img = Image.open(backdoor_fpath).convert('L')
        backdoored_tensor = transforms.ToTensor()(backdoored_img)
        # Normalize
        backdoored_tensor = transforms.Normalize((0.1307,), (0.3081,))(backdoored_tensor)
        poisoned_images.append(backdoored_tensor)

    print(f"✅ Poisoned data generated: {len(poisoned_images)} images, labels modified to (original label + 1) mod 10")
    return poisoned_images, poisoned_labels, poisoned_indices

# -------------------- Small model definition (two hidden layers) --------------------
class SmallFNN(nn.Module):
    def __init__(self):
        super(SmallFNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# -------- per-sample gradient computation -------- #
def per_sample_grads(model, images, labels, criterion):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def loss_fn_single(params, buffers, x, y):
        output = functional_call(model, (params, buffers), x.unsqueeze(0))
        return criterion(output, y.unsqueeze(0))

    grads = vmap(grad(loss_fn_single), (None, None, 0, 0))(params, buffers, images, labels)
    flat_grads = torch.cat([g.view(g.size(0), -1) for g in grads.values()], dim=1)
    return flat_grads

# -------- Validation set gradient -------- #
def compute_val_grads(model, val_images, val_labels, criterion, device):
    model.train()
    val_logits = model(val_images)
    val_loss = criterion(val_logits, val_labels)
    model.zero_grad()
    val_grads = torch.autograd.grad(val_loss, model.parameters(), create_graph=False)
    g_val = parameters_to_vector(val_grads).unsqueeze(0)
    return g_val

# -------------------- Training, evaluation, and Shapley computation --------------------
def train_evaluate_and_compute_shapley(
    model, train_loader, test_loader, optimizer, criterion, epochs, device, 
    val_images, val_labels, shapley_file, model_name="Model"
):
    print(f"Starting training {model_name} and computing Shapley values...")
    train_losses, test_losses, accuracies = [], [], []
    num_train_samples = len(train_loader.dataset)
    data_shapley = torch.zeros(num_train_samples, device=device)

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch_idx, (indices, images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            indices = indices.to(device)

            grads = per_sample_grads(model, images, labels, criterion)
            val_grad = compute_val_grads(model, val_images, val_labels, criterion, device)
            lr = optimizer.param_groups[0]['lr']
            batch_phi = -lr * (grads @ val_grad.T)
            data_shapley[indices] += batch_phi.squeeze()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_description(
                f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/(batch_idx+1):.4f}"
            )

        train_losses.append(train_loss / len(train_loader))

        # Test
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_losses.append(test_loss / len(test_loader))
        accuracies.append(100 * correct / total)
        print(f"Epoch {epoch+1}, Test Acc: {accuracies[-1]:.2f}%")

    # Save Shapley values
    os.makedirs(os.path.join(base_dir, 'results1'), exist_ok=True)
    with open(os.path.join(base_dir, 'results1', shapley_file), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "shapley_value"])
        for i in range(len(data_shapley)):
            writer.writerow([i, data_shapley[i].item()])
    print(f"Shapley values of {model_name} saved to ./xie/FNN_Shapley/results1/{shapley_file}")
    
    return train_losses, test_losses, accuracies

# -------------------- Visualization -------- #
def plot_results(train_losses, test_losses, accuracies, title_suffix=""):
    # Configure English font
    #plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False  # Ensure minus sign displays correctly
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss over Epochs{title_suffix}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Test Accuracy over Epochs{title_suffix}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'results1', f'training_results{title_suffix}.png'))
    plt.show()
    
# -------------------- Compute attack success rate (4 out of 6 cases, code implementation is two types) --------------------
def compute_attack_success_rate(base_dir, original_shapley_file, poisoned_shapley_file, mapping_file):
    original_shapley_path = os.path.join(base_dir, 'results1', original_shapley_file)
    poisoned_shapley_path = os.path.join(base_dir, 'results1', poisoned_shapley_file)
    mapping_path = os.path.join(base_dir, 'results1', mapping_file)

    # Read original Shapley values
    original_shapley = {}
    with open(original_shapley_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_shapley[int(row['index'])] = float(row['shapley_value'])
    
    # Read poisoned Shapley values
    poisoned_shapley = {}
    with open(poisoned_shapley_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            poisoned_shapley[int(row['index'])] = float(row['shapley_value'])
    
    # Read poisoned index mapping
    poisoned_indices = []
    with open(mapping_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            poisoned_indices.append(int(row['original_index']))

    # Initialize statistics variables
    success_count = 0
    total_poisoned = 0
    
    # Variables related to Forged Gain success rate
    valid_orig_sum = 0.0
    valid_pois_sum = 0.0
    valid_pairs_count = 0
    
    # Variables related to Loss Rise Rate (modified from original "decrease loss rate")
    # Two cases that cause a rise in Loss:
    # 1. Positive-contribution samples are misjudged as negative-contribution (server discards good samples)
    # 2. Negative-contribution samples are misjudged as positive-contribution (server keeps bad samples)
    good_samples_discarded = 0  # Case 1: orig > 0 and pois < 0
    bad_samples_retained = 0    # Case 2: orig < 0 and pois > 0

    # Iterate through all poisoned sample indices
    for idx in poisoned_indices:
        if idx in original_shapley and idx in poisoned_shapley:
            orig = original_shapley[idx]
            pois = poisoned_shapley[idx]
            total_poisoned += 1
            
            # Compute attack success rate (original logic)
            if pois < orig or (orig < 0 and pois > 0):
                success_count += 1
            
            # Compute Forged Gain success rate (excluding the case where orig < 0 and pois > 0)
            if not (orig < 0 and pois > 0):
                valid_orig_sum += orig
                valid_pois_sum += pois
                valid_pairs_count += 1
            
            # Compute Loss Rise Rate (counting two types of misjudgment)
            if orig > 0 and pois < 0:
                good_samples_discarded += 1  # Positive-contribution sample misjudged as negative-contribution (discards a good sample)
            elif orig < 0 and pois > 0:
                bad_samples_retained += 1    # Negative-contribution sample misjudged as positive-contribution (retains a bad sample)

    # Compute and print attack success rate
    if total_poisoned > 0:
        success_rate = success_count / total_poisoned
        print(f"🎯 Attack Success Rate: {success_rate:.2%} ({success_count}/{total_poisoned})")
    else:
        print("⚠️ No poisoned data indices found for evaluating attack success rate.")
        success_rate = 0

    # Compute and print Forged Gain success rate
    if valid_pairs_count > 0:
        forged_gain_success = 1.0 if valid_orig_sum > valid_pois_sum else 0.0
        print(f"💰 Forged Gain Success Rate: {forged_gain_success:.2%} (Original sum: {valid_orig_sum:.4f}, Poisoned sum: {valid_pois_sum:.4f})")
    else:
        print("⚠️ No valid pairs found for evaluating forged gain success rate.")
        forged_gain_success = 0

    # Compute and print Loss Rise Rate (core modification: adjusted name and description)
    loss_rise_count = good_samples_discarded + bad_samples_retained
    loss_rise_rate = 1.0 if loss_rise_count > 0 else 0.0
    print(f"📈 Loss Rise Rate: {loss_rise_rate:.2%} (Good samples discarded: {good_samples_discarded}, Bad samples retained: {bad_samples_retained})")
    
    return {
        'attack_success_rate': success_rate,
        'forged_gain_success_rate': forged_gain_success,
        'loss_rise_rate': loss_rise_rate  # Return key name synchronized with modification
    }

# -------------------- Main function entry point -------- #
def main():
    # Set random seed
    set_seed(seed)
    
    # 1. Configure paths
    ratio = 0.01  # Poisoned data ratio
    raw_dir = os.path.join(base_dir, f'save1/random_{int(ratio*100)}pct_images')
    backdoor_dir = os.path.join(base_dir, f'save1/random_{int(ratio*100)}pct_backdoored_images')
    csv_path = os.path.join(base_dir, f'save1/random_{int(ratio*100)}pct_indices.csv')
    mapping_file = 'poisoned_mapping.csv'
    mapping_path_full = os.path.join(base_dir, 'results1', mapping_file)
    
    os.makedirs(os.path.join(base_dir, 'save1'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'results1'), exist_ok=True)

    # 2. Load original dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    original_dataset = IndexedDataset(root=os.path.join(base_dir, 'data'), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=os.path.join(base_dir, 'data'), train=False, download=True, transform=transform)
    
    # 3. Randomly sample
    indices = get_random_indices_by_ratio(original_dataset, ratio)
    print(f"🎲 Randomly sampled {len(indices)} indices for poisoning")
    
    # 4. Save original sampled images and record indices
    records, poisoned_indices = save_images_and_indices_csv(original_dataset, indices, raw_dir, csv_path, prefix="random")
    
    # 5. Generate poisoned data (add backdoor + modify labels)
    poisoned_images, poisoned_labels, _ = generate_poisoned_data(raw_dir, backdoor_dir, indices, original_dataset)
    
    # 6. Save poisoned data index mapping
    with open(mapping_path_full, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["original_index", "poisoned_label"])
        for i, idx in enumerate(poisoned_indices):
            writer.writerow([idx, poisoned_labels[i]])
    print(f"Poisoned data index mapping saved to {mapping_path_full}")
    
    # 7. Create data loader - original training set
    original_train_loader = DataLoader(
        original_dataset, 
        batch_size=64, 
        shuffle=True, 
        drop_last=True,
        collate_fn=lambda batch: (
            torch.tensor([item[0] for item in batch]),
            torch.stack([item[1] for item in batch]),
            torch.tensor([item[2] for item in batch])
        )
    )
    
    # 8. Create poisoned training set
    poisoned_dataset = PoisonedDataset(original_dataset, poisoned_images, poisoned_labels, poisoned_indices)
    poisoned_train_loader = DataLoader(
        poisoned_dataset, 
        batch_size=64, 
        shuffle=True, 
        drop_last=True,
        collate_fn=lambda batch: (
            torch.tensor([item[0] for item in batch]),
            torch.stack([item[1] for item in batch]),
            torch.tensor([item[2] for item in batch])
        )
    )
    
    # 9. Create test data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.tensor([item[1] for item in batch])
        )
    )
    
    # 10. Prepare validation set
    val_images = torch.stack([img for img, _ in test_dataset]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    val_labels = torch.tensor([lab for _, lab in test_dataset]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # 11. Train models and compute Shapley values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train original model
    original_model = SmallFNN().to(device)
    original_criterion = nn.CrossEntropyLoss(reduction='sum')
    original_optimizer = optim.SGD(original_model.parameters(), lr=0.01)
    epochs = 100
    original_train_losses, original_test_losses, original_accuracies = train_evaluate_and_compute_shapley(
        original_model, original_train_loader, test_loader, original_optimizer, original_criterion, 
        epochs, device, val_images, val_labels, original_shapley_file, "Original Model"
    )
    
    # Train poisoned model
    poisoned_model = SmallFNN().to(device)
    poisoned_criterion = nn.CrossEntropyLoss(reduction='sum')
    poisoned_optimizer = optim.SGD(poisoned_model.parameters(), lr=0.01)
    poisoned_train_losses, poisoned_test_losses, poisoned_accuracies = train_evaluate_and_compute_shapley(
        poisoned_model, poisoned_train_loader, test_loader, poisoned_optimizer, poisoned_criterion, 
        epochs, device, val_images, val_labels, poisoned_shapley_file, "Poisoned Model"
    )
    
    # 12. Visualize results
    plot_results(original_train_losses, original_test_losses, original_accuracies, " (Original Model)")
    plot_results(poisoned_train_losses, poisoned_test_losses, poisoned_accuracies, " (Poisoned Model)")
    
    # 13. Compute attack success rate
    compute_attack_success_rate(base_dir, original_shapley_file, poisoned_shapley_file, mapping_file)

if __name__ == "__main__":
    main()