import os
import sys
import torch
import yaml
import argparse
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_utils import ComparisonDataGenerator, PicoDataset
from src.engine import evaluate_model
from src.pico.randaugment import RandomAugment
from src.pico.model import PiCOModel
from src.pico.resnet import SupConResNet
from src.pico.utils_loss import PartialLoss, SupConLoss


def train_pico_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device):
    model.train()
    total_loss = 0
    start_upd_prot = epoch >= pico_args['prot_start']
    
    progress_bar = tqdm(loader, desc=f"PiCO Epoch {epoch + 1}/{pico_args['epochs']}")
    for (images_w, images_s, partial_Y, true_labels, index) in progress_bar:
        images_w, images_s, partial_Y, index = images_w.to(device), images_s.to(device), partial_Y.to(device), index.to(device)
        
        cls_out, features, pseudo_target_cont, score_prot = model(images_w, images_s, partial_Y, pico_args)
        batch_size = cls_out.shape[0]

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=partial_Y)
        
        mask = torch.eq(pseudo_target_cont[:batch_size].unsqueeze(1), pseudo_target_cont.unsqueeze(0)).float() if start_upd_prot else None

        loss_cls = loss_fn(cls_out, index)
        loss_cont = loss_cont_fn(features=features, mask=mask, batch_size=batch_size)
        loss = loss_cls + pico_args['loss_weight'] * loss_cont

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
    return total_loss / len(loader)

parser = argparse.ArgumentParser(description='Run PiCO experiment.')
parser.add_argument('type', choices=['constant', 'variable'], help='Type of label generation.')
parser.add_argument('value', type=float, help='Value for k or q.')
args = parser.parse_args()

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
train_config = config['training']
pico_config = config['pico']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Data Generation ---
cifar10_train_raw = CIFAR10(root='./data', train=True, download=True)
generator = ComparisonDataGenerator(cifar10_train_raw)

if args.type == 'constant':
    k = int(args.value)
    print(f"--- Generating PL dataset for constant k = {k} ---")
    pl_dataset_raw = generator.generate_pl_dataset(k=k, num_classes=train_config['num_classes'])
else: # 'variable'
    q = args.value
    print(f"--- Generating PL dataset for variable q = {q} ---")
    pl_dataset_raw, cl_dataset_raw = generator.generate_variable_pl_cl_datasets(q=q, num_classes=train_config['num_classes'])

# --- PiCO Training ---
print("\nTraining PiCO (PL)")

# 1. Setup PiCO-specific arguments from your config
pico_args = {
    'num_class': train_config['num_classes'],
    'epochs': train_config['epochs'],
    'low_dim': pico_config['low_dim'],
    'moco_queue': pico_config['moco_queue'],
    'moco_m': pico_config['moco_m'],
    'proto_m': pico_config['proto_m'],
    'prot_start': pico_config['prot_start'],
    'loss_weight': pico_config['loss_weight'],
    'conf_ema_range': pico_config['conf_ema_range']
}

# 2. Create the PiCO model
pico_model = PiCOModel(pico_args).to(DEVICE)

# 3. Create the PiCO dataset and loader
pico_train_dataset = PicoDataset(pl_dataset_raw, generator.original_targets)
pico_loader = DataLoader(pico_train_dataset, batch_size=train_config['batch_size'], shuffle=True, drop_last=True)

# 4. Create the PiCO loss functions
initial_confidence = torch.ones(len(pico_train_dataset), pico_args['num_class']) / pico_args['num_class']
pico_cls_loss = PartialLoss(initial_confidence)
pico_cont_loss = SupConLoss()

# 5. Create the PiCO optimizer
pico_optimizer = optim.SGD(pico_model.parameters(), lr=train_config['learning_rate'], momentum=0.9, weight_decay=1e-4)

# 6. Setup the test loader
cifar10_test_raw = CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616])
]))
test_loader = DataLoader(cifar10_test_raw, batch_size=train_config['batch_size'], shuffle=False)

# 7. Run the training loop
pico_accuracies = []
for epoch in range(train_config['epochs']):
    pico_cls_loss.set_conf_ema_m(epoch, pico_args)
    avg_loss = train_pico_epoch(pico_args, pico_model, pico_loader, pico_cls_loss, pico_cont_loss, pico_optimizer, epoch, DEVICE)
    current_accuracy = evaluate_model(pico_model, test_loader, DEVICE)
    print(f"Epoch [{epoch+1}/{train_config['epochs']}], Loss: {avg_loss:.4f}, Test Accuracy: {current_accuracy:.2f}%")
    pico_accuracies.append(current_accuracy)
    
# --- Final Results ---
print("\n--- Final Results ---")
best_pico = max(pico_accuracies) if pico_accuracies else 0
print(f"Best Accuracy (PiCO): {best_pico:.2f}%")

# --- Plotting ---
epochs_range = range(1, train_config['epochs'] + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, pico_accuracies, 's-', label='PiCO Test Accuracy')

if args.type == 'constant':
    plt.title(f'Test Accuracy vs. Epochs (Constant k={int(args.value)})')
else:
    plt.title(f'Test Accuracy vs. Epochs (Variable q={args.value})')

plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

print("Done.")