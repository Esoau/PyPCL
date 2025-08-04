import os
import sys
import torch
import yaml
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
from torchvision import transforms

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import project modules
from src.data_utils import ComparisonDataGenerator, WeaklySupervisedDataset, PicoDataset
from src.models import create_model
from src.proden_loss import proden
from src.mcl_log_loss import MCL_Log
from src.engine import train_algorithm, evaluate_model
from src.pico.model import PiCOModel, train_pico_epoch
from src.pico.utils_loss import PartialLoss, SupConLoss
from src.collate import collate_fn, pico_collate_fn

# Argument parsing
parser = argparse.ArgumentParser(description='Generate weak labels and train models.')
parser.add_argument('type', choices=['constant', 'variable'], help='Type of label generation.')
parser.add_argument('value', type=float, help='Value for k (if type=constant) or q (if type=variable).')
args = parser.parse_args()

# Configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_config = config['data_generation']
train_config = config['training']
pico_config = config['pico']


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616])
])

# Get CIFAR-10 dataset
print("Loading original CIFAR-10 dataset...")
cifar10_train_raw = CIFAR10(root=data_config['cifar_path'], train=True, download=True)
generator = ComparisonDataGenerator(cifar10_train_raw)
C = train_config['num_classes']

# Create CL and PL datasets based on type
if args.type == 'constant':
    k = int(args.value)
    m = C - k
    print(f"--- Generating datasets for constant label setting ---")
    print(f"PL setting: k = {k} | CL setting: m = {m}")
    pl_dataset_raw = generator.generate_pl_dataset(k=k)
    cl_dataset_raw = generator.generate_cl_dataset(m=m)
elif args.type == 'variable':
    q = args.value
    print(f"--- Generating datasets for variable label setting ---")
    print(f"PL/CL setting: q = {q}")
    pl_dataset_raw, cl_dataset_raw = generator.generate_variable_pl_cl_datasets(q=q, num_classes=C)
else:
    raise ValueError("Invalid type specified. Choose 'constant' or 'variable'.")

# Dataloading
pl_dataset = WeaklySupervisedDataset(pl_dataset_raw.data, pl_dataset_raw.targets, transform=train_transform)
cl_dataset = WeaklySupervisedDataset(cl_dataset_raw.data, cl_dataset_raw.targets, transform=train_transform)

pl_loader = DataLoader(pl_dataset, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn)
cl_loader = DataLoader(cl_dataset, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn)

# Create test loader
cifar10_test_raw = CIFAR10(root=data_config['cifar_path'], train=False, download=True, transform=test_transform)
test_loader = DataLoader(cifar10_test_raw, batch_size=train_config['batch_size'], shuffle=False)

# Train PRODEN
print("\nTraining PRODEN (PL)")
proden_model = create_model(train_config['num_classes'])
proden_loss = proden()
proden_optimizer = optim.Adam(proden_model.parameters(), lr=train_config['learning_rate'])
proden_accuracies = train_algorithm(proden_model, pl_loader, test_loader, proden_loss, proden_optimizer, train_config['epochs'], DEVICE)

# Train MCL-LOG
print("\nTraining MCL-LOG (CL)")
mcl_log_model = create_model(train_config['num_classes'])
mcl_log_loss = MCL_Log(num_classes=train_config['num_classes'])
mcl_log_optimizer = optim.Adam(mcl_log_model.parameters(), lr=train_config['learning_rate'])
mcl_log_accuracies = train_algorithm(mcl_log_model, cl_loader, test_loader, mcl_log_loss, mcl_log_optimizer, train_config['epochs'], DEVICE)

# Train PiCO
print("\nTraining PiCO (PL)")
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
pico_model = PiCOModel(pico_args).to(DEVICE)
pico_train_dataset = PicoDataset(pl_dataset_raw, generator.original_targets)
pico_loader = DataLoader(
    pico_train_dataset,
    batch_size=train_config['batch_size'],
    shuffle=True,
    drop_last=True,
    collate_fn=pico_collate_fn
)

initial_confidence = torch.ones(len(pico_train_dataset), pico_args['num_class']) / pico_args['num_class']
pico_cls_loss = PartialLoss(initial_confidence)
pico_cont_loss = SupConLoss()

pico_optimizer = optim.SGD(pico_model.parameters(), lr=train_config['learning_rate'], momentum=0.9, weight_decay=1e-4)

pico_accuracies = []
for epoch in range(train_config['epochs']):
    pico_cls_loss.set_conf_ema_m(epoch, pico_args)
    avg_loss = train_pico_epoch(pico_args, pico_model, pico_loader, pico_cls_loss, pico_cont_loss, pico_optimizer, epoch, DEVICE)
    current_accuracy = evaluate_model(pico_model, test_loader, DEVICE)
    print(f"Epoch [{epoch+1}/{train_config['epochs']}], Loss: {avg_loss:.4f}, Test Accuracy: {current_accuracy:.2f}%")
    pico_accuracies.append(current_accuracy)


# Final results
print("\n--- Final Results ---")
best_proden = max(proden_accuracies) if proden_accuracies else 0
best_mcl_log = max(mcl_log_accuracies) if mcl_log_accuracies else 0
best_pico = max(pico_accuracies) if pico_accuracies else 0

print(f"Best Accuracy (PRODEN): {best_proden:.2f}%")
print(f"Best Accuracy (MCL-LOG): {best_mcl_log:.2f}%")
print(f"Best Accuracy (PiCO): {best_pico:.2f}%")


# Accuracy plot
epochs = range(1, train_config['epochs'] + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, proden_accuracies, 'o-', label='PRODEN Test Accuracy')
plt.plot(epochs, mcl_log_accuracies, 'x-', label='MCL-LOG Test Accuracy')
plt.plot(epochs, pico_accuracies, 's-', label='PiCO Test Accuracy')

if args.type == 'constant':
    plt.title(f'Test Accuracy vs. Epochs (Constant k={int(args.value)})')
else:
    plt.title(f'Test Accuracy vs. Epochs (Variable q={args.value})')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()