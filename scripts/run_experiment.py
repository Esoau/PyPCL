import os
import sys
import torch
import yaml
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import all our custom modules from src
from src.data_utils import ComparisonDataGenerator, WeaklySupervisedDataset
from src.models import create_model
from src.losses import proden, LogURE
from src.engine import train_algorithm

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Generate weak labels and train models.')
parser.add_argument('type', choices=['constant', 'variable'], help='Type of label generation.')
parser.add_argument('value', type=float, help='Value for k (if type=constant) or q (if type=variable).')
args = parser.parse_args()

# --- Configuration and Setup ---
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_config = config['data_generation']
train_config = config['training']

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

# --- Data Generation ---
print("Loading original CIFAR-10 dataset...")
cifar10_train_raw = CIFAR10(root=data_config['cifar_path'], train=True, download=True)
generator = ComparisonDataGenerator(cifar10_train_raw)
C = train_config['num_classes']

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

# --- Dataset and DataLoader Preparation ---
pl_dataset = WeaklySupervisedDataset(pl_dataset_raw.data, pl_dataset_raw.targets, transform=train_transform)
cl_dataset = WeaklySupervisedDataset(cl_dataset_raw.data, cl_dataset_raw.targets, transform=train_transform)

pl_loader = DataLoader(pl_dataset, batch_size=train_config['batch_size'], shuffle=True)
cl_loader = DataLoader(cl_dataset, batch_size=train_config['batch_size'], shuffle=True)

# Create test loader
cifar10_test_raw = CIFAR10(root=data_config['cifar_path'], train=False, download=True)
test_dataset = WeaklySupervisedDataset(cifar10_test_raw.data, cifar10_test_raw.targets, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False)

# --- Train PRODEN ---
print("\n--- Training PRODEN (PL) ---")
proden_model = create_model(train_config['num_classes'])
proden_loss = proden()
proden_optimizer = optim.Adam(proden_model.parameters(), lr=train_config['learning_rate'])
best_proden = train_algorithm(proden_model, pl_loader, test_loader, proden_loss, proden_optimizer, train_config['epochs'], DEVICE)

# --- Train LogURE ---
print("\n--- Training LogURE (CL) ---")
logure_model = create_model(train_config['num_classes'])
logure_loss = LogURE(num_classes=train_config['num_classes'])
logure_optimizer = optim.Adam(logure_model.parameters(), lr=train_config['learning_rate'])
best_logure = train_algorithm(logure_model, cl_loader, test_loader, logure_loss, logure_optimizer, train_config['epochs'], DEVICE)

# --- Final Results ---
print("\n--- Final Results ---")
print(f"Best Accuracy (PRODEN): {best_proden:.2f}%")
print(f"Best Accuracy (LogURE): {best_logure:.2f}%")
print("Done.")
