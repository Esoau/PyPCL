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

# return path to project root for import
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# import project modules
from src.data_utils import ComparisonDataGenerator, WeaklySupervisedDataset
from src.models import create_model
from src.losses import proden, MCL_Log
from src.engine import train_algorithm

# collate function for dataloading
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    
    # get max length of labels
    max_len = max(len(label) for label in labels)
    
    # pad label tensors to max length
    padded_labels = torch.full((len(labels), max_len), -1, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
        
    return images, padded_labels

# argument parsing
parser = argparse.ArgumentParser(description='Generate weak labels and train models.')
parser.add_argument('type', choices=['constant', 'variable'], help='Type of label generation.')
parser.add_argument('value', type=float, help='Value for k (if type=constant) or q (if type=variable).')
args = parser.parse_args()

# configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_config = config['data_generation']
train_config = config['training']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# define transforms
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

# get cifar10 dataset
print("Loading original CIFAR-10 dataset...")
cifar10_train_raw = CIFAR10(root=data_config['cifar_path'], train=True, download=True)
generator = ComparisonDataGenerator(cifar10_train_raw)
C = train_config['num_classes']

# create CL and PL datasets based on type
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

# dataloading
pl_dataset = WeaklySupervisedDataset(pl_dataset_raw.data, pl_dataset_raw.targets, transform=train_transform)
cl_dataset = WeaklySupervisedDataset(cl_dataset_raw.data, cl_dataset_raw.targets, transform=train_transform)

pl_loader = DataLoader(pl_dataset, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn)
cl_loader = DataLoader(cl_dataset, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn)

# create test loader
cifar10_test_raw = CIFAR10(root=data_config['cifar_path'], train=False, download=True)
test_dataset = WeaklySupervisedDataset(cifar10_test_raw.data, cifar10_test_raw.targets, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False)

# train proden
print("\nTraining PRODEN (PL)")
proden_model = create_model(train_config['num_classes'])
proden_loss = proden()
proden_optimizer = optim.Adam(proden_model.parameters(), lr=train_config['learning_rate'])
proden_accuracies = train_algorithm(proden_model, pl_loader, test_loader, proden_loss, proden_optimizer, train_config['epochs'], DEVICE)

# train MCL-LOG
print("\nTraining MCL-LOG (CL)")
mcl_log_model = create_model(train_config['num_classes'])
mcl_log_loss = MCL_Log(num_classes=train_config['num_classes'])
mcl_log_optimizer = optim.Adam(mcl_log_model.parameters(), lr=train_config['learning_rate'])
mcl_log_accuracies = train_algorithm(mcl_log_model, cl_loader, test_loader, mcl_log_loss, mcl_log_optimizer, train_config['epochs'], DEVICE)

# best accuracy results
print("\n--- Final Results ---")
best_proden = max(proden_accuracies)
best_mcl_log = max(mcl_log_accuracies)
print(f"Best Accuracy (PRODEN): {best_proden:.2f}%")
print(f"Best Accuracy (MCL-LOG): {best_mcl_log:.2f}%")

# accuracy plot
epochs = range(1, train_config['epochs'] + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, proden_accuracies, 'o-', label='PRODEN Test Accuracy')
plt.plot(epochs, mcl_log_accuracies, 'x-', label='MCL-LOG Test Accuracy')
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
