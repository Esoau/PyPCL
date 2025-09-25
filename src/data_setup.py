import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np

from src.data_utils import ComparisonDataGenerator, WeaklySupervisedDataset, PicoDataset, SoLarDataset
from src.collate import collate_fn, pico_collate_fn, solar_collate_fn
from src.clcifar import CLCIFAR10 as CLCIFAR10_dataset

def get_transforms():
    """Returns train and test transforms for CIFAR-10."""
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
    return train_transform, test_transform

def prepare_cifar10_datasets(args, data_config, train_config):
    """Prepares and returns the PL and CL datasets for CIFAR-10."""
    print("Loading original CIFAR-10 dataset...")
    cifar10_train_raw = CIFAR10(root=data_config['cifar_path'], train=True, download=True)

    eta_val = args.eta if args.noise == 'noisy' else 0.0
    generator = ComparisonDataGenerator(cifar10_train_raw, noise_type=args.noise, eta=eta_val)
    if args.noise == 'noisy':
        print(f"--- Generating noisy datasets with eta = {eta_val} ---")

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

    # Convert labels to tensors to ensure type consistency
    pl_dataset_raw.targets = [torch.tensor(t, dtype=torch.long) for t in pl_dataset_raw.targets]
    cl_dataset_raw.targets = [torch.tensor(t, dtype=torch.long) for t in cl_dataset_raw.targets]

    return pl_dataset_raw, cl_dataset_raw, generator.original_targets

def prepare_clcifar10_datasets(data_config):
    """Prepares and returns the PL and CL datasets for CLCIFAR-10."""
    print("Loading CLCIFAR-10 dataset...")
    cl_cifar10_train_raw = CLCIFAR10_dataset(root=data_config['cifar_path'])

    # Use all three complementary labels
    cl_labels_as_tensors = [torch.tensor(t, dtype=torch.long) for t in cl_cifar10_train_raw.targets]
    cl_dataset_raw = WeaklySupervisedDataset(data=cl_cifar10_train_raw.data, targets=cl_labels_as_tensors, transform=None)

    # PL dataset is the complement of all three CL labels
    num_classes = 10
    pl_labels = []
    for cl_label_list in cl_cifar10_train_raw.targets:
        pl_label = list(range(num_classes))
        for cl_label in cl_label_list:
            if cl_label in pl_label:
                pl_label.remove(cl_label)
        pl_labels.append(torch.tensor(pl_label, dtype=torch.long))

    pl_dataset_raw = WeaklySupervisedDataset(data=cl_cifar10_train_raw.data, targets=pl_labels, transform=None)

    # Keep original targets for evaluation
    original_targets = torch.tensor(cl_cifar10_train_raw.ord_labels)

    return pl_dataset_raw, cl_dataset_raw, original_targets


def prepare_datasets(args, data_config, train_config):
    """Dispatches to the correct dataset preparation function."""
    if args.dataset == 'cifar10':
        return prepare_cifar10_datasets(args, data_config, train_config)
    elif args.dataset == 'clcifar10':
        return prepare_clcifar10_datasets(data_config)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

def get_dataloaders(args, data_config, pl_dataset_raw, cl_dataset_raw, original_targets):
    """Creates and returns dataloaders for all algorithms."""
    train_transform, test_transform = get_transforms()

    # Standard PL/CL loaders
    pl_dataset = WeaklySupervisedDataset(pl_dataset_raw.data, pl_dataset_raw.targets, transform=train_transform)
    cl_dataset = WeaklySupervisedDataset(cl_dataset_raw.data, cl_dataset_raw.targets, transform=train_transform)
    pl_loader = DataLoader(pl_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    cl_loader = DataLoader(cl_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Test loader
    cifar10_test_raw = CIFAR10(root=data_config['cifar_path'], train=False, download=True, transform=test_transform)
    test_loader = DataLoader(cifar10_test_raw, batch_size=args.batch_size, shuffle=False)

    # PiCO loader
    pico_train_dataset = PicoDataset(pl_dataset_raw, original_targets)
    pico_loader = DataLoader(pico_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=pico_collate_fn, pin_memory=True)

    # SoLar loader
    solar_train_dataset = SoLarDataset(pl_dataset_raw, original_targets)
    solar_loader = DataLoader(solar_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=solar_collate_fn)

    return {
        'pl': pl_loader,
        'cl': cl_loader,
        'test': test_loader,
        'pico': pico_loader,
        'solar': solar_loader
    }, solar_train_dataset