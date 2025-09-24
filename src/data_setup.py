import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from src.data_utils import ComparisonDataGenerator, WeaklySupervisedDataset, PicoDataset, SoLarDataset
from src.collate import collate_fn, pico_collate_fn, solar_collate_fn

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

def prepare_datasets(args, data_config, train_config):
    """Prepares and returns the PL and CL datasets."""
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
    
    return pl_dataset_raw, cl_dataset_raw, generator.original_targets

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
