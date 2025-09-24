import os
import sys
import torch
import yaml
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gc

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import project modules
from src.data_utils import ComparisonDataGenerator, WeaklySupervisedDataset, PicoDataset, SoLarDataset
from src.models import create_model
from src.proden_loss import proden
from src.mcl_losses import MCL_LOG, MCL_MAE, MCL_EXP
from src.engine import train_algorithm, evaluate_model, train_pico_epoch, train_solar
from src.pico.model import PiCOModel
from src.pico.utils_loss import PartialLoss, SupConLoss
from src.solar.utils_loss import partial_loss as solar_partial_loss
from src.collate import collate_fn, pico_collate_fn, solar_collate_fn
from src.plotting import save_accuracy_plot
from src.args import parse_arguments

def main():
    # Configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data_generation']
    train_config = config['training']
    pico_config = config['pico']
    solar_config = config['solar']

    # Argument parsing
    args = parse_arguments(data_config, train_config)


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

    # Initialize the data generator with noise parameters
    eta_val = args.eta if args.noise == 'noisy' else 0.0
    generator = ComparisonDataGenerator(cifar10_train_raw, noise_type=args.noise, eta=eta_val)
    if args.noise == 'noisy':
        print(f"--- Generating noisy datasets with eta = {eta_val} ---")

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

    pl_loader = DataLoader(pl_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    cl_loader = DataLoader(cl_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Create test loader
    cifar10_test_raw = CIFAR10(root=data_config['cifar_path'], train=False, download=True, transform=test_transform)
    test_loader = DataLoader(cifar10_test_raw, batch_size=args.batch_size, shuffle=False)

    # Initialize accuracy lists and dictionary
    proden_accuracies = []
    mcl_log_accuracies = []
    mcl_mae_accuracies = []
    mcl_exp_accuracies = []
    pico_accuracies = []
    solar_accuracies = []

    all_accuracies = {
        'PRODEN': proden_accuracies,
        'MCL-LOG': mcl_log_accuracies,
        'MCL-MAE': mcl_mae_accuracies,
        'MCL-EXP': mcl_exp_accuracies,
        'PiCO': pico_accuracies,
        'SoLar': solar_accuracies
    }
    epochs_range = range(1, args.epochs + 1)

    # Train PRODEN
    print("\nTraining PRODEN (PL)")
    proden_model = create_model(train_config['num_classes'])
    proden_loss = proden()
    proden_optimizer = optim.SGD(proden_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    proden_accuracies.extend(train_algorithm(proden_model, pl_loader, test_loader, proden_loss, proden_optimizer, args.epochs, DEVICE))
    save_accuracy_plot(all_accuracies, epochs_range, args, project_root)
    del proden_model, proden_loss, proden_optimizer


    # Train MCL-LOG
    print("\nTraining MCL-LOG (CL)")
    mcl_log_model = create_model(train_config['num_classes'])
    mcl_log_loss = MCL_LOG(num_classes=train_config['num_classes'])
    mcl_log_optimizer = optim.SGD(mcl_log_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    mcl_log_accuracies.extend(train_algorithm(mcl_log_model, cl_loader, test_loader, mcl_log_loss, mcl_log_optimizer, args.epochs, DEVICE))
    save_accuracy_plot(all_accuracies, epochs_range, args, project_root)
    del mcl_log_model, mcl_log_loss, mcl_log_optimizer

    # Train MCL-MAE
    print("\nTraining MCL-MAE (CL)")
    mcl_mae_model = create_model(train_config['num_classes'])
    mcl_mae_loss = MCL_MAE(num_classes=train_config['num_classes'])
    mcl_mae_optimizer = optim.SGD(mcl_mae_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    mcl_mae_accuracies.extend(train_algorithm(mcl_mae_model, cl_loader, test_loader, mcl_mae_loss, mcl_mae_optimizer, args.epochs, DEVICE))
    save_accuracy_plot(all_accuracies, epochs_range, args, project_root)
    del mcl_mae_model, mcl_mae_loss, mcl_mae_optimizer

    # Train MCL-EXP
    print("\nTraining MCL-EXP (CL)")
    mcl_exp_model = create_model(train_config['num_classes'])
    mcl_exp_loss = MCL_EXP(num_classes=train_config['num_classes'])
    mcl_exp_optimizer = optim.SGD(mcl_exp_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    mcl_exp_accuracies.extend(train_algorithm(mcl_exp_model, cl_loader, test_loader, mcl_exp_loss, mcl_exp_optimizer, args.epochs, DEVICE))
    save_accuracy_plot(all_accuracies, epochs_range, args, project_root)
    del mcl_exp_model, mcl_exp_loss, mcl_exp_optimizer


    # Train PiCO
    print("\nTraining PiCO (PL)")
    pico_args = {
        'num_class': train_config['num_classes'],
        'epochs': args.epochs,
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
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=pico_collate_fn,
        pin_memory=True
    )

    initial_confidence = torch.ones(len(pico_train_dataset), pico_args['num_class']) / pico_args['num_class']
    pico_cls_loss = PartialLoss(initial_confidence.to(DEVICE))
    pico_cont_loss = SupConLoss()

    pico_optimizer = optim.SGD(pico_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        pico_cls_loss.set_conf_ema_m(epoch, pico_args)
        avg_loss = train_pico_epoch(pico_args, pico_model, pico_loader, pico_cls_loss, pico_cont_loss, pico_optimizer, epoch, DEVICE)
        current_accuracy = evaluate_model(pico_model, test_loader, DEVICE)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {current_accuracy:.2f}%")
        pico_accuracies.append(current_accuracy)
        
        # Add memory cleanup every 10 epochs
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    save_accuracy_plot(all_accuracies, epochs_range, args, project_root)
    del pico_model, pico_loader, pico_train_dataset, pico_cls_loss, pico_cont_loss, pico_optimizer, initial_confidence

    # Train SoLar
    print("\nTraining SoLar (PL)")
    solar_args = {
        'num_class': train_config['num_classes'],
        'epochs': args.epochs,
        'warmup_epoch': solar_config['warmup_epoch'],
        'rho_range': solar_config['rho_range'],
        'lamd': solar_config['lamd'],
        'eta': solar_config['eta'],
        'tau': solar_config['tau'],
        'est_epochs': solar_config['est_epochs'],
        'gamma1': solar_config['gamma1'],
        'gamma2': solar_config['gamma2']
    }
    solar_model = create_model(train_config['num_classes']).to(DEVICE)
    solar_train_dataset = SoLarDataset(pl_dataset_raw, generator.original_targets)
    solar_loader = DataLoader(
        solar_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=solar_collate_fn
    )

    print("Creating full label matrix for SoLar loss initialization...")
    num_classes = train_config['num_classes']
    solar_given_label_matrix = torch.zeros(len(solar_train_dataset), num_classes)
    for i, p_label in enumerate(solar_train_dataset.given_label_matrix_sparse):
        solar_given_label_matrix[i, p_label] = 1.0

    solar_loss_fn = solar_partial_loss(solar_given_label_matrix, DEVICE)
    solar_optimizer = optim.SGD(solar_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    queue = torch.zeros(64 * args.batch_size, train_config['num_classes']).to(DEVICE)

    solar_accuracies.extend(train_solar(solar_args, solar_model, solar_loader, test_loader, solar_loss_fn, solar_optimizer, DEVICE, queue))
    save_accuracy_plot(all_accuracies, epochs_range, args, project_root)

    # Final results
    print("\n--- Final Results ---")
    best_proden = max(proden_accuracies) if proden_accuracies else 0
    best_mcl_log = max(mcl_log_accuracies) if mcl_log_accuracies else 0
    best_mcl_mae = max(mcl_mae_accuracies) if mcl_mae_accuracies else 0
    best_mcl_exp = max(mcl_exp_accuracies) if mcl_exp_accuracies else 0
    best_pico = max(pico_accuracies) if pico_accuracies else 0
    best_solar = max(solar_accuracies) if solar_accuracies else 0

    print(f"Best Accuracy (PRODEN): {best_proden:.2f}%")
    print(f"Best Accuracy (MCL-LOG): {best_mcl_log:.2f}%")
    print(f"Best Accuracy (MCL-MAE): {best_mcl_mae:.2f}%")
    print(f"Best Accuracy (MCL-EXP): {best_mcl_exp:.2f}%")
    print(f"Best Accuracy (PiCO): {best_pico:.2f}%")
    print(f"Best Accuracy (SoLar): {best_solar:.2f}%")


    # Final accuracy plot is saved by the last call to save_accuracy_plot
    # The final `plt.show()` will display the last saved plot.
    # To ensure the final plot with all data is shown, we can call it one last time.
    save_accuracy_plot(all_accuracies, epochs_range, args, project_root)
    plt.show()

if __name__ == "__main__":
    main()