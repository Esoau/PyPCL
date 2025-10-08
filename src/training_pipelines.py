import torch
import gc
from src.engine import train_algorithm, evaluate_model, train_pico_epoch, train_solar

def run_proden_training(args, loaders, train_config, device):
    """Runs the training pipeline for the PRODEN model."""
    from src.model_setup import setup_proden
    print("\nTraining PRODEN (PL)")
    model, loss, optimizer = setup_proden(args, train_config)
    accuracies = train_algorithm(model, loaders['pl'], loaders['test'], loss, optimizer, args.epochs, device)
    del model, loss, optimizer
    gc.collect()
    return accuracies

def run_mcl_training(args, loaders, train_config, device, loss_type='log'):
    """Runs the training pipeline for the MCL model."""
    from src.model_setup import setup_mcl
    print(f"\nTraining MCL-{loss_type.upper()} (CL)")
    model, loss, optimizer = setup_mcl(args, train_config, loss_type)
    accuracies = train_algorithm(model, loaders['cl'], loaders['test'], loss, optimizer, args.epochs, device)
    del model, loss, optimizer
    gc.collect()
    return accuracies

def run_pico_training(args, loaders, train_config, pico_config, pl_dataset_raw, original_targets, device):
    """Runs the training pipeline for the PiCO model."""
    from src.model_setup import setup_pico
    from src.data_utils import PicoDataset
    print("\nTraining PiCO (PL)")
    
    pico_train_dataset = PicoDataset(pl_dataset_raw, original_targets)
    model, (cls_loss, cont_loss), optimizer, pico_args = setup_pico(args, train_config, pico_config, pico_train_dataset, device)
    
    accuracies = []
    for epoch in range(args.epochs):
        cls_loss.set_conf_ema_m(epoch, pico_args)
        avg_loss = train_pico_epoch(pico_args, model, loaders['pico'], cls_loss, cont_loss, optimizer, epoch, device)
        current_accuracy = evaluate_model(model, loaders['test'], device)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {current_accuracy:.2f}%")
        accuracies.append(current_accuracy)
        
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
    del model, cls_loss, cont_loss, optimizer, pico_train_dataset
    gc.collect()
    return accuracies

def run_solar_training(args, loaders, train_config, solar_config, solar_train_dataset, device):
    """Runs the training pipeline for the SoLar model."""
    from src.model_setup import setup_solar
    print("\nTraining SoLar (PL)")
    model, loss_fn, optimizer, solar_args, queue = setup_solar(args, train_config, solar_config, solar_train_dataset, device)
    accuracies = train_solar(solar_args, model, loaders['solar'], loaders['test'], loss_fn, optimizer, device, queue)
    del model, loss_fn, optimizer, queue
    gc.collect()
    return accuracies
