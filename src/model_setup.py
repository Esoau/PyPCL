import torch
import torch.optim as optim

from src.models import create_model
from src.proden_loss import proden
from src.mcl_losses import MCL_LOG, MCL_MAE, MCL_EXP
from src.pico.model import PiCOModel
from src.pico.utils_loss import PartialLoss, SupConLoss
from src.solar.utils_loss import partial_loss as solar_partial_loss

def setup_proden(args, train_config):
    model = create_model(train_config['num_classes'])
    loss = proden()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return model, loss, optimizer

def setup_mcl(args, train_config, loss_type='log'):
    model = create_model(train_config['num_classes'])
    if loss_type == 'log':
        loss = MCL_LOG(num_classes=train_config['num_classes'])
    elif loss_type == 'mae':
        loss = MCL_MAE(num_classes=train_config['num_classes'])
    elif loss_type == 'exp':
        loss = MCL_EXP(num_classes=train_config['num_classes'])
    else:
        raise ValueError(f"Unknown MCL loss type: {loss_type}")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return model, loss, optimizer

def setup_pico(args, train_config, pico_config, pico_train_dataset, device):
    pico_args = {
        'num_class': train_config['num_classes'], 'epochs': args.epochs, 'low_dim': pico_config['low_dim'],
        'moco_queue': pico_config['moco_queue'], 'moco_m': pico_config['moco_m'], 'proto_m': pico_config['proto_m'],
        'prot_start': pico_config['prot_start'], 'loss_weight': pico_config['loss_weight'],
        'conf_ema_range': pico_config['conf_ema_range']
    }
    model = PiCOModel(pico_args).to(device)
    
    initial_confidence = torch.ones(len(pico_train_dataset), pico_args['num_class']) / pico_args['num_class']
    cls_loss = PartialLoss(initial_confidence.to(device))
    cont_loss = SupConLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    return model, (cls_loss, cont_loss), optimizer, pico_args

def setup_solar(args, train_config, solar_config, solar_train_dataset, device):
    solar_args = {
        'num_class': train_config['num_classes'], 'epochs': args.epochs, 'warmup_epoch': solar_config['warmup_epoch'],
        'rho_range': solar_config['rho_range'], 'lamd': solar_config['lamd'], 'eta': solar_config['eta'],
        'tau': solar_config['tau'], 'est_epochs': solar_config['est_epochs'], 'gamma1': solar_config['gamma1'],
        'gamma2': solar_config['gamma2']
    }
    model = create_model(train_config['num_classes']).to(device)
    
    num_classes = train_config['num_classes']
    solar_given_label_matrix = torch.zeros(len(solar_train_dataset), num_classes)
    for i, p_label in enumerate(solar_train_dataset.given_label_matrix_sparse):
        solar_given_label_matrix[i, p_label] = 1.0
        
    loss_fn = solar_partial_loss(solar_given_label_matrix, device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    queue = torch.zeros(64 * args.batch_size, train_config['num_classes']).to(device)
    
    return model, loss_fn, optimizer, solar_args, queue
