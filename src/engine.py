import torch
from tqdm import tqdm
from src.pico.model import PiCOModel

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if isinstance(model, PiCOModel):
                outputs = model(images, eval_only=True)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_algorithm(model, loader, test_loader, loss_fn, optimizer, epochs, device):
    best_accuracy = 0.0
    accuracies = []
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
        avg_loss = total_loss / len(loader)
        current_accuracy = evaluate_model(model, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {current_accuracy:.2f}%")
        accuracies.append(current_accuracy)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
    print(f"Training finished. Best accuracy: {best_accuracy:.2f}%\n")
    return accuracies

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