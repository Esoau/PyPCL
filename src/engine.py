import torch
from tqdm import tqdm

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


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