import torch

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def calculate_sparsity(model, threshold=0.1):  # 🔥 Increased threshold
    total = 0
    zero = 0

    for layer in [model.fc1, model.fc2, model.fc3]:
        gates = torch.sigmoid(10 * layer.gate_scores)

        total += gates.numel()
        zero += (gates < threshold).sum().item()

    return 100 * zero / total