import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from model import PrunableNet
from evaluate import evaluate, calculate_sparsity

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data (normalized)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 🔥 Reduce dataset (FAST)
train_dataset = Subset(train_dataset, range(10000))
test_dataset = Subset(test_dataset, range(2000))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 🔥 Strong sparsity function
def normalized_sparsity(model):
    total = 0
    count = 0

    for layer in [model.fc1, model.fc2, model.fc3]:
        gates = torch.sigmoid(10 * layer.gate_scores)
        total += gates.sum()
        count += gates.numel()

    return total / count

# 🔥 Only 2 lambda values (FAST)
lambdas = [1e-3, 5e-3]

results = []

for lam in lambdas:
    print(f"\nTraining with lambda = {lam}")

    model = PrunableNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 🔥 Only 3 epochs (FAST)
    for epoch in range(3):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            ce_loss = F.cross_entropy(outputs, labels)

            sparsity_loss = normalized_sparsity(model)

            loss = ce_loss + lam * sparsity_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    acc = evaluate(model, test_loader, device)
    sparsity = calculate_sparsity(model)

    print(f"Accuracy: {acc:.2f}% | Sparsity: {sparsity:.2f}%")

    results.append((lam, acc, sparsity))

print("\nFinal Results:")
for r in results:
    print(f"Lambda={r[0]} | Acc={r[1]:.2f}% | Sparsity={r[2]:.2f}%")