import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# -------------------------
# 1. Device Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------
# 2. Load MNIST Dataset
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=1000,
    shuffle=False
)

print("MNIST dataset loaded successfully.")

# -------------------------
# 3. CNN Model
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Feature extraction layers
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )

        # Classification layers
        self.fc1 = nn.Linear(32 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Create model
model = SimpleCNN().to(device)

# Training setup
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.CrossEntropyLoss()

print("CNN model created successfully.")

# -------------------------
# 4. Train Model
# -------------------------
def train(model, loader, epochs=2):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {total_loss:.4f}"
        )

# -------------------------
# 5. Evaluate Model
# -------------------------
def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    print(
        f"Baseline Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    return accuracy

# -------------------------
# 6. FGSM Attack
# -------------------------
def fgsm_attack(image, epsilon, gradient):

    perturbation = epsilon * gradient.sign()

    adversarial_image = image + perturbation

    adversarial_image = torch.clamp(
        adversarial_image,
        0,
        1
    )

    return adversarial_image

# -------------------------
# 7. Evaluate Under Attack
# -------------------------
def evaluate_fgsm(
    model,
    loader,
    epsilon=0.25
):

    model.eval()

    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        images.requires_grad = True

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        model.zero_grad()

        loss.backward()

        gradient = images.grad.data

        adversarial_images = fgsm_attack(
            images,
            epsilon,
            gradient
        )

        adversarial_outputs = model(
            adversarial_images
        )

        predictions = adversarial_outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    accuracy = correct / total

    print(
        f"FGSM Accuracy "
        f"(epsilon={epsilon}): "
        f"{accuracy * 100:.2f}%"
    )

    return accuracy

# -------------------------
# 8. Adversarial Training
# -------------------------
def adversarial_train(
    model,
    loader,
    epochs=2,
    epsilon=0.25
):

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            images.requires_grad = True

            # First forward pass
            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            model.zero_grad()

            loss.backward()

            gradient = images.grad.data

            # Create attacked images
            adversarial_images = fgsm_attack(
                images,
                epsilon,
                gradient
            )

            # Train on attacked images
            optimizer.zero_grad()

            defended_outputs = model(
                adversarial_images.detach()
            )

            defended_loss = criterion(
                defended_outputs,
                labels
            )

            defended_loss.backward()

            optimizer.step()

            total_loss += defended_loss.item()

        print(
            f"Defense Epoch "
            f"{epoch + 1}/{epochs}, "
            f"Loss: {total_loss:.4f}"
        )

# -------------------------
# Run Baseline Training
# -------------------------
print("\nTraining baseline model...")
train(model, train_loader)

print("\nEvaluating baseline model...")
baseline_accuracy = evaluate(
    model,
    test_loader
)

print("\nLaunching FGSM attack...")

fgsm_accuracy = evaluate_fgsm(
    model,
    test_loader,
    epsilon=0.25
)

print("\nRunning adversarial defense...")

adversarial_train(
    model,
    train_loader
)

print("\nTesting defended model...")

defended_accuracy = evaluate_fgsm(
    model,
    test_loader,
    epsilon=0.25
)

# -------------------------
# 9. Plot Results
# -------------------------
labels = [
    "Baseline",
    "FGSM Attack",
    "Defense"
]

values = [
    baseline_accuracy,
    fgsm_accuracy,
    defended_accuracy
]

plt.bar(
    labels,
    values
)

plt.ylabel("Accuracy")

plt.title(
    "Model Performance Under Adversarial Attack"
)

plt.savefig(
    "results/model_results.png"
)

plt.show()

print(
    "\nGraph saved to results folder."
)