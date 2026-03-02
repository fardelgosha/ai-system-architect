from typing import Final
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self, dim_fc1_out: int) -> None:
        super().__init__()
        # Fully connected layer 1
        self.fc1 = nn.Linear(28*28, dim_fc1_out)
        print(f"Constructed a fully-connected layer 1 with {dim_fc1_out} neurons.")
        # Rectified linear unit
        self.relu = nn.ReLU()
        # Fully connected layer 2
        self.fc2 = nn.Linear(dim_fc1_out, 10)

    def forward(self, x) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DigitClassifier:
    _DATA_STORAGE_PATH: Final[str] = "./data"

    def __init__(self, dim_fc1_out: int) -> None:
        self.device = torch.device("cpu")

        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(
            root=self._DATA_STORAGE_PATH, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(
            root=self._DATA_STORAGE_PATH, train=False, download=True, transform=transform)

        self.train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False)

        self.model = MLP(dim_fc1_out).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # for g in self.optimizer.param_groups:
        #     g['lr'] = 0.1

        lr = self.optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {lr}")

    def get_logits(self, images) -> torch.Tensor:
        return self.model(images.to(self.device))

    def get_loss(self, logits, labels) -> torch.Tensor:
        return self.criterion(logits, labels.to(self.device))

    def train(self) -> None:
        for images, labels in self.train_loader:
            self.optimizer.zero_grad()
            logits = self.get_logits(images)
            loss = self.get_loss(logits, labels)
            loss.backward()
            self.optimizer.step()

    def log_gradient_norms(self) -> None:
        for name, p in self.model.named_parameters():
            grad = p.grad
            if grad is not None:
                print(f"Parameter: {name}, Gradient Norm: {grad.norm()}")

    def evaluate(self) -> float:
        self.model.eval()
        number_of_test_samples = len(self.test_loader.dataset)
        correct = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                logits = self.get_logits(images)
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()

        return 100 * correct / number_of_test_samples


def main() -> None:
    classifier = DigitClassifier(128)
    start = time.perf_counter()
    classifier.train()
    classifier.log_gradient_norms()
    accuracy = classifier.evaluate()
    end = time.perf_counter()
    print(
        f"Accuracy: {accuracy}, training and evaluation time in seconds: {end - start:.6f}")


if __name__ == "__main__":
    main()
