from typing import Final
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]


@dataclass
class MetricTracker:
    # Metric history used for plotting
    grad_history: list[float] = field(default_factory=list)
    ratio_history: list[float] = field(default_factory=list)

    # Temporary buffers
    _grad_buffer: list[float] = field(default_factory=list)
    _param_buffer: list[float] = field(default_factory=list)

    def flush_to_history(self, log_step: int) -> None:
        if not self._grad_buffer:
            return

        grad_buffer_sum = sum(self._grad_buffer)
        avg_grad = grad_buffer_sum / log_step
        norm_ratio = grad_buffer_sum / sum(self._param_buffer)

        self.grad_history.append(avg_grad)
        self.ratio_history.append(norm_ratio)

        self._grad_buffer.clear()
        self._param_buffer.clear()


class MLP(nn.Module):
    def __init__(self, hidden_layer_size: int) -> None:
        super().__init__()
        # Fully connected layer 1
        self.fc1 = nn.Linear(28 * 28, hidden_layer_size)
        # Rectified linear unit
        self.relu = nn.ReLU()
        # Fully connected layer 2
        self.fc2 = nn.Linear(hidden_layer_size, 10)

    def forward(self, x) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DigitClassifier:
    _DATA_STORAGE_PATH: Final[str] = "./data"
    _MODEL_LOG_STEP = 50

    def __init__(
        self,
        hidden_layer_size: int,
        train_batch_size: int = 64,
        learning_rate: float = 1e-3,
        epoch: int = 1,
    ) -> None:
        self.device = torch.device("cpu")
        self.train_batch_size = train_batch_size
        self.epoch = epoch

        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(
            root=self._DATA_STORAGE_PATH, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=self._DATA_STORAGE_PATH,
            train=False,
            download=True,
            transform=transform,
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.train_batch_size, shuffle=True
        )
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        self.hidden_layer_size = hidden_layer_size
        self.model = MLP(hidden_layer_size).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.metrics = defaultdict(MetricTracker)

        if learning_rate is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = learning_rate

        self.lr = self.optimizer.param_groups[0]["lr"]

        print(
            f"Hidden layer size: {self.hidden_layer_size}, train batch size: {self.train_batch_size}, learning rate: {self.lr}, epoch: {epoch}, traning dataset size: {len(self.train_loader)}, test dataset size: {len(self.test_loader)}"
        )

    def get_logits(self, images) -> torch.Tensor:
        return self.model(images.to(self.device))

    def get_loss(self, logits, labels) -> torch.Tensor:
        return self.criterion(logits, labels.to(self.device))

    def log_metrics(self, sample_index: int) -> None:
        is_log_step = (sample_index + 1) % self._MODEL_LOG_STEP == 0

        for name, p in self.model.named_parameters():
            metrics = self.metrics[name]

            if is_log_step:
                metrics.flush_to_history(self._MODEL_LOG_STEP)

            if p.grad is not None:
                metrics._grad_buffer.append(p.grad.norm().item())
                metrics._param_buffer.append(p.norm().item())

    def train(self) -> None:
        for _ in range(self.epoch):
            for i, (images, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                logits = self.get_logits(images)
                loss = self.get_loss(logits, labels)
                loss.backward()
                self.optimizer.step()
                self.log_metrics(i + 1)

    def evaluate(self) -> None:
        self.model.eval()
        number_of_test_samples = len(self.test_loader.dataset)
        correct = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                logits = self.get_logits(images)
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()

        self.accuracy = 100 * correct / number_of_test_samples

    def plot_metrics(self) -> None:
        fig1 = plt.figure("Gradients")
        fig2 = plt.figure("Ratios")

        for name, metrics in self.metrics.items():
            x_axis = [
                (i + 1) * self._MODEL_LOG_STEP for i in range(len(metrics.grad_history))
            ]

            plt.figure(fig1.number)
            plt.plot(x_axis, metrics.grad_history, label=name, marker="o")

            plt.figure(fig2.number)
            plt.plot(x_axis, metrics.ratio_history, label=name, marker="o")

        plt.figure(fig1.number)
        plt.title("Gradient Norms")
        plt.xlabel("Samples Processed")
        plt.legend()

        plt.figure(fig2.number)
        plt.title("Norm Ratios")
        plt.xlabel("Samples Processed")
        plt.legend()

        plt.show()


def main() -> None:
    hidden_layer_size = 128
    train_batch_size = 64
    learning_rate = 1e-3
    epoch = 1

    classifier = DigitClassifier(
        hidden_layer_size, train_batch_size, learning_rate, epoch
    )
    start = time.perf_counter()
    classifier.train()
    classifier.evaluate()
    end = time.perf_counter()
    print(f"Accuracy: {classifier.accuracy}, train + eval time: {end - start:.3f} sec")
    classifier.plot_metrics()


if __name__ == "__main__":
    main()
