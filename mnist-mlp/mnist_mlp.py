import sys
import copy
import time
from itertools import pairwise
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass, field, replace
from collections import defaultdict
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]


@dataclass(frozen=True)
class MLPConfig:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers_sizes: Tuple[int, ...] = (28 * 28, 128, 10)
    activation: nn.Module = nn.ReLU()
    transform: transforms.ToTensor = transforms.ToTensor()
    data_storage_path: str = "./data"
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 1
    model_log_step: int = 50


class MLP(nn.Module):
    def __init__(self, mlp_config: MLPConfig) -> None:
        super().__init__()

        if len(mlp_config.layers_sizes) < 2:
            sys.exit("ERROR: An MLP must have at least two layers!")

        layers = []
        for in_f, out_f in pairwise(mlp_config.layers_sizes):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(copy.deepcopy(mlp_config.activation))

        layers.pop()

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        return self.model(x)


@dataclass
class MetricTracker:
    # Metric history used for plotting
    grad_history: list[float] = field(default_factory=list)
    param_history: list[float] = field(default_factory=list)
    ratio_history: list[float] = field(default_factory=list)

    # Temporary buffers
    _grad_buffer: list[Tensor] = field(default_factory=list)
    _param_buffer: list[Tensor] = field(default_factory=list)

    def flush_to_history(self) -> None:
        if not self._grad_buffer:
            return

        grads = torch.stack(self._grad_buffer)
        params = torch.stack(self._param_buffer)

        eps = 1e-8  # Used to avoid division by zero
        avg_grad = grads.mean().item()
        # avg_norm_ratio = (grads / (params + eps)).mean().item()
        avg_norm_ratio = (grads.sum() / params.sum()).item()

        self.grad_history.append(avg_grad)
        self.param_history.append(params.mean().item())
        self.ratio_history.append(avg_norm_ratio)

        self._grad_buffer.clear()
        self._param_buffer.clear()


class MLPTrainer:
    def __init__(self, mlp_config: MLPConfig, mlp: MLP):
        self.config = mlp_config

        train_dataset = datasets.MNIST(
            root=self.config.data_storage_path,
            train=True,
            download=True,
            transform=self.config.transform,
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        self.model = mlp.to(self.config.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        self.metrics = defaultdict(MetricTracker)

    def get_loss(self, logits, labels) -> torch.Tensor:
        return self.criterion(logits, labels.to(self.config.device))

    def log_metrics(self, sample_index: int) -> None:
        is_log_step = ((sample_index + 1) % self.config.model_log_step == 0) or (
            sample_index + 1 == len(self.train_loader)
        )

        for name, p in self.model.named_parameters():
            metrics = self.metrics[name]

            if p.grad is not None:
                metrics._grad_buffer.append(p.grad.norm().detach())
                metrics._param_buffer.append(p.norm().detach())

            if is_log_step:
                metrics.flush_to_history()

    def train(self) -> dict[str, MetricTracker]:
        self.model.train()
        for _ in range(self.config.epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                logits = self.model(images.to(self.config.device))
                loss = self.get_loss(logits, labels)
                loss.backward()
                self.optimizer.step()
                self.log_metrics(i)

        return self.metrics


class MLPEvaluator:
    def __init__(self, mlp_config: MLPConfig, mlp: MLP):
        self.config = mlp_config

        test_dataset = datasets.MNIST(
            root=self.config.data_storage_path,
            train=False,
            download=True,
            transform=self.config.transform,
        )

        self.test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        self.model = mlp.to(self.config.device)

    def evaluate(self) -> float:
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                logits = self.model(images.to(self.config.device))
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()

        return 100 * correct / len(self.test_loader.dataset)


class MLPPlotter:
    def __init__(self, metrics: dict[str, MetricTracker], mlp_config: MLPConfig):
        self.metrics = metrics
        self.config = mlp_config

    def plot_metrics(self) -> None:
        fig1 = plt.figure("Gradients", figsize=(10, 5))
        fig2 = plt.figure("Parameters", figsize=(10, 5))
        fig3 = plt.figure("Ratios", figsize=(10, 5))

        for name, metrics in self.metrics.items():
            steps = [
                (i + 1) * self.config.model_log_step * self.config.batch_size
                for i in range(len(metrics.grad_history))
            ]

            for fig_num, metric in [
                (fig1.number, metrics.grad_history),
                (fig2.number, metrics.param_history),
                (fig3.number, metrics.ratio_history),
            ]:
                plt.figure(fig_num)
                plt.plot(steps, metric, label=name, marker="o", markersize=4)

        for fig_num, title in [
            (fig1.number, "Gradient Norms"),
            (fig2.number, "Parameter Norms"),
            (fig3.number, "Norm Ratios"),
        ]:
            plt.figure(fig_num)
            plt.title(title)
            plt.xlabel("Samples Processed")
            plt.legend()

        plt.show()


def main() -> None:
    mlp_config = MLPConfig()
    # mlp_config = replace(mlp_config, batch_size=128)

    mlp = MLP(mlp_config)
    print(mlp)
    mlp_trainer = MLPTrainer(mlp_config, mlp)
    mlp_evaluator = MLPEvaluator(mlp_config, mlp)

    start = time.perf_counter()
    metrics = mlp_trainer.train()
    accuracy = mlp_evaluator.evaluate()
    end = time.perf_counter()
    print(f"Accuracy: {accuracy}%, train + eval time: {end - start:.3f} sec")

    mlp_plotter = MLPPlotter(metrics, mlp_config)
    mlp_plotter.plot_metrics()


if __name__ == "__main__":
    main()
