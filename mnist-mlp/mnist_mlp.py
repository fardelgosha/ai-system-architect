import time
from itertools import pairwise
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass, field, fields, replace
from collections import defaultdict
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]


@dataclass(frozen=True)
class MLPConfig:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers_sizes: Tuple[int, ...] = (28 * 28, 128, 10)
    activation_factory: type[nn.Module] = nn.ReLU
    transform = transforms.ToTensor()
    data_storage_path: str = "./data"
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 1
    model_log_step: int = 25  # step size in the number of batches


class MLP(nn.Module):
    def __init__(self, config: MLPConfig) -> None:
        super().__init__()

        if len(config.layers_sizes) < 2:
            raise ValueError("An MLP must have at least two layers!")

        layers = []
        for in_f, out_f in pairwise(config.layers_sizes):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(config.activation_factory())

        layers.pop()

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        return self.model(x)


@dataclass
class Metric:
    description: str
    plot_label: str
    # Metric history used for plotting
    history: list[float] = field(default_factory=list)


@dataclass
class MetricTracker:
    grad: Metric = field(
        default_factory=lambda: Metric(
            description="Average Gradient Norm", plot_label="$L_2$ Norm"
        )
    )

    param: Metric = field(
        default_factory=lambda: Metric(
            description="Average Parameters Norm", plot_label="$L_2$ Norm"
        )
    )

    grad_to_param_ratio: Metric = field(
        default_factory=lambda: Metric(
            description="Average Gradient $L_2$ Norm to Parameters $L_2$ Norm",
            plot_label="Relative Norms",
        )
    )

    # Temporary buffers
    _grad_buffer: list[Tensor] = field(default_factory=list)
    _param_buffer: list[Tensor] = field(default_factory=list)

    def get_all_metrics(self) -> list[Metric]:
        return [getattr(self, f.name) for f in fields(self) if f.type is Metric]

    def flush_to_history(self) -> None:
        if not self._grad_buffer:
            return

        grads = torch.stack(self._grad_buffer)
        params = torch.stack(self._param_buffer)

        eps = 1e-12  # Used to avoid division by zero
        self.grad.history.append(grads.mean().item())
        self.param.history.append(params.mean().item())
        self.grad_to_param_ratio.history.append((grads / (params + eps)).mean().item())

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

    def log_metrics(self, batch_index: int) -> None:
        is_log_step = ((batch_index + 1) % self.config.model_log_step == 0) or (
            batch_index + 1 == len(self.train_loader) * self.config.epochs
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
        # batch_index is the global batch index across all epochs
        batch_index = 0
        for _ in range(self.config.epochs):
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                logits = self.model(images.to(self.config.device))
                loss = self.criterion(logits, labels.to(self.config.device))
                loss.backward()
                self.log_metrics(batch_index)
                self.optimizer.step()
                batch_index += 1

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
        _, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

        for name, metrics in self.metrics.items():
            steps = [
                (i + 1) * self.config.model_log_step
                for i in range(len(metrics.grad.history))
            ]

            for ax_idx, metric in enumerate(metrics.get_all_metrics()):
                axes[ax_idx].plot(
                    steps,
                    metric.history,
                    label=name,
                    linestyle="--",
                    linewidth=1.5,
                    marker="o",
                    markersize=4,
                )
                axes[ax_idx].set_title(metric.description)
                axes[ax_idx].set_ylabel(metric.plot_label)
                axes[ax_idx].grid(True, alpha=0.3)
                axes[ax_idx].legend()

        axes[-1].set_xlabel("Training Batches")

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
