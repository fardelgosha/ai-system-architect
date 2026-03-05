from typing import Final
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]


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
    _MODEL_PARAMS_NORMS_LOG_STEP = 50

    def __init__(
        self, hidden_layer_size: int, learning_rate: float = 1e-3, epoch: int = 1
    ) -> None:
        self.device = torch.device("cpu")
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

        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        print(
            f"Traning dataset size: {len(self.train_loader)}, test dataset size: {len(self.test_loader)}"
        )

        self.hidden_layer_size = hidden_layer_size
        self.model = MLP(hidden_layer_size).to(self.device)

        # Initialize the dictionary of model parameters norms
        names = [name for name, _ in self.model.named_parameters()]
        self.model_params_norms = {name: [] for name in names}
        self.model_params_buffers = {name: [] for name in names}

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        if learning_rate is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = learning_rate

        self.lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"Hidden layer size: {hidden_layer_size}, learning Rate: {self.lr}, epoch: {epoch}"
        )

    def get_logits(self, images) -> torch.Tensor:
        return self.model(images.to(self.device))

    def get_loss(self, logits, labels) -> torch.Tensor:
        return self.criterion(logits, labels.to(self.device))

    def log_params_norms(self, sample_index: int) -> None:
        is_log_step = sample_index % self._MODEL_PARAMS_NORMS_LOG_STEP == 0
        for name, p in self.model.named_parameters():
            buffer = self.model_params_buffers[name]

            if is_log_step and buffer:
                self.model_params_norms[name].append(sum(buffer) / len(buffer))
                buffer.clear()

            if p.grad is not None:
                buffer.append(p.grad.norm().item())

    def train(self) -> None:
        for _ in range(self.epoch):
            for i, (images, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                logits = self.get_logits(images)
                loss = self.get_loss(logits, labels)
                loss.backward()
                self.optimizer.step()
                self.log_params_norms(i + 1)

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

    def plot_parameters_norms(self, train_eval_time: float) -> None:
        for name, norms in self.model_params_norms.items():
            x_axis = [
                (i + 1) * self._MODEL_PARAMS_NORMS_LOG_STEP for i in range(len(norms))
            ]
            plt.plot(x_axis, norms, label=name, marker="o")

        plt.title(
            f"MLP Params Norms: LR: {self.lr}, hidden layer: {self.hidden_layer_size}, epoch: {self.epoch}, accuracy: {self.accuracy}%, train + eval time: {train_eval_time:.3f}, norms log step: {self._MODEL_PARAMS_NORMS_LOG_STEP}",
            wrap=True,
        )
        plt.xlabel("Samples Processed")
        plt.ylabel("Parameters Norms")
        plt.legend()
        plt.show()


def main() -> None:
    hidden_layer_size = 512
    learning_rate = None
    epoch = 10

    classifier = DigitClassifier(hidden_layer_size, learning_rate, epoch)
    start = time.perf_counter()
    classifier.train()
    classifier.evaluate()
    end = time.perf_counter()
    train_eval_time = end - start
    print(f"Accuracy: {classifier.accuracy}, train + eval time: {train_eval_time:.3f}")
    classifier.plot_parameters_norms(train_eval_time)


if __name__ == "__main__":
    main()
