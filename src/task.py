import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from ProFed.partitioner import Environment, Region, download_dataset, split_train_validation, partition_to_subregions

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLP(nn.Module):

    def __init__(self, h1=128):
        super().__init__()
        self.fc1 = nn.Linear(28*28, h1)
        self.fc2 = nn.Linear(h1, 27)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


fds = None  # Cache FederatedDataset

def load_data(
        dataset_name: str,
        number_subregions: int,
        number_of_devices_per_subregion: int,
        partitioning_method: str,
        seed: int,
):
    global fds
    fds = get_mapping(dataset_name, number_subregions, number_of_devices_per_subregion, partitioning_method, seed)

def get_data(partition_id: int):
    partition = fds[partition_id]
    trainloader = DataLoader(partition[0], batch_size=32, shuffle=True)
    valloader = DataLoader(partition[1], batch_size=32)
    return trainloader, valloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def get_mapping(dataset_name, number_subregions: int, number_of_devices_per_subregion: int, partitioning_method: str, seed: int):
    train_data, _ = download_dataset(dataset_name)
    train_data, validation_data = split_train_validation(train_data, 0.8)

    environment = partition_to_subregions(train_data, validation_data, dataset_name, partitioning_method, number_subregions, seed)
    mapping = {}
    current_id = 0
    for region_id in range(number_subregions):
        mapping_devices_data = environment.from_subregion_to_devices(region_id, number_of_devices_per_subregion)
        for device_index, data in mapping_devices_data.items():
            mapping[current_id] = data
            current_id += 1
    return mapping