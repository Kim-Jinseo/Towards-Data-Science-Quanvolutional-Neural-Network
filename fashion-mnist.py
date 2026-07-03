import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pennylane as qml
import time
import pandas as pd
import numpy as np
import copy
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


torch.manual_seed(42)
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001 

print("--- Loading Fashion-MNIST CSVs ---")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])

class FashionCSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:, 0].values
        self.imgs = self.data.iloc[:, 1:].values.astype(np.uint8).reshape(-1, 28, 28)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        img = Image.fromarray(img, mode='L')
        if self.transform:
            img = self.transform(img)
        return img, label

train_csv_path = "fashion-mnist_train.csv"
test_csv_path = "fashion-mnist_test.csv"

full_train_dataset = FashionCSVDataset(train_csv_path, transform=train_transform)

val_size = int(0.1 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

val_subset.dataset.transform = eval_transform
test_dataset = FashionCSVDataset(test_csv_path, transform=eval_transform)

train_loader = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class ClassicalCNN(nn.Module):
    def __init__(self):
        super(ClassicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.activation = nn.GELU()
        
        self.fc1 = nn.Linear(288, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool1(self.activation(self.bn1(self.conv1(x))))
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


n_qubits = 4 
n_layers = 3 

try:
    dev = qml.device("lightning.gpu", wires=n_qubits)
    print("Backend: cuQuantum lightning.gpu initialized.")
except:
    dev = qml.device("default.qubit", wires=n_qubits)
    print("Backend: default.qubit CPU initialized.")

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs * torch.pi, wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuanvolutionalLayer(nn.Module):
    def __init__(self):
        super(QuanvolutionalLayer, self).__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.bias = nn.Parameter(torch.zeros(1, 4, 1, 1))

    def forward(self, x):
        B, C, H, W = x.size()
        out_height = H - 1
        out_width = W - 1
        
        patches = F.unfold(x, kernel_size=2, stride=1)
        L = patches.shape[-1] 
        
        patches = patches.transpose(1, 2).reshape(B * L, 4)
        q_results = self.qlayer(patches)
        
        q_results = q_results.view(B, L, 4).transpose(1, 2)
        out = q_results.view(B, 4, out_height, out_width)
        return out + self.bias

class HybridQNN(nn.Module):
    def __init__(self):
        super(HybridQNN, self).__init__()
        self.qconv = QuanvolutionalLayer()
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.activation = nn.GELU()
        
        self.fc1 = nn.Linear(288, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x_cpu = x.to('cpu')
        q_out = self.qconv(x_cpu) 
        x = q_out.to(x.device)
        
        x = self.pool1(self.activation(self.bn1(x)))
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_benchmark(model, name, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if "Hybrid" in name:
        model.qconv.to('cpu')
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    metrics = {
        "train_accuracy": [],
        "val_accuracy": [],
        "epoch_times": [],
        "total_parameters": count_parameters(model)
    }
    
    print(f"\n[{name}] Initialized on {device} | Trainable Params: {metrics['total_parameters']}")
    
    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        train_correct, train_total = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            predicted = torch.argmax(outputs, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        epoch_train_acc = train_correct / train_total
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                
                outputs = model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_val_acc = correct / total
        epoch_time = time.time() - epoch_start
        
        scheduler.step()
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
        
        metrics["train_accuracy"].append(epoch_train_acc)
        metrics["val_accuracy"].append(epoch_val_acc)
        metrics["epoch_times"].append(epoch_time)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.5f} | Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}")
    
    print(f"\nReloading best checkpoint (Val Acc: {best_val_acc:.4f}) for Final Evaluation on Unseen Test Data ({len(test_loader.dataset)} images)...")
    model.load_state_dict(best_model_weights)
    model.eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    test_acc = correct / total
    print(f"[{name}] FINAL TEST ACCURACY: {test_acc:.4f}")
    metrics["test_accuracy"] = test_acc
    
    return metrics

if __name__ == "__main__":
    cnn_model = ClassicalCNN()
    cnn_results = train_and_benchmark(cnn_model, "Classical CNN", train_loader, val_loader, test_loader)

    qnn_model = HybridQNN()
    qnn_results = train_and_benchmark(qnn_model, "Hybrid QNN", train_loader, val_loader, test_loader)
    
    print("\n=== FINAL PAPER COMPARISON ===")
    print(f"CNN | Params: {cnn_results['total_parameters']} | Avg Epoch Time: {sum(cnn_results['epoch_times'])/EPOCHS:.2f}s | Test Acc: {cnn_results['test_accuracy']:.4f}")
    print(f"QNN | Params: {qnn_results['total_parameters']} | Avg Epoch Time: {sum(qnn_results['epoch_times'])/EPOCHS:.2f}s | Test Acc: {qnn_results['test_accuracy']:.4f}")