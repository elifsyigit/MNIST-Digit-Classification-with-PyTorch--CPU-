import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from datetime import datetime

training_dataset = pd.read_csv("dataset/mnist_train.csv")
training_dataset = np.array(training_dataset)
m,n = training_dataset.shape

# Split
val_data = training_dataset[:1000]
train_data = training_dataset[1000:]

train_loss_history = []
val_loss_history = []
val_acc_history = []
train_acc_history = []

log_filename = "pytorch_cpu_training_metrics.txt"
log_file = open(log_filename, 'w')


X_val = val_data[:, 1:].T / 255.0   # shape (784, 1000)
Y_val = val_data[:, 0].astype(int)   # shape (1000,)

X_train = train_data[:, 1:].T / 255.0
Y_train = train_data[:, 0].astype(int)

X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.long)
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)   # logits
        return x


X_train = torch.t(X_train)
X_val = torch.t(X_val) 
BATCH_SIZE = 32

# Create the TensorDatasets
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

# Create the DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Set device explicitly to 'cpu'
device = torch.device("cpu") 

model = NeuralNetwork()

model.to(device) #Move the model to the CPU

# Define loss function 
criterion = nn.CrossEntropyLoss()

# Define Optimizer
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Model instantiated and set to device: {device}")
print(f"Loss Function: {criterion.__class__.__name__}")
print(f"Optimizer: {optimizer.__class__.__name__}")


# ---  The Training ---

EPOCHS = 10 
start_time = time.time()
total_start_time = start_time

# Log start
timestamp = datetime.now().isoformat()
log_file.write(f"[{timestamp}] Run 'train' started for {EPOCHS} epochs, lr={learning_rate}\n")
log_file.write(f"[{timestamp}] epoch,train_accuracy,val_accuracy,step_time_s,cumulative_time_s\n")
log_file.flush()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    model.train() 
    running_train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        loss = criterion(outputs, target)

        optimizer.zero_grad() 
        loss.backward()        
        optimizer.step()   

        running_train_loss += loss.item()
        
        pred = outputs.argmax(dim=1)

        train_correct += (pred == target).sum().item()
        train_total += target.size(0)
    
    avg_train_loss = running_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)
    train_accuracy = 100. * train_correct / train_total
    train_acc_history.append(train_accuracy)

    # --- Validation---
    model.eval()
    
    val_loss = 0.0
    val_correct = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            val_loss += loss.item()
            pred = outputs.argmax(dim=1)
            val_correct += (pred == target).sum().item()
    
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    cumulative_time = epoch_end_time - total_start_time
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * val_correct / len(val_loader.dataset)
    
    val_loss_history.append(avg_val_loss)
    val_acc_history.append(val_accuracy)
    
    timestamp = datetime.now().isoformat()
    log_file.write(f"[{timestamp}] {epoch+1},{train_accuracy/100:.4f},{val_accuracy/100:.4f},{epoch_time:.4f},{cumulative_time:.4f}\n")
    log_file.flush()
    
    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}% | "
        f"Epoch Time: {epoch_time:.4f}s | "
        f"Total Time: {cumulative_time:.4f}s"
    )

# Log training completion
total_time = time.time() - total_start_time
timestamp = datetime.now().isoformat()
log_file.write(f"[{timestamp}] Run 'train' completed in {total_time:.2f}s\n")
log_file.write(f"[{timestamp}] Final Train Accuracy: {train_acc_history[-1]:.2f}%\n")
log_file.write(f"[{timestamp}] Final Val Accuracy: {val_acc_history[-1]:.2f}%\n")
log_file.close()

epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_history, label="Train Loss")
plt.plot(epochs, val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc_history, label="Train Accuracy")
plt.plot(epochs, val_acc_history, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Epoch")
plt.legend()

plt.tight_layout()
plt.show()