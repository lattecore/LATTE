import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.optim import lr_scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read the dataset
df = pd.read_csv("ground_truth.csv")
# Filter the DataFrame to only include rows where the label is 0, 1, or 2
df = df[df['label'].isin([0, 1, 2])]

# Prepare the data
X = df.drop('label', axis=1).values
y = df['label'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.LongTensor(y_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.LongTensor(y_test).to(device)

# Prepare DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)


# Definition of Layer Algorithm Selector
# class Layer_Algorithm_Selector(nn.Module):
#     def __init__(self):
#         super(Layer_Algorithm_Selector, self).__init__()
#         self.fc1 = nn.Linear(X_train.shape[1], 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, 32)
#         self.fc5 = nn.Linear(32, 16)
#         self.fc6 = nn.Linear(16, 3)# Assuming 3 classes (0-2)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = torch.relu(self.fc4(x))
#         x = torch.relu(self.fc5(x))
#         x = self.fc6(x)
#         return x


# Definition of Layer Algorithm Selector
class Layer_Algorithm_Selector(nn.Module):
    def __init__(self):
        super(Layer_Algorithm_Selector, self).__init__()
        self.attention = nn.Linear(X_train.shape[1], X_train.shape[1])
        self.feature_importance = nn.Parameter(torch.ones(X_train.shape[1]))
        self.fc1 = nn.Linear(X_train.shape[1], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 3)
        self.initialize_feature_importance()


    def initialize_feature_importance(self):
        # feature_importance result from SHAP
        self.feature_importance.data[9] = 2.0  # stride_height
        self.feature_importance.data[10] = 2.0  # stride_width
        self.feature_importance.data[2] = 0.5  # input_h
        self.feature_importance.data[3] = 0.5  # input_w
        self.feature_importance.data[4] = 0.5  # output_c


    def forward(self, x):
        attention_weights = torch.sigmoid(self.attention(x))
        combined_weights = attention_weights * self.feature_importance
        x = x * combined_weights
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

if __name__ == '__main__':
    # Initialize the model, loss, and optimizer
    net = Layer_Algorithm_Selector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=3, verbose=True)

    # Initialize variable to store the highest training accuracy
    highest_train_accuracy = 0.0

    # Rest of the code remains the same
    EP = 1500
    # Training loop
    for epoch in range(EP):
        epoch_loss = 0  # To keep track of total loss over an epoch
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # Forward pass
            output = net(X_batch)
            loss = criterion(output, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # Add batch loss

        # Average epoch loss
        epoch_loss /= len(train_loader)

        # Calculate train accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = net(X_batch)
                _, predicted = torch.max(output.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        train_accuracy = 100 * correct / total

        # Save the model if this epoch has the highest training accuracy so far
        if train_accuracy > highest_train_accuracy:
            highest_train_accuracy = train_accuracy
            torch.save(net, 'ML.pt')

        print(f'Epoch [{epoch+1}/{EP}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # Update the learning rate based on the average training loss for the epoch
        scheduler.step(epoch_loss)
