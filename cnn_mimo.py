# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import DeepMIMO  


# Data Loading using DeepMIMO v2 (Using Amplitude Only, 1D Extraction)
def load_DeepMIMO_1D(noise_std=0.1, num_samples=5000, num_subcarriers=64, seed=42, scale_factor=1e8):
    """
    Loading DeepMIMO v2 data and processing it.
    
    - Generate the DeepMIMO dataset from the specified scenario.
    - Extract the channel from the BS's user data.
    - Convert the complex channel to amplitude.
    - Scales the amplitude using 'scale_factor' - Helps learn patterns/features effectively.
    - Extract a representative 1D channel vector by taking the first row of the channel image.
    - Add AWGN to produce the final recieved noisy observation (X).
    
    - X_tensor: Noisy channel observations, shape [num_samples, num_subcarriers].
    - Y_tensor: Ground-truth channel amplitudes, shape [num_samples, num_subcarriers].
    """
    np.random.seed(seed)

    parameters = {
        'dataset_folder': r'C:\Users\sreya\OneDrive\Desktop\DeepMIMO_CNN\scenarios',  
        'scenario': 'O1_60',
        'dynamic_settings': {'first_scene': 1, 'last_scene': 1},
        'num_paths': 15,
        'active_BS': np.array([1]),
        'user_row_first': 1,
        'user_row_last': 100,
        'row_subsampling': 1,
        'user_subsampling': 1,
        'bs_antenna': {
            'shape': np.array([1, 32, 4]),  # Simulating mMIMO Conditions
            'spacing': 0.5,
            'radiation_pattern': 'isotropic'
        },
        'ue_antenna': {
            'shape': np.array([1, 1, 1]),
            'spacing': 0.5,
            'radiation_pattern': 'isotropic'
        },
        'enable_BS2BS': 1,
        'OFDM_channels': 1,
        'OFDM': {
            'subcarriers': num_subcarriers,
            'subcarriers_limit': num_subcarriers,
            'subcarriers_sampling': 1,
            'bandwidth': 0.05,
            'RX_filter': 0
        }
    }
    
    dataset = DeepMIMO.generate_data(parameters)
    basestation = dataset[0]
    user_data = basestation['user']
    channels = user_data['channel']  # Shape: [N, 1, H, num_subcarriers]
    channels = channels[:num_samples, :, :, :]
    
    # Convert complex channels to amplitude
    channels_amp = np.abs(channels).astype(np.float32)
    channels_amp = channels_amp * scale_factor  # Scale the amplitudes
    
    # Extract a representative 1D channel vector by taking the first row (index 0) of H dimension
    Y = channels_amp[:, 0, 0, :]  # shape: [num_samples, num_subcarriers]
    
    # Generate noisy observations by adding AWGN
    X = Y + np.random.normal(0, noise_std, Y.shape).astype(np.float32)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    return X_tensor, Y_tensor


# CNN-based Channel Estimator 
class CNNChannelEstimator(nn.Module):
    def __init__(self, num_subcarriers=64):
        super(CNNChannelEstimator, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)   # Conv Layer 1
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)  # Conv Layer 2
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # Conv Layer 3
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)  # Conv Layer 4
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, padding=1)  # Conv Layer 5
        # After conv5, output shape: [B, 64, num_subcarriers]
        # Flattening features: 64 * num_subcarriers
        self.fc = nn.Linear(num_subcarriers * 64, num_subcarriers)  # Fully connected layer: maps 64 * num_subcarriers to num_subcarriers
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: [B, num_subcarriers]
        x = x.unsqueeze(1)  # Shape: [B, 1, num_subcarriers]
        x = self.relu(self.conv1(x))  # [B, 16, num_subcarriers]
        x = self.relu(self.conv2(x))  # [B, 32, num_subcarriers]
        x = self.relu(self.conv3(x))  # [B, 64, num_subcarriers]
        x = self.relu(self.conv4(x))  # [B, 64, num_subcarriers]
        x = self.relu(self.conv5(x))  # [B, 64, num_subcarriers]
        x = x.view(x.shape[0], -1)    # Flattening to [B, 64 * num_subcarriers]
        x = self.fc(x)                # Output: [B, num_subcarriers]
        return x

# Training Model (80/20 split)
def train_model(model, train_loader, num_epochs=20, lr=1e-3, lambda_phys=0.1, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()
    training_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mse_loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(train_loader.dataset)
        training_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}: Training Loss = {epoch_loss:.6f}")
    return model, training_losses


# Model Evaluation
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    mse_loss_fn = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = mse_loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    total_loss /= len(test_loader.dataset)
    return total_loss


# Execution Starts Here
if __name__ == '__main__':
    # Load DeepMIMO v2 1D data (64 subcarriers)
    noise_std = 1.0
    num_samples = 10000
    num_subcarriers = 64
    X, Y = load_DeepMIMO_1D(noise_std=noise_std, num_samples=num_samples, num_subcarriers=num_subcarriers, seed=42, scale_factor=1e8)
    print("X shape:", X.shape)  # Expected: [5000, 64]
    print("Y shape:", Y.shape)  # Expected: [5000, 64]
    
    # Split the data: 80% train, 20% test
    num_train = int(0.8 * X.shape[0])
    num_test = X.shape[0] - num_train
    X_train = X[:num_train]
    Y_train = Y[:num_train]
    X_test = X[num_train:]
    Y_test = Y[num_train:]
    
    batch_size = 32
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    
    num_epochs = 150
    model = CNNChannelEstimator(num_subcarriers=num_subcarriers)
    model, train_losses = train_model(model, train_loader, num_epochs=num_epochs, lr=1e-3, device=device)
    test_loss = evaluate_model(model, test_loader, device=device)
    print(f"Test Loss: {test_loss:.6f}")
    
    # Plotting training loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='CNN Channel Estimator')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Test evaluation: Plot predictions for 5 test samples
    X_test_samples = X_test[:5].to(device)
    Y_test_samples = Y_test[:5].cpu().numpy()  # shape: [5, 64]
    
    model.eval()
    with torch.no_grad():
        preds = model(X_test_samples).cpu().numpy()  # shape: [5, 64]
    
    num_samples_to_plot = 5
    plt.figure(figsize=(12, 6))
    for i in range(num_samples_to_plot):
        plt.subplot(1, num_samples_to_plot, i+1)
        plt.plot(Y_test_samples[i], label="True Channel", linestyle="dashed")
        plt.plot(preds[i], label="Estimated Channel", alpha=0.8)
        plt.xlabel("Subcarriers")
        plt.ylabel("Channel Amplitude")
        plt.title(f"Sample {i+1}")
        plt.legend()
    plt.suptitle("True vs. Estimated Channel Response (Test Samples)")
    plt.show()
