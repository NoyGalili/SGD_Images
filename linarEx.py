
import mrcfile
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF

# 1️⃣ Create dummy data
x = torch.linspace(0, 10, 100).unsqueeze(1)  # Shape: (100, 1)
y = 2 * x + 3 + torch.randn(x.size()) * 0.5  # y = 2x + 3 + noise

# 2️⃣ Define a simple linear model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input, one output

    def forward(self, x):
        return self.linear(x)

model = torch.nn.Parameter(torch.zeros((100, 4), dtype=torch.float32))

# 3️⃣ Define loss function and SGD optimizer
criterion = nn.MSELoss()  # Mean Squared Error
optimizer = optim.SGD(model, lr=0.01)  # SGD with learning rate 0.01

# 4️⃣ Training loop
epochs = 100
for epoch in range(epochs):
    model.train()

    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()   # Clear previous gradients
    loss.backward()         # Backpropagation
    optimizer.step()        # Update weights

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5️⃣ Plot the result
predicted = model(x).detach()

plt.scatter(x, y, label='Actual Data')
plt.plot(x, predicted, color='red', label='Fitted Line')
plt.legend()
plt.title('Linear Regression with SGD in PyTorch')
plt.show()