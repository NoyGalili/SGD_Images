import mrcfile
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as TF
import gc

def estimate_transformations(image_name, A_init, lr=0.002, epochs=50, batch_size=2):
    """Estimate transformations using PyTorch SGD optimization with mini-batch training."""
    batch_X = []
    sigma = 0.1
    A_est = torch.nn.Parameter(torch.tensor(A_init, dtype=torch.float32, requires_grad=True))

    sigma_moves = torch.tensor(0.05 * np.shape(A_init)[0], dtype=torch.float32)
    print("Start SGD")

    K = 180
    angles = torch.linspace(0, 2 * torch.pi, K, device=A_est.device)

    pixels = int(0.1 * A_init.shape[0])
    linex = torch.linspace(-pixels, pixels, int(pixels/2), device=A_est.device)
    liney = torch.linspace(-pixels, pixels, int(pixels/2), device=A_est.device)

    # Compute PDF correctly
    with torch.no_grad():
        PDF = comput_PDF(linex, liney, sigma_moves)

    optimizer = torch.optim.SGD([A_est], lr=lr, momentum=0.6)

    J = np.shape(A_init)[0] * np.shape(A_init)[1]
    Const_in_loss = (
            J * torch.log(1 / (sigma * torch.sqrt(torch.tensor(2, dtype=torch.float32) * torch.pi))) +
            (2 * torch.log(1 / (2 * sigma_moves * torch.pi))) +
            torch.log(2 * torch.pi / torch.tensor(K, dtype=torch.float32))
    )

    min_loss = float('inf')
    best_A = None
    step_best = 0

    for step in range(epochs):
        gc.collect()
        print(f"Epoch {step}")
        optimizer.zero_grad()

        batch_indices = random.sample(range(100), batch_size)
        batch_X = get_data_list(batch_indices, image_name)

        loss = loss_function(batch_X, A_est, angles, linex, liney, PDF, sigma, Const_in_loss)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if abs(loss.item()) < min_loss:
                min_loss = abs(loss.item())
                best_A = A_est.detach().clone()
                step_best = step

        print(f"Step {step}: loss = {loss.item()}")
    print(f"Step best {step_best}: loss = {min_loss}")
    return [best_A.detach(),batch_X]



def comput_PDF(x, y, sigma_):
    """Precompute a 2D probability density function (PDF)."""

    PDF = torch.zeros((len(x), len(y)), dtype=torch.float32)
    for x_i, y_i in zip(range(len(x)), range(len(y))):
        PDF[x_i, y_i] = (x[x_i]**2 + y[y_i]**2)/(2*sigma_**2)
    return PDF

def loss_function(X_list, A_est, angles, linex, liney, PDF, sigma, Const_in_loss):
    """Compute loss function between transformed A and X_list images."""
    num_images = len(X_list)
    rotated_images = [TF.rotate(A_est.unsqueeze(0), angle=float(angle * (180 / torch.pi)),
                                interpolation=TF.InterpolationMode.BILINEAR) for angle in angles]
    loss = torch.tensor([0], dtype=torch.float32, device=A_est.device)
    for I in X_list:
        gc.collect()
        a_list = []
        t2 = torch.as_tensor(I, dtype=torch.float32, device=A_est.device).unsqueeze(0)
        for transformed_A in rotated_images:
            for x, y in zip(linex, liney):
                x_idx = torch.clamp(x - linex.min(), 0, PDF.shape[0] - 1).long()
                y_idx = torch.clamp(y - liney.min(), 0, PDF.shape[1] - 1).long()

                # Apply shift
                shifted_A = torch.roll(transformed_A, shifts=(int(x.item()), int(y.item())), dims=(1, 2))

                # Compute term inside log-sum-exp
                norm_diff = torch.linalg.matrix_norm((t2 - shifted_A)) ** 2
                a = (-norm_diff / (2 * sigma ** 2))+PDF[x_idx][y_idx]
                a_list.append(a)
        a_stack = torch.stack(a_list)
        max_val = torch.max(a_stack)
        loss += ((torch.logsumexp(a_stack-max_val, dim=0) - max_val) + Const_in_loss)

    return loss

def normalize(tensor):
    """Normalize a tensor to [0,1] range."""
    max =tensor.max()
    min = tensor.min()
    if max > 1 or min < 0:
        return (tensor - min) / (max - min)
    return tensor
def get_data_list(index_list, image_name):
    """Load and normalize items from the data."""
    arr = []
    for i in index_list:
        img = mrcfile.read(f'/mnt/c/Users/noyga/PycharmProjects/SGD_Images/output/{image_name + str(i)}.mrc')
        arr.append(normalize(img))

    return arr
def get_mean_img():
    """create a mean of all images."""
    arr = []
    for i in range(100):
        img = mrcfile.read(f'/mnt/c/Users/noyga/PycharmProjects/SGD_Images/output/{image_name + str(i)}.mrc')
        arr.append(normalize(img))
    stack = np.stack(arr, axis=0)
    return normalize(np.mean(stack, axis=0))
if __name__ == '__main__':
    #image_name = input('Enter image name:')
    image_name = 'pikacho2'
    A_init = get_mean_img()
    gc.collect()
    # lr = input("Enter learning rate: (Recommended value: (0.001-0.004) higher for less details images):")
    # epochs = input("Enter number of epochs: (Recommended 50):")
    # batch_size = input("Enter batch size: (Recommended 2 for small CPU):")

    [A_est, batch_X] = estimate_transformations(image_name, A_init,lr = 0.002, epochs=30, batch_size= 2)

    figure, axis = plt.subplots(1, 2)
    A = A_est.squeeze(0)

    axis[0].imshow(batch_X[0], cmap='gray')
    axis[1].imshow(A, cmap='gray')

    plt.show()
