import mrcfile
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# def generate_transformed_images(A_est, linex, liney, angles):
#
#     # Ensure A_est has batch & channel dimensions
#     A_est = A_est.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
#
#     K, J, I = len(angles), len(liney), len(linex)
#     H, W = A_est.shape[-2], A_est.shape[-1]  # Image dimensions
#
#     # Storage for results
#     R_phi_ijk_A = torch.zeros((K, J, I, H, W), device=A_est.device)
#
#     # Normalize shifts for grid_sample
#     shift_x = linex / (W / 2)  # Normalize shift_x to [-1, 1] range
#     shift_y = liney / (H / 2)  # Normalize shift_y to [-1, 1] range
#
#     for k in range(K):
#         # Apply rotation to A_est
#         rotated_A = TF.rotate(A_est, angles[k].item(), interpolation=TF.InterpolationMode.BILINEAR)
#
#         for j in range(J):
#             for i in range(I):
#                 # Create affine transformation matrix for translation
#                 affine_matrix = torch.tensor([[1.0, 0.0, shift_x[i]],  # X translation
#                                               [0.0, 1.0, shift_y[j]]],  # Y translation
#                                               device=A_est.device, dtype=torch.float32)  # Ensure float32
#                 affine_matrix = affine_matrix.unsqueeze(0)  # Shape: (1, 2, 3)
#
#                 # Generate affine grid
#                 grid = F.affine_grid(affine_matrix, rotated_A.shape, align_corners=True)
#
#                 # Apply transformation (translation)
#                 transformed = F.grid_sample(rotated_A, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
#
#                 # Store result after removing batch & channel dimensions
#                 R_phi_ijk_A[k, j, i] = transformed.squeeze(0).squeeze(0)
#
#     return R_phi_ijk_A

def estimate_transformations(X_list, A_init, lr=0.001, epochs=50, batch_size=5):
    """Estimate transformations using PyTorch SGD optimization with mini-batch training."""

    num_images = len(X_list)
    sigma = 0.1
    A_est = torch.nn.Parameter(torch.tensor(A_init, dtype=torch.float32, requires_grad=True))

    sigma_moves = torch.tensor(0.05 * np.shape(A_init)[0], dtype=torch.float32)
    print("Start SGD")

    K = 360
    angles = torch.linspace(0, 2 * torch.pi, K, device=A_est.device)

    pixels = int(0.1 * A_init.shape[0])
    linex = torch.linspace(-pixels, pixels, pixels, device=A_est.device)
    liney = torch.linspace(-pixels, pixels, pixels, device=A_est.device)

    # Compute PDF correctly
    PDF = comput_PDF(linex, liney, A_init.shape[0], sigma_moves)

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
        print(f"Epoch {step}")
        optimizer.zero_grad()

        batch_indices = random.sample(range(num_images), batch_size)
        batch_X = [X_list[j] for j in batch_indices]

        loss = loss_function(batch_X, A_est, angles, linex, liney, PDF, sigma, Const_in_loss, K, pixels)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if abs(loss.item()) < min_loss:
                min_loss = abs(loss.item())
                best_A = A_est.detach().clone()
                step_best = step

        print(f"Step {step}: loss = {loss.item()}")

    print(f"Step best {step_best}: loss = {min_loss}")
    return best_A.detach()



def comput_PDF(x, y, image_size, sigma_):
    """Precompute a 2D probability density function (PDF)."""


    PDF = torch.zeros((len(x), len(y)), dtype=torch.float32)
    for x_i, y_i in zip(range(len(x)), range(len(y))):
        PDF[x_i, y_i] = (x[x_i]**2 + y[y_i]**2)/(2*sigma_**2)


    return PDF

#
# def loss_function(X_list, A_est, angles, linex, liney,   sigma,sigmaMoves,Const_in_loss, grid):
#     """Compute loss function between transformed A and X_list images."""
#
#     loss = torch.tensor([0], dtype=torch.float32, device=A_est.device)
#
#     num_images = len(X_list)
#
#     x = linex  # Shape: (N,)
#     y = liney  # Shape: (M,)
#
#     # Compute the coordinate terms
#     x_term = - (x ** 2) - (y ** 2) / (2 * sigmaMoves ** 2)  # Shape: (N,)
#     # Shape: (M,)
#
#     # Reshape for broadcasting
#     x_term = x_term.view(1, 1, -1)  # Shape: (N,1)
#     print(x_term.shape)
#     sigma = torch.tensor(sigma)
#     for I in X_list:
#         # Compute the norm term || I - R_phi_ijk A ||
#         I = torch.tensor(I,  dtype=torch.float32)
#         norm_term = -torch.norm(I - grid, dim= 2) / (2 * sigma ** 2)
#
#
#         # Combine the terms
#         E = norm_term + x_term   # Should align with i, j, k dimensions
#         loss += (torch.logsumexp(E, dim=(0, 1, 2)).detach())
#     return Const_in_loss + loss

def loss_function(X_list, A_est, angles, linex, liney, PDF, sigma, Const_in_loss, K, pixel):
    """Compute loss function between transformed A and X_list images."""

    loss = torch.tensor([0], dtype=torch.float32, device=A_est.device)  # Prevent log(0)
    a_list = []

    num_images = len(X_list)
    rotated_images = [TF.rotate(A_est.unsqueeze(0), angle=float(angle * (180 / torch.pi)),
                                interpolation=TF.InterpolationMode.BILINEAR) for angle in angles]

    for i in range(num_images):
        t2 = torch.tensor(np.array(X_list[i]), dtype=torch.float32, device=A_est.device).unsqueeze(0)

        for transformed_A in rotated_images:
            for x, y in zip(linex, liney):
                x_idx = torch.clamp(x - linex.min(), 0, PDF.shape[0] - 1).long()
                y_idx = torch.clamp(y - liney.min(), 0, PDF.shape[1] - 1).long()

                # Apply shift
                shifted_A = torch.roll(transformed_A, shifts=(int(x.item()), int(y.item())), dims=(1, 2))

                # Compute term inside log-sum-exp
                norm_diff = torch.linalg.matrix_norm((t2 - shifted_A)) ** 2
                a = (-norm_diff / (2 * sigma ** 2)) - PDF[x_idx, y_idx]
                a_list.append(a)

    # Convert to tensor and apply logsumexp safely
    a_stack = torch.stack(a_list)
    #min_val = torch.min(a_stack)
    loss = torch.logsumexp(a_stack, dim=0)

    return Const_in_loss + loss


def Get_data(image_name):
    """Load and normalize dataset."""
    arr = []
    for i in range(100):
        img = mrcfile.read(f'/mnt/c/Users/noyga/PycharmProjects/SGD_Images/output/{image_name + str(i)}.mrc')
        arr.append(normalize(img))

    return arr


def normalize(tensor):
    """Normalize a tensor to [0,1] range."""
    max =tensor.max()
    min = tensor.min()
    if max > 1 or min < 0:
        return (tensor - min) / (max - min)
    return tensor


if __name__ == '__main__':
    image_name = 'stitch-S'
    A_original = plt.imread(image_name+ ".png")
    images = Get_data(image_name)

    data = images.copy()
    stack = np.stack(data, axis=0)

    mean_img = normalize(np.mean(stack, axis=0))
    A_init = mean_img

    A_est = estimate_transformations(images, A_init, epochs=20)

    figure, axis = plt.subplots(1, 2)
    A = A_est.squeeze(0)

    axis[0].imshow(A_init, cmap='gray')
    axis[1].imshow(normalize(A), cmap='gray')

    plt.show()
