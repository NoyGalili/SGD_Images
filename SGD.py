import mrcfile
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as TF


def estimate_transformations(X_list, A_init, lr=0.001, epochs=50, batch_size=5):
    """Estimate transformations using PyTorch SGD optimization with mini-batch training."""
    num_images = len(X_list)

    # ✅ Ensure `A_est` is a learnable parameter
    A_est = torch.nn.Parameter(torch.tensor(A_init, dtype=torch.float32))

    print("Start SGD")
    K = 180  # ✅ Reduce K from 360 to 180 (speed up computation)
    angles = torch.linspace(0, 2 * torch.pi, K, device=A_est.device)

    linex = torch.linspace(-30, 30, 30, device=A_est.device)  # ✅ Reduce step size
    liney = torch.linspace(-30, 30, 30, device=A_est.device)

    PDF = comput_PDF(linex, liney, A_init.shape[0])

    optimizer = torch.optim.SGD([A_est], lr=lr, momentum=0.9)  # ✅ Increased momentum

    min_loss = float('inf')
    best_A = None

    for step in range(epochs):
        print(f"Epoch {step}")
        optimizer.zero_grad()

        batch_indices = random.sample(range(num_images), batch_size)
        batch_X = [X_list[j] for j in batch_indices]

        loss = loss_function(batch_X, A_est, angles, linex, liney, PDF)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_A = A_est.clone()

        print(f"Step {step}: loss = {loss.item()}")

    return best_A.detach()


def apply_transformation(image, rotation, translation=(0, 0), scale=1.0):
    """Apply rotation, translation, and scaling properly."""
    if len(image.shape) == 2:
        image = image.unsqueeze(0)

    r = rotation * (180 / torch.pi)

    transformed_image = TF.rotate(image, angle=r)
    tx, ty = translation
    transformed_image = torch.roll(transformed_image, shifts=(int(ty), int(tx)), dims=(1, 2))
    transformed_image = TF.affine(transformed_image, angle=0, translate=[0, 0], scale=scale, shear=[0, 0])

    return transformed_image


def comput_PDF(x, y, image_size):
    """Precompute a 2D probability density function (PDF)."""
    sigma_ = 0.05 * (image_size ** 2)
    mu_x, sigma_x = 0, sigma_
    mu_y, sigma_y = 0, sigma_

    X, Y = torch.meshgrid(x, y, indexing='ij')

    pdf_x = (1 / (sigma_x * torch.sqrt(torch.tensor(2 * torch.pi, device=X.device)))) * torch.exp(
        -0.5 * ((X - mu_x) ** 2 / sigma_x))
    pdf_y = (1 / (sigma_y * torch.sqrt(torch.tensor(2 * torch.pi, device=X.device)))) * torch.exp(
        -0.5 * ((Y - mu_y) ** 2 / sigma_y))

    return pdf_x * pdf_y


def loss_function(X_list, A_est, angles, linex, liney, PDF):
    """Compute loss function between transformed A and X_list images."""
    sigma = 0.1
    loss = torch.tensor([0], dtype=torch.float32, device=A_est.device)

    num_images = len(X_list)

    rotated_images = [TF.rotate(A_est.unsqueeze(0), angle=float(angle * (180 / torch.pi))) for angle in
                      angles]  # ✅ Precompute rotations

    for i in range(num_images):
        t2 = torch.tensor(np.array(X_list[i]), dtype=torch.float32, device=A_est.device).unsqueeze(0)

        min_loss = float('inf')

        for transformed_A in rotated_images:
            for x, y in zip(linex, liney):
                # ✅ Ensure valid indexing
                x_idx = torch.clamp(x - linex.min(), 0, PDF.shape[0] - 1).long()
                y_idx = torch.clamp(y - liney.min(), 0, PDF.shape[1] - 1).long()

                shifted_A = torch.roll(transformed_A, shifts=(int(x.item()), int(y.item())), dims=(1, 2))

                a = (1 / sigma ** 2) * torch.linalg.matrix_norm((t2 - shifted_A)) ** 2
                a += PDF[x_idx, y_idx]

                if a < min_loss:
                    min_loss = a

        loss += min_loss  # ✅ Accumulate best loss per image

    return loss


def Get_data():
    """Load and normalize dataset."""
    arr = []
    for i in range(100):
        img = mrcfile.read(f'/mnt/c/Users/noyga/PycharmProjects/SGD_Images/output/pikacho2{i}.mrc')
        arr.append(normalize(img))

    return arr


def normalize(tensor):
    """Normalize a tensor to [0,1] range."""
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


if __name__ == '__main__':
    images = Get_data()

    data = images.copy()
    stack = np.stack(data, axis=0)

    mean_img = normalize(np.mean(stack, axis=0))
    A_init = mean_img

    # ✅ Run estimation with optimized functions
    A_est = estimate_transformations(images, A_init, epochs=20)

    # ✅ Display result
    figure, axis = plt.subplots(1, 2)
    A = A_est.squeeze(0)

    axis[0].imshow(A_init, cmap='gray')
    axis[1].imshow(A, cmap='gray')

    plt.show()
