import mrcfile
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as TF
import torch.distributions as dist

def estimate_transformations(X_list, A_init, lr=0.001, epochs=50, batch_size=5):
    """Estimate transformations using PyTorch SGD optimization with mini-batch training."""
    num_images = len(X_list)

    A_est = torch.nn.Parameter(torch.tensor(A_init, dtype=torch.float32))

    print("Start SGD")
    K = 360
    angles = torch.linspace(0, 2 * torch.pi, K, device=A_est.device)
    pixels = int(0.1 * A_init.shape[0])
    linex = torch.linspace(-pixels, pixels, 2*pixels, device=A_est.device)
    liney = torch.linspace(-pixels, pixels, 2*pixels, device=A_est.device)
    PDF = comput_PDF(linex, liney, A_init.shape[0])
    optimizer = torch.optim.SGD([A_est], lr=lr, momentum=0.4)

    min_loss = float('inf')
    best_A = None
    step_best = 0
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
                step_best = step

        print(f"Step {step}: loss = {loss.item()}")
    print(f"Step best{step_best}: loss = {min_loss}")
    return best_A.detach()



def comput_PDF(x, y, image_size):
    """Precompute a 2D probability density function (PDF)."""
    sigma_ = torch.tensor((0.05 * image_size)**2, dtype=torch.float32)

    PDF = torch.zeros((len(x), len(y)), dtype=torch.float32)
    for x_i, y_i in zip(range(len(x)), range(len(y))):
        PDF[x_i, y_i] = (x[x_i]**2 + y[y_i]**2)/(2*sigma_)


    return PDF


def loss_function(X_list, A_est, angles, linex, liney,  PDF):
    """Compute loss function between transformed A and X_list images."""
    sigma = 0.1
    loss = torch.tensor([0], dtype=torch.float32, device=A_est.device)

    num_images = len(X_list)

    rotated_images = [TF.rotate(A_est.unsqueeze(0), angle=float(angle * (180 / torch.pi))) for angle in
                      angles]
    for i in range(num_images):
        t2 = torch.tensor(np.array(X_list[i]), dtype=torch.float32, device=A_est.device).unsqueeze(0)

        min_loss = float('inf')

        for transformed_A in rotated_images:
            for x, y in zip(linex, liney):
                x_idx = torch.clamp(x - linex.min(), 0, PDF.shape[0] - 1).long()
                y_idx = torch.clamp(y - liney.min(), 0, PDF.shape[1] - 1).long()

                shifted_A = torch.roll(transformed_A, shifts=(int(x.item()), int(y.item())), dims=(1, 2))

                a = (1 / sigma ** 2) * torch.linalg.matrix_norm((t2 - shifted_A)) ** 2
                a += PDF[x_idx, y_idx]


                if a < min_loss:
                    min_loss = a

        loss += min_loss
    return loss


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
    image_name = 'pikacho-S'
    A_original = plt.imread(image_name+ ".png")
    images = Get_data(image_name)

    data = images.copy()
    stack = np.stack(data, axis=0)

    mean_img = normalize(np.mean(stack, axis=0))
    A_init = mean_img

    A_est = estimate_transformations(images, A_init, epochs=10)

    figure, axis = plt.subplots(1, 2)
    A = A_est.squeeze(0)

    axis[0].imshow(A_original, cmap='gray')
    axis[1].imshow(A, cmap='gray')

    plt.show()
