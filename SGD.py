
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import gc
import tools as t
import time
from torch.optim.lr_scheduler import OneCycleLR

def calculate_const(A_init, sigma, sigma_moves, alpha, beta):
    J = np.shape(A_init)[0] * np.shape(A_init)[1]
    return (
            J * torch.log(1 / (sigma * torch.sqrt(torch.tensor(2, dtype=torch.float32) * torch.pi))) +
            (2 * torch.log(1 / (2 * sigma_moves * torch.pi))) -
            torch.log(torch.lgamma(alpha).exp() * torch.lgamma(beta).exp() / torch.lgamma(alpha + beta).exp())
    )
def estimate_transformations(image_name, A_init, lr=0.001, epochs=50, batch_size=2, sigma=0.1, total_samples=50):
    """Optimized transformation estimation with mini-batch Stochastic Gradient Descent (SGD).
    This function estimates a transformation matrix `A_est` by iteratively minimizing a loss function."""
    A_est = torch.nn.Parameter(torch.tensor(A_init, dtype=torch.float32, requires_grad=True))
    img_size = A_init.shape[0]
    best_poses = [[0,0,0,1]] * total_samples

    sigma_moves = torch.tensor(0.05 * img_size, dtype=torch.float32)
    alpha = torch.tensor(2, dtype=torch.float32)
    beta = torch.tensor(5, dtype=torch.float32)
    Const_in_loss = calculate_const(A_init, sigma, sigma_moves, alpha, beta)
    optimizer = torch.optim.SGD([A_est], lr=lr, momentum=0.6)
    a = int(total_samples / batch_size)
    step_total_loss = 0
    total_loss = 0
    for e in range(epochs):
        print(e)
        if step_total_loss != 0 and abs(total_loss - step_total_loss) / abs(step_total_loss) < 1e-4:
            return A_est.detach()
        step_total_loss = total_loss
        total_loss = 0
        for step in range(a):
            optimizer.zero_grad()
            batch_X = t.get_data_list(range(step*batch_size , (step+1)*batch_size), image_name)
            Pose_X = best_poses[step*batch_size : (step+1)*batch_size]

            loss, pose = loss_function(batch_X, A_est, sigma, Const_in_loss, e, img_size, sigma_moves, alpha, beta, Pose_X)
            for i in range(batch_size):
                if len(pose[i])>0:
                    best_poses[step*batch_size + i] = pose[i]
            loss.backward()
            optimizer.step()

            total_loss += (loss.item() / total_samples)

    return t.normalize(A_est.clone().detach())



def loss_function(X_list, A_est, sigma, Const_in_loss, step, img_size, sigma_moves, alpha, beta, best_poses):
    """Compute loss function using optimized transformations.

    This function estimates the transformation loss by comparing transformed versions
    of `A_est` with images in `X_list`, considering rotations, translations, and scaling.

    Args:
        X_list (list): List of input images.
        A_est (torch.Tensor): Current estimated transformation matrix.
        sigma (float): Standard deviation for Gaussian likelihood.
        Const_in_loss (torch.Tensor): Precomputed normalization constants.
        step (int): Current training step (affects transformation search space).
        img_size (int): Size of the input images.
        sigma_moves (torch.Tensor): Standard deviation for translation.
        alpha (torch.Tensor): Beta distribution parameter (scaling prior).
        beta (torch.Tensor): Beta distribution parameter (scaling prior).
        best_poses (list): Previously estimated best transformation poses.

    Returns:
        tuple: (loss value, list of best transformation parameters for each image)
    """
    loss = torch.tensor(0.0, dtype=torch.float32, device=A_est.device)
    min_pose = []
    img_num = 0
    for I in X_list:
        [angles, linex, liney, lineScale, K, p] = t.Calculate_linespaces(best_poses[img_num], step, img_size, A_est)
        Const_in_loss += (2 * torch.log(1 / torch.tensor(p, dtype=torch.float32)) +
                          torch.log(2 * torch.pi / torch.tensor(K, dtype=torch.float32)) +
                          torch.log(1 / torch.tensor(3, dtype=torch.float32))
                          )
        rotated_images = torch.stack([
            TF.rotate(A_est.unsqueeze(0), angle=float(angle * (180 / torch.pi)),
                      interpolation=TF.InterpolationMode.BILINEAR)
            for angle in angles
        ])
        min_value = float('inf')
        tmp_pose = []
        t2 = torch.as_tensor(I, dtype=torch.float32, device=A_est.device).unsqueeze(0)
        norm_diffs = []
        index_angle = 0
        for transformed_A in rotated_images:
            for x in linex:
                for y in liney:
                    shifted_A = torch.roll(transformed_A, shifts=(int(x.item()), int(y.item())), dims=(1, 2))
                    for scale in lineScale:
                        scaled_A = scale * shifted_A
                        norm_diff = torch.linalg.matrix_norm(t2 - scaled_A) ** 2
                        a = ((-norm_diff / (2 * sigma ** 2)) -
                             ((x ** 2 + y ** 2) / (2 * sigma_moves ** 2)) +
                             torch.log((scale ** (alpha - 1)) * ((1 - scale) ** (beta - 1))))
                        norm_diffs.append(a.clone())
                        if abs(a) < min_value:
                            min_value = abs(a)
                            tmp_pose = [angles[index_angle],x,y,scale]
            index_angle += 1
        norm_diffs = torch.stack(norm_diffs)
        max_val = torch.max(norm_diffs)
        min_pose.append(tmp_pose)
        loss += ((torch.logsumexp(norm_diffs - max_val, dim=0).squeeze() - max_val) + Const_in_loss.squeeze())
        img_num += 1
    return [loss, min_pose]


def runAlgo(lr, sigma, epochs, batch_size, image_name, total_samples):
    start = time.time()

    data = t.get_data_list([8], image_name)[0]
    A_init = t.get_mean_img(total_samples, image_name)
    gc.collect()
    # lr = input("Enter learning rate: (Recommended value: (0.001-0.004) higher for less details images):")
    # epochs = input("Enter number of epochs: (Recommended 50):")
    # batch_size = input("Enter batch size: (Recommended 2 for small CPU):")

    A_est = estimate_transformations(image_name, A_init, lr=lr, epochs=epochs, batch_size=batch_size, sigma=sigma,
                                     total_samples=total_samples)

    figure, axis = plt.subplots(1, 2)
    A = A_est.squeeze(0)
    end = time.time()

    print(f"Runtime: {end - start:.4f} seconds")

    axis[0].set(title="Data example1")
    axis[0].imshow(data, cmap='gray')
    axis[1].set(title=f"Output: {lr}")
    axis[1].imshow(A_est, cmap='gray')
    plt.show()

if __name__ == '__main__':
    runAlgo(0.001, 0.1, 50, 1, 'pikacho-S-0.1-', 100)

