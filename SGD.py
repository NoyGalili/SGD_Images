
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import gc
import tools as t
import time
import Config as C

def calculate_const(A_init, sigma, sigma_moves, alpha, beta):
    J = np.shape(A_init)[0] * np.shape(A_init)[1]
    return (
        J * torch.log(1 / (sigma * torch.sqrt(torch.tensor(2, dtype=torch.float32) * torch.pi))) +
        (2 * torch.log(1 / (2 * sigma_moves * torch.pi))) -
        torch.log(torch.lgamma(alpha).exp() * torch.lgamma(beta).exp() / torch.lgamma(alpha + beta).exp())
    )


def estimate_transformations(folder_path, A_init, lr=0.001, epochs=50, batch_size=2, sigma=0.1, total_samples=50):
    """Optimized transformation estimation with mini-batch Stochastic Gradient Descent (SGD).
    This function estimates a transformation matrix `A_est` by iteratively minimizing a loss function."""
    A_est = torch.nn.Parameter(torch.tensor(A_init, dtype=torch.float32, requires_grad=True))

    # Get the image size (assuming a square image)
    img_size = A_init.shape[0]

    # Initialize the best transformation poses for each sample
    # Each pose consists of [rotation_angle, x_shift, y_shift, scale]
    best_poses = [[0, 0, 0, 1]] * total_samples
    # Define parameters for translation variance and scale priors
    sigma_moves = torch.tensor(0.05 * img_size, dtype=torch.float32)
    alpha = torch.tensor(2, dtype=torch.float32)
    beta = torch.tensor(5, dtype=torch.float32)
    Const_in_loss = calculate_const(A_init, sigma, sigma_moves, alpha, beta)

    # Initialize the optimizer (SGD with momentum)
    optimizer = torch.optim.SGD([A_est], lr=lr, momentum=0.6)
    count_batch = int(total_samples / batch_size)
    step_total_loss = 0
    total_loss = 0

    # Training loop over epochs
    for e in range(epochs):
        print(f"Epoch: {e}")

        # Early stopping: Stop training if the relative change in loss is below the threshold (convergence check)
        if step_total_loss != 0 and abs(total_loss - step_total_loss) / abs(step_total_loss) < C.threashold:
            return A_est.detach()
        step_total_loss = total_loss
        total_loss = 0

        # Iterate over mini-batches
        for step in range(count_batch):
            optimizer.zero_grad()# Reset gradients

            # Load mini-batch data from the dataset
            batch_X = t.get_data_list(range(step * batch_size, (step + 1) * batch_size), folder_path)
            # Extract corresponding best transformation poses for the batch
            Pose_X = best_poses[step * batch_size: (step + 1) * batch_size]

            # Compute the loss and updated transformation poses
            loss, pose = loss_function(batch_X, A_est, sigma, Const_in_loss, e, img_size, sigma_moves, alpha, beta, Pose_X)

            # Update best transformation poses based on the new pose estimates
            for i in range(batch_size):
                if len(pose[i]) > 0:
                    best_poses[step * batch_size + i] = pose[i]
            loss.backward()# Backpropagate gradients
            optimizer.step()# Perform an optimization step (update A_est)

            total_loss += (loss.item() / total_samples)
    # Return the final estimated transformation after normalizing it
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

    # Iterate over all images in the given batch
    for I in X_list:

        # Compute transformation search space (angles, translations, scaling)
        [angles, linex, liney, lineScale, K, p] = t.Calculate_linespaces(best_poses[img_num], step, img_size, A_est)
        # Update the constant term in the loss function
        Const_in_loss += (2 * torch.log(1 / torch.tensor(p, dtype=torch.float32)) +
                          torch.log(2 * torch.pi / torch.tensor(K, dtype=torch.float32)) +
                          torch.log(1 / torch.tensor(3, dtype=torch.float32)))
        # Generate rotated versions of A_est for all specified angles
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

        # Iterate over all rotated versions of A_est
        for transformed_A in rotated_images:
            for x in linex:
                for y in liney:
                    shifted_A = torch.roll(transformed_A, shifts=(int(x.item()), int(y.item())), dims=(1, 2))
                    for scale in lineScale:
                        scaled_A = scale * shifted_A
                        # Compute the squared norm difference between the transformed and original image
                        norm_diff = torch.linalg.matrix_norm(t2 - scaled_A) ** 2
                        # Compute loss component considering:
                        # - Rotation (norm_diff)
                        # - Translation penalty ((x^2 + y^2) term)
                        # - Scale prior (log Beta distribution term)
                        a = ((-norm_diff / (2 * sigma ** 2)) -
                             ((x ** 2 + y ** 2) / (2 * sigma_moves ** 2)) +
                             torch.log((scale ** (alpha - 1)) * ((1 - scale) ** (beta - 1))))
                        norm_diffs.append(a.clone())

                        # Update best transformation parameters if current loss is smaller
                        if abs(a) < min_value:
                            min_value = abs(a)
                            tmp_pose = [angles[index_angle], x, y, scale]
            index_angle += 1
        norm_diffs = torch.stack(norm_diffs)
        max_val = torch.max(norm_diffs)
        min_pose.append(tmp_pose)
        # Compute loss contribution for this image using the log-sum-exp
        loss += ((torch.logsumexp(norm_diffs - max_val, dim=0).squeeze() - max_val) + Const_in_loss.squeeze())
        img_num += 1
    # Return the total loss and the best transformation parameters
    return [loss, min_pose]


def runAlgo(lr, sigma, epochs, batch_size, folder_path, total_samples):
    start = time.time()
    data = t.get_data_list([0], folder_path)[0]

    # Compute the initial average image (used as the starting point for estimation)
    A_init = t.get_mean_img(total_samples, folder_path)
    gc.collect()
    # Estimate the underlying image using SGD optimization
    A_est = estimate_transformations(folder_path, A_init,
                                     lr=lr, epochs=epochs, batch_size=batch_size, sigma=sigma,
                                     total_samples=total_samples)

    figure, axis = plt.subplots(1, 2)
    end = time.time()

    print(f"Runtime: {end - start:.4f} seconds")

    axis[0].set(title="Data example")
    axis[0].imshow(data, cmap='gray')
    axis[1].set(title=f"Output (lr={lr})")
    axis[1].imshow(A_est, cmap='gray')
    plt.show()

if __name__ == '__main__':
    print(C.total_samples)
    folder_path = input("Please enter folder path for data")
    sigma = float(t.estimate_noise_batch(folder_path))
    print(sigma)
    runAlgo(C.learning_rate, sigma, C.epochs, C.batch_size, folder_path, C.total_samples)
