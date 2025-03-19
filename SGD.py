
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import gc
import tools as t
import time

def calculate_const(A_init, sigma, sigma_moves, alpha, beta):
    """Calculate the const value for the sum"""
    J = np.shape(A_init)[0] * np.shape(A_init)[1]
    return (
            J * torch.log(1 / (sigma * torch.sqrt(torch.tensor(2, dtype=torch.float32) * torch.pi))) +
            (2 * torch.log(1 / (2 * sigma_moves * torch.pi))) -
            torch.log(torch.lgamma(alpha).exp() * torch.lgamma(beta).exp() / torch.lgamma(alpha + beta).exp())
    )
def estimate_transformations(image_name, A_init, lr=0.001, epochs=50, batch_size=2, sigma=0.1, total_samples=50):
    """Optimized transformation estimation with mini-batch SGD."""
    A_est = torch.nn.Parameter(torch.tensor(A_init, dtype=torch.float32, requires_grad=True))
    best_poses = [[0,0,0,1]] * total_samples

    # Set transformation priors and parameters
    img_size = A_init.shape[0]
    sigma_moves = torch.tensor(0.05 * img_size, dtype=torch.float32)
    alpha = torch.tensor(2, dtype=torch.float32)
    beta = torch.tensor(5, dtype=torch.float32)
    Const_in_loss = calculate_const(A_init, sigma, sigma_moves, alpha, beta)

    # Initialize SGD optimizer with momentum
    optimizer = torch.optim.SGD([A_est], lr=lr, momentum=0.6)

    # Variables to track loss for early stopping
    step_total_loss = 0
    total_loss = 0

    # Training loop over epochs
    for epoch in range(epochs):
        print(f"current epoch:{epoch}")
        # Early stopping: Check if loss change is below threshold (convergence check)
        if step_total_loss != 0 and abs(total_loss - step_total_loss) / abs(step_total_loss) < 1e-4:
            return A_est.detach()
        step_total_loss = total_loss
        total_loss = 0
        mini_batch = int(total_samples / batch_size)
        # Iterate over mini-batches
        for step in range(mini_batch):
            optimizer.zero_grad() # Reset gradients

            # Load mini-batch data from the dataset
            batch_X = t.get_data_list(range(step*batch_size , (step+1)*batch_size), image_name)
            Pose_X = best_poses[step*batch_size : (step+1)*batch_size]

            # Compute loss and updated transformation poses
            loss, pose = loss_function(batch_X, A_est, sigma, Const_in_loss, epoch, img_size, sigma_moves, alpha, beta, Pose_X)

            # Compute loss and updated transformation poses
            for i in range(batch_size):
                if len(pose[i])>0:
                    best_poses[step*batch_size + i] = pose[i]

            loss.backward() # Backpropagate gradients
            optimizer.step()# Perform optimization step (update A_est)

            # Accumulate total loss (normalized by total samples)
            total_loss += (loss.item() / mini_batch)
    # Return final estimated transformation after training
    return A_est.clone().detach()


def loss_function(X_list, A_est, sigma, Const_in_loss, step, img_size, sigma_moves, alpha, beta, best_poses):
    """Compute loss function using optimized transformations."""
    loss = torch.tensor(0.0, dtype=torch.float32, device=A_est.device)
    min_pose = []
    img_num = 0
    # Loop over each image in X_list
    for I in X_list:
        # Get the linespace for the current iteration base on the last best pose for the current image
        [angles, linex, liney, lineScale, K, p] = t.Calculate_linespaces(best_poses[img_num], step, img_size, A_est)

        # Update constant term in loss function with precomputed normalization values
        Const_in_loss += (2 * torch.log(1 / torch.tensor(p, dtype=torch.float32)) +
                          torch.log(2 * torch.pi / torch.tensor(K, dtype=torch.float32)) +
                          torch.log(1 / torch.tensor(3, dtype=torch.float32))
                          )
        # Generate rotated versions of A_est for all given angles
        rotated_images = torch.stack([
            TF.rotate(A_est.unsqueeze(0), angle=float(angle * (180 / torch.pi)),
                      interpolation=TF.InterpolationMode.BILINEAR)
            for angle in angles
        ])
        min_value = float('inf')
        tmp_pose = []
        t2 = torch.as_tensor(I, dtype=torch.float32, device=A_est.device).unsqueeze(0)

        # Store norm differences for log-sum-exp computation
        norm_diffs = []
        index_angle = 0
        # Iterate over rotated versions of A_est
        for transformed_A in rotated_images:
            for x in linex: # Iterate over x translations
                for y in liney: # Iterate over y translations
                    shifted_A = torch.roll(transformed_A, shifts=(int(x.item()), int(y.item())), dims=(1, 2))
                    for scale in lineScale: # Iterate over scaling factors
                        scaled_A = scale * shifted_A
                        norm_diff = torch.linalg.matrix_norm(t2 - scaled_A) ** 2

                        # Compute loss component considering norm difference, translation penalty, and scale prior
                        a = ((-norm_diff / (2 * sigma ** 2)) -
                             ((x ** 2 + y ** 2) / (2 * sigma_moves ** 2)) +
                             torch.log((scale ** (alpha - 1)) * ((1 - scale) ** (beta - 1))))
                        norm_diffs.append(a.clone())

                        # Update best transformation parameters if the current loss is smaller
                        if abs(a) < min_value:
                            min_value = abs(a)
                            tmp_pose = [angles[index_angle],x,y,scale]
            index_angle += 1
        norm_diffs = torch.stack(norm_diffs)
        max_val = torch.max(norm_diffs)
        min_pose.append(tmp_pose)

        # Calculate the log of the current Image
        loss += ((torch.logsumexp(norm_diffs - max_val, dim=0).squeeze() - max_val) + Const_in_loss.squeeze())
        img_num += 1
    # Return total loss and best transformation parameters for each image
    return [loss, min_pose]



if __name__ == '__main__':
    #image_name = input('Enter image name:')
    start = time.time()
    image_name = 'pikacho2'
    total_samples = 100
    data = t.get_data_list([1], image_name)[0]
    A_init = t.get_mean_img(total_samples, image_name)
    gc.collect()
    sigma = 0.1
    # lr = input("Enter learning rate: (Recommended value: (0.001-0.004) higher for less details images):")
    # epochs = input("Enter number of epochs: (Recommended 50):")
    # batch_size = input("Enter batch size: (Recommended 2 for small CPU):")
    lr = 0.0001
    A_est = estimate_transformations(image_name, A_init,lr = lr, epochs=30, batch_size= 1,sigma = sigma, total_samples = total_samples)

    figure, axis = plt.subplots(1, 2)
    A = A_est.squeeze(0)
    end = time.time()

    print(f"Runtime: {end - start:.4f} seconds")

    axis[0].set(title="Data example")
    axis[0].imshow(data, cmap='gray')
    axis[1].set(title=f"Output: {lr}")
    axis[1].imshow(A, cmap='gray')

    plt.show()
