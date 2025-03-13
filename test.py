import numpy as np
import scipy.ndimage as ndimage
from scipy.stats import beta
import matplotlib.pyplot as plt
import os
import mrcfile

def apply_transformation(image, rotation, translation, scale):
    """Apply transformation (rotation, translation, scale) to an image without resizing incorrectly."""
    transformed_image = ndimage.rotate(image, np.degrees(rotation), reshape=False, mode='nearest')
    transformed_image = ndimage.shift(transformed_image, shift=translation, mode='nearest')


    # Crop or pad to maintain shape consistency
    output_shape = image.shape
    if transformed_image.shape != output_shape:
        transformed_image = np.pad(transformed_image,
                                   [(0, max(0, output_shape[0] - transformed_image.shape[0])),
                                    (0, max(0, output_shape[1] - transformed_image.shape[1]))],
                                   mode='constant')[:output_shape[0], :output_shape[1]]

    return transformed_image



def create_images (image_name):
    num_images = 100
    sigma = 0.1

    # Generate synthetic images
    A_true = plt.imread(image_name + '.png')  # True underlying image

    image_size = A_true.shape[0]
    if len(A_true.shape) == 3:
        A_true = np.dot(A_true[..., :3], [0.2989, 0.5870, 0.1140])
    X_list = []

    for i in range(num_images):
        rotation = np.random.uniform(0, 2 * np.pi)
        translation = np.random.normal(0, 0.05 * image_size, size=2)
        scale = 5 * beta.rvs(2, 5)
        name = '/mnt/c/Users/noyga/PycharmProjects/SGD_Images/output/' + image_name + str(i) +'.mrc'
        X_list.append(
            apply_transformation(A_true, rotation, translation, scale) + sigma * np.random.randn(image_size, image_size))
        data = X_list[i].astype(np.float32)
        with mrcfile.new(name, overwrite=True) as mrc:
            mrc.set_data(data)
        print("Saved image path:", os.path.abspath(name))

# Example usage
if __name__ == "__main__":
    create_images('simple')
