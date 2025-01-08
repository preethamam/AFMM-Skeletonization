import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pyskelgrad
import time
from skimage.morphology import skeletonize

def process_image(image_path):
    """
    Process an image using the skeletonization algorithm.
    
    Args:
        image_path (str): Path to the input image
        threshold (int): Threshold for binary conversion (0-255)
    
    Returns:
        tuple: (skeleton_gradient, radius) numpy arrays
    """
    # Read image using PIL
    img = Image.open(image_path).convert('L')
    
    # Convert to numpy array and then to binary
    img_array = np.array(img)
    binary_img = (img_array > 128).astype(np.uint8)
    
    # Compute skeleton gradient
    s1 = time.time()
    skg, rad = pyskelgrad.compute_skeleton_gradient(binary_img)
    s2 = time.time()
    print("AFMM Skelgrad time taken: ", s2-s1)
    
    return skg, rad, img

def main():
    
    # Image path
    image_path = "../../images/crack.png"
    
    # Threshold for binary conversion
    threshold = 25
    
    # Process image
    skg, rad, img = process_image(image_path)
    
    # perform skeletonization
    skeleton = skeletonize(skg > threshold)

    # save a image using extension
    skeleton_thinned_pil = Image.fromarray(skeleton)
    skeleton_thinned_pil.save("crack_m1_pybind.png")
    
    # Visualize results        
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))

    # Plot images with color bars
    im1 = axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Input Binary Image')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(skeleton, cmap='gray')
    axes[0, 1].set_title('Skeleton')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[1, 0].imshow(skg, cmap='viridis')
    axes[1, 0].set_title('Skeleton Gradient')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(rad, cmap='viridis')
    axes[1, 1].set_title('Radius')
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()