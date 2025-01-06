import numpy as np
from PIL import Image
import pyafmm
import matplotlib.pyplot as plt
import time
from skimage.morphology import skeletonize

def test_skeletonization(image_path, threshold, use_rgb):
        
    # Load image
    img = Image.open(image_path)
    
    # Convert to numpy array
    if use_rgb:
        # For RGB mode
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        img_array = np.array(img)
        if img_array.shape[2] != 3:
            img_array = img_array[:, :, :3]
        binary = img_array
        is_rgb = True
    else:
        # For grayscale mode
        img = img.convert('L')
        img_array = np.array(img)
        binary = (img_array > 128).astype(np.uint8) * 255
        is_rgb = False
        
    # Run skeletonization
    s1 = time.time()
    skeleton, deltaU, dt = pyafmm.skeletonize(binary, threshold, is_rgb)
    s2 = time.time()
    print('AFMM time taken:', s2 - s1)
    
    # perform skeletonization
    skeleton = skeletonize(skeleton)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    im1 = axes[0, 0].imshow(binary, cmap='gray')
    axes[0, 0].set_title('Input Binary Image')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(skeleton, cmap='gray')
    axes[0, 1].set_title('Skeleton')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[1, 0].imshow(deltaU, cmap='viridis')
    axes[1, 0].set_title('DeltaU')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(dt, cmap='viridis')
    axes[1, 1].set_title('Distance Transform')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Image path
    image_path = "../../imgs/example.png"
    
    # Threshold
    threshold = 100.0
    
    # Try with RGB mode
    test_skeletonization(image_path, threshold, use_rgb=True)
    
    # Try with grayscale mode
    # test_skeletonization(image_path, use_rgb=False)