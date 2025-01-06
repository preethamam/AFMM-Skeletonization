import sys

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.morphology import skeletonize, thin
from skelgrad import compute_skeleton_gradient


def main():
    # Load image
    img_path = "../../imgs/crack 2.png"
    # "../imgs/mushroom.png"
    # "../imgs/crack 2.png"
    # "../imgs/crack.bmp"
    img = Image.open(img_path)

    # Get skeleton
    threshold = 20  # Adjust as needed

    # Convert to binary image
    if img.mode != "L":
        print("Converting to grayscale image.")
        img = img.convert("L")

    # Load your binary image (should be boolean or 0/1 values)
    img = np.array(img)  # Your binary image here
    # img = np.array(img/img.max(), dtype=np.int32)
    
    # Compute the skeleton gradient transform and radius
    # Get the start time
    st1 = time.time()
    skg, rad = compute_skeleton_gradient(img)
    # Get the end time
    et1 = time.time()
    
    # Print the execution time
    elapsed_time = et1 - st1
    print(f"Algorithm Execution Time (Skeletonize): {elapsed_time:.4f} seconds.")

    skg_threshold = skg > threshold

    # perform skeletonization
    skeleton = skeletonize(skg_threshold)

    # perform skeletonization
    skeleton_thinned = thin(skeleton)

    # Display images:  Create a figure with 2 subplots (1 row, 2 columns)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # Plot data on the first subplot
    im1 = ax1.imshow(img, cmap="gray")
    ax1.set_title("Binary Image")
    ax1.axis("off")

    # Plot data on the second subplot
    im2 = ax2.imshow(skg, cmap="jet")
    ax2.set_title("Skel. Gradient")
    # fig.colorbar(im2, ax=ax2)
    ax2.axis("off")

    # Plot data on the second subplot
    im3 = ax3.imshow(skeleton, cmap="gray")
    ax3.set_title("Skeleton")
    ax3.axis("off")

    # Plot data on the second subplot
    im4 = ax4.imshow(skeleton_thinned, cmap="gray")
    ax4.set_title("Skel. Thinned")
    ax4.axis("off")

    plt.tight_layout()
    plt.show()

    # Save the skeleton
    # skeleton.save("skeleton.png")


if __name__ == "__main__":
    import time

    # Get the start time
    st1 = time.time()

    # Execute main function
    main()

    # Get the end time
    et1 = time.time()

    # Print the execution time
    elapsed_time = et1 - st1
    print(
        f"Algorithm Execution Time (Complete Skeletonize): {elapsed_time:.4f} seconds."
    )
