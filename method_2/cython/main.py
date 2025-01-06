import sys

sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
from PIL import Image
import afmm


def main():
    # Load image
    image = Image.open("../../imgs/crack 2.png")

    # Get skeleton
    threshold = 5  # Adjust as needed

    # FMM method
    fmm_method = 'afmm'  # "fmm" or "afmm"

    # Parse image type
    parse_image_type = 'binary'  # "binary" or "rgb"

    # Dislay Delta U and DT
    show_afmm_deltaU_dt = False  # True or False

    # Skeletonize the image
    dt_img, deltaU_img, skeleton = afmm.get_skeleton(
        image, threshold, fmm_method, parse_image_type, thinning=True
    )
    
    # Display images
    if fmm_method == "fmm":
        # Display the skeleton
        plt.imshow(dt_img, cmap="viridis")
        plt.colorbar()

        # Save the figure
        plt.savefig("fmm_dt.png")
        plt.close()

    else:
        if show_afmm_deltaU_dt:
            # Create a figure with 2 subplots (1 row, 2 columns)
            fig, (ax1, ax2) = plt.subplots(1, 2)

            # Plot data on the first subplot
            im1 = ax1.imshow(dt_img, cmap="viridis")
            ax1.set_title("DT")
            fig.colorbar(im1, ax=ax1, shrink=0.5)

            # Plot data on the second subplot
            im2 = ax2.imshow(deltaU_img, cmap="viridis")
            ax2.set_title("Delta U")
            fig.colorbar(im2, ax=ax2, shrink=0.5)

            plt.tight_layout()

            # Save the figure
            plt.savefig("afmm_dt_both.png")
            plt.close()
        else:
            # Display the skeleton
            plt.imshow(deltaU_img, cmap="viridis")
            plt.colorbar()

            # Save the figure
            plt.savefig("afmm_dt.png")
            plt.close()

        # Save the skeleton
        skeleton.save("skeleton.png")


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
        f"Algorithm Execution Time (AFMM TBFT Skeletonize): {elapsed_time:.4f} seconds."
    )
