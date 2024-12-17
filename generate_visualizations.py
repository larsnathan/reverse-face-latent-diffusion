import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import glob

def generate_test_grid():
    # Example data
    original_images = [f"./test_data/img_align_celeba/{f}" for f in ["000330.jpg", "001596.jpg", "006672.jpg", "011972.jpg", "012761.jpg"]]
    original_images = [Image.open(img).resize((64, 64)) for img in original_images]  # Load and resize original images

    images = []
    for image_file in sorted(glob.glob("./test_data/predicted/*.jpg")):
        images.append(Image.open(image_file))

    # Parameters for the grid
    rows = len(original_images)  # One row per original image
    cols = 11  # 1 column for original + 10 columns for other images

    # Create a figure and GridSpec
    fig = plt.figure(figsize=(cols, rows + 1))  # Adjust height for extra padding
    gs = GridSpec(rows, cols, width_ratios=[1.5] + [1] * 10)  # Wider first column

    axes = []
    for row in range(rows):
        row_axes = []
        for col in range(cols):
            ax = fig.add_subplot(gs[row, col])
            row_axes.append(ax)
        axes.append(row_axes)

    # Loop through each row to populate the grid
    for row_idx, ax_row in enumerate(axes):
        # Add the original image to the first column
        ax_row[0].imshow(original_images[row_idx], cmap='gray')
        ax_row[0].axis('off')  # Hide axes for original images

        # Fill the remaining columns with other images
        start_idx = row_idx * 10
        for col_idx, ax in enumerate(ax_row[1:], start=1):
            img_idx = start_idx + col_idx - 1
            if img_idx < len(images):
                ax.imshow(images[img_idx], cmap='gray')
            ax.axis('off')  # Hide axes for all images

    # Draw a continuous vertical line
    line = plt.Line2D([0.12, 0.12], [0.02, 0.97], transform=fig.transFigure, color="black")
    fig.add_artist(line)

    # Add a caption for the first column
    fig.text(0.065, 0.97, "Originals", fontsize=16, fontweight='bold', ha='center', va='top')

    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add padding at the top for the title
    plt.subplots_adjust(wspace=0.3)  # Add spacing between subplots
    plt.savefig("./visualizations/test_images.jpg")

def generate_training_grid():
    original_images = ["./eval_images/base_image.jpg", "./test_data/img_align_celeba/001596.jpg"] # [f"./test_data/img_align_celeba/{f}" for f in ["000330.jpg", "001596.jpg", "006672.jpg", "011972.jpg", "012761.jpg"]]
    original_images = [Image.open(img).resize((64, 64)) for img in original_images]  # Load and resize original images

    epochs = [1, 3, 5, 10, 20, 30, 40, 50, 70, 100]
    images = [f'./eval_images/eval-{e}.jpg' for e in epochs]
    images.extend([f'./eval_images/eval-test-{e}.jpg' for e in epochs])
    print(images)
    images = [Image.open(img).resize((64, 64)) for img in images]
    for image_file in sorted(glob.glob("./eval_images/*.jpg")):
        images.append(Image.open(image_file))

    # Parameters for the grid
    rows = len(original_images)  # One row per original image
    cols = len(epochs)+1  # 1 column for original + 10 columns for other images

    # Create a figure and GridSpec
    fig = plt.figure(figsize=(cols, rows + 1))  # Adjust height for extra padding
    gs = GridSpec(rows, cols, width_ratios=[1.5] + [1] * 10)  # Wider first column

    axes = []
    for row in range(rows):
        row_axes = []
        for col in range(cols):
            ax = fig.add_subplot(gs[row, col])
            row_axes.append(ax)
        axes.append(row_axes)

    # Loop through each row to populate the grid
    for row_idx, ax_row in enumerate(axes):
        # Add the original image to the first column
        ax_row[0].imshow(original_images[row_idx], cmap='gray')
        ax_row[0].axis('off')  # Hide axes for original images

        # Fill the remaining columns with other images
        start_idx = row_idx * 10
        for col_idx, ax in enumerate(ax_row[1:], start=1):
            img_idx = start_idx + col_idx - 1
            if img_idx < len(images):
                ax.imshow(images[img_idx], cmap='gray')
            ax.axis('off')  # Hide axes for all images

    # Draw a continuous vertical line
    line = plt.Line2D([0.12, 0.12], [0.02, 0.97], transform=fig.transFigure, color="black")
    fig.add_artist(line)

    # Add a caption for the first column
    fig.text(0.065, 0.97, "Originals", fontsize=16, fontweight='bold', ha='center', va='top')

    # Add titles for each epoch column
    for i, epoch in enumerate(epochs):
        x_position = 0.12 + (0.88 / len(epochs)) * (i + 0.5)  # Calculate normalized x-position for each title
        fig.text(x_position, 0.50, f"Epoch {epoch}", fontsize=10, ha='center', va='top')

    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add padding at the top for the title
    plt.subplots_adjust(wspace=0.3)  # Add spacing between subplots
    plt.savefig("./visualizations/train_images.jpg")

if __name__ == "__main__":
    generate_training_grid()