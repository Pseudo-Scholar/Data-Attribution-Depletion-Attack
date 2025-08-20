import numpy as np
from PIL import Image
import os

def block_enlarge(image_path, target_size=1466):
    # Read image, get mode and dimensions
    original_image = Image.open(image_path)
    mode = original_image.mode
    original_h, original_w = original_image.size[1], original_image.size[0]  # height, width (PIL size is (w,h))
    print(f"Loading image: {image_path}")
    print(f"Image mode: {mode}, Original size: (width={original_w}, height={original_h})")

    # Convert to numpy array
    original_np = np.array(original_image, dtype=np.uint8)

    # Calculate block size and remaining pixels for height
    block_h = target_size // original_h  # Base block height
    remaining_h = target_size % original_h  # Remaining pixels in height dimension
    # Calculate block size and remaining pixels for width
    block_w = target_size // original_w  # Base block width
    remaining_w = target_size % original_w  # Remaining pixels in width dimension

    # Initialize the enlarged image (create array with corresponding dimensions based on mode)
    if mode == 'L':  # Grayscale image (height, width)
        enlarged_image = np.zeros((target_size, target_size), dtype=np.uint8)
    elif mode == 'RGB':  # Color image (height, width, 3)
        enlarged_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported image mode: {mode}. Only L (grayscale) and RGB (color) are supported.")

    # Iterate through each pixel of the original image, enlarging it into a block (handling remaining pixels to avoid black borders)
    current_y = 0  # Current filled height position
    for i in range(original_h):
        # Height dimension: the first 'remaining_h' blocks are one pixel taller to consume remaining height
        h_size = block_h + 1 if i < remaining_h else block_h
        current_x = 0  # Current filled width position
        for j in range(original_w):
            # Width dimension: the first 'remaining_w' blocks are one pixel wider to consume remaining width
            w_size = block_w + 1 if j < remaining_w else block_w

            # Calculate the end coordinates of the current block
            y_end = current_y + h_size
            x_end = current_x + w_size

            # Fill the block (select corresponding dimensions based on image mode)
            if mode == 'L':
                enlarged_image[current_y:y_end, current_x:x_end] = original_np[i, j]
            else:  # RGB
                enlarged_image[current_y:y_end, current_x:x_end, :] = original_np[i, j, :]

            current_x = x_end  # Update width fill position
        current_y = y_end  # Update height fill position

    # Save the enlarged image
    dir_name = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    new_filename = f"{base_name}_block_{target_size}px.png"
    save_path = os.path.join(dir_name, new_filename)
    result = Image.fromarray(enlarged_image, mode)
    result.save(save_path)

    print(f"Original size: {original_image.size} (width x height)")
    print(f"Enlarged size: {result.size} (width x height)")
    print(f"Saved to: {save_path}")

# Example usage
if __name__ == "__main__":
    img_path = './xie/FNN_Shapley/test/images/3/sample_37.png'  # Your image path
    block_enlarge(img_path, target_size=1466)