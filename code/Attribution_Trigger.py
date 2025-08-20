import numpy as np
from PIL import Image
import os

def add_backdoor_trigger(img_path, output_path, amplitude=20):
    """
    Adds a low-visibility backdoor trigger to a color image (based on amplitude perturbation).
    :param img_path: Path to the input image.
    :param output_path: Path to save the output image.
    :param amplitude: The perturbation amplitude (controls trigger visibility, 5-30 recommended).
    """
    # Open image, ensure it's RGB
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img, dtype=np.int32)  # Use int32 to avoid overflow during addition/subtraction
    h, w = img_array.shape[:2]

    # Copy the original image for modification
    backdoored = img_array.copy()

    # Define trigger coordinates (consistent with original BadNets, but with perturbation instead of replacement)
    black_coords = [
        (h-3, w-3), (h-3, w-2), 
        (h-2, w-3), (h-2, w-1), 
        (h-1, w-2)
    ]
    white_coords = [
        (h-3, w-1), (h-2, w-2), 
        (h-1, w-3), (h-1, w-1)
    ]

    # Safety check: ensure coordinates are within image bounds
    valid_black = [(y, x) for y, x in black_coords if 0 <= y < h and 0 <= x < w]
    valid_white = [(y, x) for y, x in white_coords if 0 <= y < h and 0 <= x < w]

    # For "black" coordinates: subtract amplitude from original pixel values (decrease brightness)
    for y, x in valid_black:
        backdoored[y, x, :] = np.clip(backdoored[y, x, :] - amplitude, 0, 255)

    # For "white" coordinates: add amplitude to original pixel values (increase brightness)
    for y, x in valid_white:
        backdoored[y, x, :] = np.clip(backdoored[y, x, :] + amplitude, 0, 255)

    # Convert back to uint8 and save
    backdoored = backdoored.astype(np.uint8)
    backdoored_img = Image.fromarray(backdoored)
    backdoored_img.save(output_path)
    print(f"Low-visibility backdoored image saved to: {output_path}")

    return output_path

def add_corners_backdoor_trigger(img_path, output_path, amplitude=20):
    """
    Adds a low-visibility backdoor trigger to the four corners of an image.
    :param img_path: Path to the input image.
    :param output_path: Path to save the output image.
    :param amplitude: The perturbation amplitude (controls trigger visibility, 5-30 recommended).
    """
    # Open image, ensure it's RGB
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img, dtype=np.int32)
    h, w = img_array.shape[:2]

    # Copy original image for modification
    backdoored = img_array.copy()

    # Trigger coordinates for the four corners
    corners = {
        "top_left": {
            "black": [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)],
            "white": [(0, 0), (1, 1), (2, 0), (2, 2)]
        },
        "top_right": {
            "black": [(0, w-3), (0, w-2), (1, w-1), (1, w-3), (2, w-2)],
            "white": [(0, w-1), (1, w-2), (2, w-3), (2, w-1)]
        },
        "bottom_left": {
            "black": [(h-3, 1), (h-3, 2), (h-2, 0), (h-2, 2), (h-1, 1)],
            "white": [(h-3, 0), (h-2, 1), (h-1, 0), (h-1, 2)]
        },
        "bottom_right": {
            "black": [(h-3, w-3), (h-3, w-2), (h-2, w-3), (h-2, w-1), (h-1, w-2)],
            "white": [(h-3, w-1), (h-2, w-2), (h-1, w-3), (h-1, w-1)]
        }
    }

    # Add trigger to each corner
    for corner in corners.values():
        # Process "black" coordinates
        for y, x in corner["black"]:
            if 0 <= y < h and 0 <= x < w:
                backdoored[y, x, :] = np.clip(backdoored[y, x, :] - amplitude, 0, 255)

        # Process "white" coordinates
        for y, x in corner["white"]:
            if 0 <= y < h and 0 <= x < w:
                backdoored[y, x, :] = np.clip(backdoored[y, x, :] + amplitude, 0, 255)

    # Convert back to uint8 and save
    backdoored = backdoored.astype(np.uint8)
    backdoored_img = Image.fromarray(backdoored)
    backdoored_img.save(output_path)
    print(f"Four-corner low-visibility backdoored image saved to: {output_path}")

    return output_path

def block_enlarge(image_path, target_size=1466):
    """
    Enlarges an image using a block-based method to avoid black borders.
    """
    original_image = Image.open(image_path)
    mode = original_image.mode
    print(f"Loading image: {image_path}")
    print(f"Image mode: {mode}")

    original_np = np.array(original_image, dtype=np.uint8)
    original_h, original_w = original_np.shape[:2]

    # Calculate block size for each dimension
    block_h = target_size // original_h
    remaining_h = target_size % original_h
    block_w = target_size // original_w
    remaining_w = target_size % original_w

    # Initialize the enlarged image
    if mode == 'L':
        enlarged_image = np.zeros((target_size, target_size), dtype=np.uint8)
    elif mode == 'RGB':
        enlarged_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported image mode: {mode}. Only L (grayscale) and RGB (color) are supported.")

    # Iterate through each pixel of the original image and enlarge it into a block
    current_y = 0
    for i in range(original_h):
        h_size = block_h + 1 if i < remaining_h else block_h
        current_x = 0
        for j in range(original_w):
            w_size = block_w + 1 if j < remaining_w else block_w
            y_end = current_y + h_size
            x_end = current_x + w_size

            if mode == 'L':
                enlarged_image[current_y:y_end, current_x:x_end] = original_np[i, j]
            else:
                enlarged_image[current_y:y_end, current_x:x_end, :] = original_np[i, j, :]

            current_x = x_end
        current_y = y_end

    # Convert to image and save
    result = Image.fromarray(enlarged_image, mode)
    dir_name = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    new_filename = f"{base_name}_block_{target_size}px.png"
    save_path = os.path.join(dir_name, new_filename)
    result.save(save_path)

    print(f"Original dimensions: {original_image.size}")
    print(f"Enlarged dimensions: {result.size}")
    print(f"Saved to: {save_path}")
    return save_path

if __name__ == "__main__":
    # Define paths and parameters
    img_path = './xie/FNN_Shapley/test/images/1/random_90_label_6.png'  # Input image path
    output_path = './xie/FNN_Shapley/test/images/1/random_90_label_6_four128.png'  # Output path
    trigger_amplitude = 128  # Trigger amplitude

    # Add four-corner low-visibility backdoor trigger
    saved_image_path = add_corners_backdoor_trigger(img_path, output_path, amplitude=trigger_amplitude)
    
    # Enlarge the image using block-based method
    block_enlarge(saved_image_path, target_size=1466)