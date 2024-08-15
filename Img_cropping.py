from PIL import Image
import os

# Function to crop images
def crop_images(input_folder, output_folder, crop_size):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in input folder
    files = os.listdir(input_folder)

    for file in files:
        # Skip non-image files
        if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            continue

        # Open each image
        img_path = os.path.join(input_folder, file)
        try:
            img = Image.open(img_path)
        except (IOError, OSError) as e:
            print(f"Error opening image {img_path}: {e}")
            continue

        # Get image dimensions
        width, height = img.size

        # Calculate number of full crops in each dimension
        num_crops_width = width // crop_size
        num_crops_height = height // crop_size

        # Crop and save each segment
        for i in range(num_crops_width):
            for j in range(num_crops_height):
                left = i * crop_size
                upper = j * crop_size
                right = left + crop_size
                lower = upper + crop_size

                # Crop image segment
                cropped_img = img.crop((left, upper, right, lower))

                # Save cropped image with a systematic filename
                output_filename = f"{file.split('.')[0]}_{i}_{j}.png"
                output_path = os.path.join(output_folder, output_filename)
                cropped_img.save(output_path)

    print("Image cropping complete.")


if __name__ == "__main__":
    # Example usage:
    input_folder = '/Users/vnpawan/Documents/CNT images_batch 1 - tiff conversion'  # Change this to your input folder containing 4096x4096 images
    output_folder = '/Users/vnpawan/Documents/CNT images - tiff inv_1024'  # Change this to where you want to save cropped 256x256 images
    crop_size = 1024  # Size of each cropped image

    crop_images(input_folder, output_folder, crop_size)