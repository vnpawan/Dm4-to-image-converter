import cv2
import numpy as np
import os
from tqdm import tqdm

def initial_filtering(image):
    # Apply a mild bilateral filter to reduce noise while preserving edges
    bilateral_filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return bilateral_filtered


def remove_noise_fft(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Create a mask with a high-pass filter
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 20  # Radius for the high-pass filter
    center = (ccol, crow)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= r * r
    mask[mask_area] = 0

    # Apply mask and inverse FFT
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Convert back to BGR
    img_back_bgr = cv2.cvtColor(np.uint8(img_back), cv2.COLOR_GRAY2BGR)

    return img_back_bgr


def enhance_local_contrast(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Convert the enhanced grayscale image back to BGR
    enhanced_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    return enhanced_image


def final_filtering(image):
    # Apply Gaussian blur
    gaussian_blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply median blur
    median_blurred = cv2.medianBlur(gaussian_blurred, 5)

    return median_blurred


def batch_process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read the image
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Initial filtering
            filtered_image = initial_filtering(image)

            # Remove noise using FFT
            # noise_removed_image = remove_noise_fft(filtered_image)

            # Enhance local contrast
            enhanced_image = enhance_local_contrast(filtered_image)

            # Final filtering
            # final_image = final_filtering(enhanced_image)

            # Save the segmented image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, enhanced_image)
            print(f"Processed {filename}")


input_directory = "/Users/vnpawan/Documents/CNT images - tiff inv_1024"
output_directory = '/Users/vnpawan/Documents/Output_SAM_comp input_1024'

batch_process_images(input_directory, output_directory)
