import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
import timeit
from matplotlib.colors import ListedColormap
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, disk
from scipy.ndimage import binary_fill_holes
import glob
import ncempy.io.dm as dm


# image_path = "/Users/vnpawan/Documents/CNT images - Shared by Bob/CNT images_batch 1_BF/OneView 1279_BF_pxsize=0.031738978pxU=nm.tiff"
def dm4_converter(dm4_path, out_folder_path):
    aa = dm.dmReader(dm4_path)
    dat = aa['data']
    dat[dat < 0] = 0
    bw_ar = np.divide(dat, np.max(dat) / 255)
    plt.imsave(out_folder_path + dm4_path.replace(dm4_path[:71], '')[:-4] + '_BF_pxsize=' + str(
        aa['pixelSize'][0]) + 'pxU=' + str(aa['pixelUnit'][0]) + '.tiff', bw_ar)


def convert_rgb_to_binary_with_watershed(image_path, min_object_size, output_path):
    # start_time = timeit.default_timer()
    start_time = timeit.default_timer()
    # Open the image
    image = cv2.imread(image_path)

    # Convert to RGB if needed
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Apply a bilateral filter to the image
    bilateral_filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
    gray_image[gray_image < 0] = 0
    # Apply a binary threshold to the grayscale image
    _, binary_image = cv2.threshold(gray_image, np.mean(gray_image), 255, cv2.THRESH_BINARY)
    # Fill holes in the binary image
    filled_image = binary_fill_holes(binary_image).astype(np.uint8) * 255

    # Compute the distance transform
    distance_transform = ndi.distance_transform_edt(filled_image)

    # Find local maxima in the distance transform
    local_maxi = peak_local_max(distance_transform, footprint=np.ones((3, 3)), labels=filled_image)

    # Create markers for the watershed algorithm
    markers = np.zeros_like(filled_image, dtype=int)
    markers[tuple(local_maxi.T)] = np.arange(1, local_maxi.shape[0] + 1)

    # Apply the watershed algorithm using skimage
    labels = watershed(-distance_transform, markers, mask=filled_image)

    # Get region properties
    regions = regionprops(labels)

    # Create a binary mask for objects larger than min_object_size pixels
    mask = np.zeros_like(labels)
    for region in regions:
        if region.area > min_object_size:
            mask[labels == region.label] = region.label

    # Create a colormap with different colors for each label
    cmap = plt.cm.tab20
    new_cmap = ListedColormap(cmap(np.linspace(0, 1, np.max(mask) + 1)))

    # Display the binary image with watershed applied using different colors for each label
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap=new_cmap, interpolation='none')
    plt.colorbar(ticks=[])
    plt.title('Watershed Segmentation Result with Filled Holes')
    plt.axis('off')
    # plt.show(block=True)

    plt.imsave(output_path + image_path[75:-5] + '_wat_a1.tiff',mask)

    end_time = timeit.default_timer()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":

    im_folder = glob.glob('/Users/vnpawan/Documents/CNT images - Shared by Bob/CNT images_batch 1_BF2/*.tiff')
    for i in range(len(im_folder)):
        convert_rgb_to_binary_with_watershed(im_folder[i], 150,
                                             '/Users/vnpawan/Documents/CNT images - Shared by Bob/CNT images_batch 1_BF2/Watershed_attempt1/')

    # all_files_list = glob.glob("/Users/vnpawan/Documents/CNT images - Shared by Bob/CNT images_batch 1/*.dm4")
    # for i in range(len(all_files_list)):
    #     dm4_converter(all_files_list[i],out_folder)
