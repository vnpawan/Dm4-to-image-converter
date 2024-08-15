These codes can be used to convert dm4 files into suitable input for the SAM segmentation algorithm.
1. dm4_conv.py - Convert dm4 to tiff/png/jpeg images. Color map options in the plt.imsave command can be 'grays' (default) or 'binary' if you prefer different contrast.
2. Img_cropping.py - This file is used to crop the existing 4096 by 4096 images into smaller crops of specified sizes.
3. EC_image analysis.py - This code is used for adding a bilateral filter and enhancing the contrast. There are submodules for FFT based noise removal but it is deemed unnecessary at this moment. In future, line number 80 and 86 can be uncommented to use these features for image cleanup.
