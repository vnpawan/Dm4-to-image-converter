import ncempy.io.dm as dm
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd

input_folder = "/Users/vnpawan/Documents/focal series/TF03"
output_folder = "/Users/vnpawan/Documents/Focal series - tiff images/TF03"
os.makedirs(output_folder)

all_files_list = os.listdir(input_folder)
df = []
for file in tqdm(all_files_list):
    if file.lower().endswith('.dm4'):
        img_path = os.path.join(input_folder, file)
        aa = dm.dmReader(img_path)
        dat = aa['data']
        if aa['pixelUnit'][0] == 'nm':
            pxval = round(aa['pixelSize'][0].astype(float), 3)
        else:
            pxval = round(aa['pixelSize'][0].astype(float) * 1000, 3)
        output_filename = file.split('.')[0] + '_px=' + str(pxval) + '.tiff'
        output_path = os.path.join(output_folder, output_filename)
        plt.figure()
        plt.imshow(dat, cmap='Grays')
        # plt.show(block=True)
        plt.imsave(output_path, dat, cmap='gray')

        # ab = dm.fileDM(img_path)
        # bb = ab.getMetadata(0).keys()
        # bc = ab.getMetadata(0).values()
        # new = pd.DataFrame([list(bb), list(bc)]).T
        # act_mag = new.loc[new[0] == 'Microscope Info Formatted Actual Mag'][1].values[0]
        # def_mag = new.loc[new[0] == 'Microscope Info Formatted Indicated Mag'][1].values[0]
        # df.append([file, act_mag, def_mag, aa['pixelUnit'][0], aa['pixelSize'][0]])

# new1 = pd.DataFrame(df)
# new1.columns = ['File_name', 'Actual magnification', ' Indicated magnification', 'Pixel Unit', 'Pixel Val']
# new1.to_csv(os.path.join(output_folder, 'Magnification information_batch1.csv'), index=True, header=True)
