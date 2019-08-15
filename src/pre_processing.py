from PIL import Image
from tqdm import tqdm
import os

 

def check_and_remove_corrupt_images(dataset):
    sku_folders = [dataset+'/'+folder for folder in os.listdir(dataset)]
    all_image_paths = []
    for folder in sku_folders:
        all_image_paths.extend([folder+'/'+image for image in os.listdir(folder)])
    corrupt_image_paths = []
    for i in tqdm(range(len(all_image_paths))):
        filename = all_image_paths[i]
        if filename.endswith('.png') or filename.endswith('.jpg'):
            try:
                img = Image.open(filename) # open the image file
                img.verify() # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                print('Bad file:', filename)# print out the names of corrupt files
                corrupt_image_paths.append(filename)
                os.remove(filename)
                continue
    return corrupt_image_paths

s = check_and_remove_corrupt_images('/home/caffe/data/chinadrink_prod_train/')
print(len(s))
