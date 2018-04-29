from PIL import Image
from os import listdir
from resizeimage import resizeimage
import os

input = 'validation-set/'
output = 'uc-128-v/'
shape = [128, 128]

if not os.path.exists(output):
    os.makedirs(output)

for folder_index, folder in enumerate(listdir(input)):
    for file_index,file in enumerate(listdir(input + folder)):

        with open(input + folder + '/' + file, 'r+b') as f:
            with Image.open(f) as image:
                cover = resizeimage.resize_cover(image, shape)
                if not os.path.exists(output+ folder):
                    os.makedirs(output + folder)
                cover.save(output + folder  + '/' +  str(file_index) + '.png', "PNG")