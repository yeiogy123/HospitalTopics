import SimpleITK as sitk

import glob

import numpy as np

from PIL import Image

import cv2

import matplotlib.pyplot as plt  # plt 用於顯示圖片

def save_array_as_nii_volume(data, filename, reference_name=None):

    img = sitk.GetImageFromArray(data)

    if (reference_name is not None):

        img_ref = sitk.ReadImage(reference_name)

        img.CopyInformation(img_ref)

    sitk.WriteImage(img, filename)



image_path = './image21'
image_arr = glob.glob(str(image_path) + str("/*"))



#print(image_arr, len(image_arr))

newspaceing=[1,1,1]
reader = sitk.ImageSeriesReader()
reader.SetFileNames(image_arr)
vol = reader.Execute()
vol.SetSpacing(newspaceing)
vol = sitk.GetArrayFromImage(vol)
print(vol.shape)
target = np.zeros([vol.shape[0], vol.shape[1], vol.shape[2]])
for a in range(vol.shape[0]):
    target[a] = cv2.cvtColor(vol[a], cv2.COLOR_RGB2GRAY)
sitk.WriteImage(target, 'AugDataWithoutN.nii')

