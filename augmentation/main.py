#READ OT FILE
import numpy as np

import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import imageio  # 转换成图像
from PIL import Image
import cv2
np.set_printoptions(threshold=np.inf)


def nii_to_image(niifile):
    return 0


filepath = './'  # 读取本代码同个文件夹下所有的nii格式的文件
filenames = os.listdir(filepath)
imgfile = './'

slice_trans = []

for f in filenames:  # 开始读取nii文件
    s = f[-4:]
    print(s)

    if s != '.nii':
        continue
    s1 = f[:-4]
    print(s1)
    imgfile_path = imgfile + s1
    print("imgfile_path:" + imgfile_path)
    img_path = os.path.join(filepath, f)
    img = nib.load(img_path)  # 读取nii
    print("img:")
    print(img)
    img_fdata = img.get_fdata()

    fname = f.replace('.nii', '')  # 去掉nii的后缀名
    img_f_path = os.path.join(imgfile, fname)
    if not os.path.exists(img_f_path):
        os.mkdir(img_f_path)

    # 创建nii对应的图像的文件夹
    # # if not os.path.exists(img_f_path):
    # os.mkdir(img_f_path) #新建文件夹
    # #开始转换为图像
    if '.gz' in s1:
        (x, y, z, _) = img.shape
        print("img2:")
        print(img.shape)
    else:
        (x, y, z) = img.shape
        print("img3:")
        print(img.shape)

    for i in range(z):  # z是图像的序列
        from skimage import img_as_ubyte
        silce = img_fdata[:, :, i]  # 选择哪个方向的切片都可以
        print(img_fdata.dtype)
        print(silce)
        #import cv2
        #img = cv2.normalize(src=silce, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
        #print(i, img.dtype)
        imageio.imwrite(os.path.join(img_f_path, '{}_mask.png'.format(i)),silce)
        img = Image.open(os.path.join(img_f_path, '{}_mask.png'.format(i)))

        img.save(os.path.join(img_f_path, '{}_mask.png'.format(i)))