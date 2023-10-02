import SimpleITK as sitk
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
import glob
import imageio
def showNii(num, img, label):
    a = num
    tmp_path='C:\\Users\\User\\Desktop\\Isles2018\\ISLES2018_Testing\\original\\'
    tmp2_path = 'C:\\Users\\User\\Desktop\\Isles2018\\ISLES2018_Testing\\Skull-stripped\\'
    print(img.shape[0])

    for i in range(img.shape[0]):
        a=a+1
        #plt.imshow(img[ i, :, :], cmap='gray')
        #plt.axis('off')
        #plt.savefig(str(tmp_path)+str(a)+'.png', bbox_inches='tight', pad_inches=0)
        #plt.close()
        imageio.imwrite(str(tmp_path)+str(a)+'.png', img[i,:,:])
        print(str(tmp_path)+str(a)+'.png')
    a = num
    l = True
    if l == True:
        for i in range(label.shape[0]):
            a =a+ 1
            #plt.imshow(label[i ,:,: ], cmap='gray')
            #plt.axis('off')
            #print(np.max(label[i,:,:]))
            #plt.savefig(str(tmp2_path)+str(a)+'.png', bbox_inches='tight', pad_inches=0)
            imageio.imwrite(str(tmp2_path) + str(a) + '.png', label[i, :, :])
            print(str(tmp2_path) + str(a) + '.png')
            #plt.close()
    print("now a =", a)
    return a

def findF(file):
    F = glob.glob(str(file) + str("\\*"))
    b = []
    for a in F:
        if a.find("nii") > 0:
            b = a

    return b

def transform2Img(num,Tmax, CBF, CBV, MTT, CT, OT):
    print(num)
    #print(Tmax)
    #print(CBF)
    #print(CBV)
    #print(MTT)
    print(CT)
    #print(OT)
    itk_img = sitk.ReadImage(Tmax)
    itk_img1 = sitk.ReadImage(CBF)
    itk_img2 = sitk.ReadImage(CBV)
    itk_img3 = sitk.ReadImage(MTT)
    itk_img4 = sitk.ReadImage(CT)
    itk_img5 = sitk.ReadImage(OT)

    #print(itk_img.GetPixelIDTypeAsString())
    #print(itk_img2.GetPixelIDTypeAsString())
    #print(itk_img3.GetPixelIDTypeAsString())
    #print(itk_img1.GetPixelIDTypeAsString())
    print(itk_img4.GetPixelIDTypeAsString())

    img1 = sitk.GetArrayFromImage(itk_img1)
    img2 = sitk.GetArrayFromImage(itk_img2)
    img3 = sitk.GetArrayFromImage(itk_img3)
    img = sitk.GetArrayFromImage(itk_img)
    img4 = sitk.GetArrayFromImage(itk_img4)
    img5 = sitk.GetArrayFromImage(itk_img5)

    print(img4.shape)
    #print(np.max(img))
    #print(img.shape)  # (155, 240, 240) 表示各個維度的切片數量
    #print(img2.shape)
    #print(img3.shape)
    #print(img1.shape)
    #print(img)
    tt = img1 + img2
    tt = tt + img3
    tt = tt + img
    #tt = (((tt - np.min(tt)) / (np.max(tt) - np.min(tt))) * 256).astype('uint8')


    #for k in range(tt.shape[0]):
     #   for i in range(256):
      #      for j in range(256):
       #         if tt[k, i, j] != 0:
        #            tt[k, i, j] = img4[k, i, j]
                    #tt[k,i,j]=255
         #       else:
          #          tt[k, i, j] = 0
    #print("tt=",tt.shape[0])
    #print(tt.dtype)
    #print(tt.dtype)
    #print(np.max(img5))
    #num = showNii(num, img4 ,tt)
    #return num
    # print([1])
    # print(img.data)
    #print('MAX',np.max(img4))
    return img5

path = 'C:\\Users\\User\\Desktop\\Isles2018\\ISLES2018_Training\\TRAINING'
os.chdir(path)
training_file = glob.glob(str(path)+str("\\*"))
print(training_file)
a = 0
target = np.empty([502,256,256])
number = 0
for case in training_file:
    now_path = glob.glob(str(case)+str("\\*"))
    print(now_path)
    Tmax =[]
    CBF=[]
    CBV=[]
    MTT=[]
    CT=[]
    OT=[]
    for file in now_path:
        if file.find("Tmax") > 0 :
            Tmax = findF(file)
        elif file.find("CBF") > 0:
            CBF = findF(file)
        elif file.find("CBV") > 0:
            CBV = findF(file)
        elif file.find("MTT") > 0:
            MTT = findF(file)
        elif file.find("O.CT.") > 0:
            CT = findF(file)
        elif file.find("OT") > 0:
            OT = findF(file)
    a = transform2Img(a,Tmax, CBF, CBV, MTT, CT, OT)
    print(a.shape[0])
    for index in range(a.shape[0]):
        print(a[index].shape)
        target[number]=a[index]
        number+=1
        print(target)
print(number)
print(target.shape)
print(np.max(target))
target = sitk.GetImageFromArray(target)
sitk.WriteImage(target, 'input_mask.nii.gz')
