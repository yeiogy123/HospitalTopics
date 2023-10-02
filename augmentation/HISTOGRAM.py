import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
dir = os.path.dirname(os.path.realpath(__file__))
dir = dir + '\\valid'
print(dir)
all = os.listdir(dir)
num = 0
for i in all:
    num = num + 1
    image = cv2.imread(dir+'\\'+i ,cv2.IMREAD_GRAYSCALE)
    #hist = cv2.calcHist([image],[0],None,[256],[0,256])
    #plt.plot(hist)
    #plt.show()
    target = np.zeros((256,256,3))
    test = image
    test = np.power(test, 2.4)
    #plt.imshow(test)
    for a in range(256):
        for b in range(256):
            target[a,b,0] = math.sqrt(test[a,b])
            if image[a,b] <=255 and image[a,b] >= 58:
                image[a,b]=0
            target[a,b,1] = image[a,b]



    eq = cv2.equalizeHist(image)
    for a in range(256):
        for b in range(256):
            target[a,b,2] = eq[a,b]
    #hist = cv2.calcHist([eq],[0],None,[256],[0,256])
    #target = np.power(target, 1.25)
    #plt.imshow((target).astype('uint8'), )
    cv2.imwrite(dir+'\\'+i,target)
    #plt.axis('off')
    #plt.show()
    #print(dir+'\\'+i)
