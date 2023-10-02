import os
import glob
import cv2
import numpy as np
nowpath = os.getcwd()
image = nowpath + '\image21\\'
mask = nowpath + '\mask10\\'
imageF = glob.glob(image+'*')
maskF = glob.glob(mask+'*')
number = 0
final = ''
tmp = False
a = 0
print('start')
for case in maskF:
    print(case)
    temp = cv2.imread(case, cv2.IMREAD_GRAYSCALE)
    print(np.max(temp))

    if np.max(temp)==0 :
        print('delete',case)
        case2 = case.replace('mask10','image21')
        print('delete', case2)
        os.remove(case)
        os.remove(case2)
    a = a + 1