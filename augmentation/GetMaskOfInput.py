CBF_File = './SMIR.Brain.XX.O.CT_CBF.345563'
CBV_File = './SMIR.Brain.XX.O.CT_CBV.345564'
MTT_File = './SMIR.Brain.XX.O.CT_MTT.345565'
Tmax_File = './SMIR.Brain.XX.O.CT_Tmax.345567'
CT_File = './SMIR.Brain.XX.O.CT.345562'
import glob
import os
from PIL import Image
import numpy
import cv2
CBF = glob.glob(os.path.join(CBF_File, "*"))
CBV = glob.glob(os.path.join(CBV_File, "*"))
MTT = glob.glob(os.path.join(MTT_File, "*"))
Tmax = glob.glob(os.path.join(Tmax_File, "*"))
CT = glob.glob(os.path.join(CT_File, "*"))
Mask = []
for i in range(len(CBF)):
    cbf = numpy.array(Image.open(CBF[i]).convert('L'))
    cbv = numpy.array(Image.open(CBV[i]))
    print(cbv.shape)
    mtt = numpy.array(Image.open(MTT[i]).convert('L'))
    tmax = numpy.array(Image.open(Tmax[i]).convert('L'))
    ct = numpy.array(Image.open(CT[i]).convert('L'))
    temp = numpy.add(cbf,cbv)
    temp = numpy.add(temp, mtt)
    temp = numpy.add(temp,tmax)
    temp = numpy.reshape(temp, -1)
    ct = numpy.reshape(ct, -1)
    temp = numpy.multiply(temp, ct)
    temp = numpy.reshape(temp,(256,256))
    Mask.append(temp)
print(len(Mask))
MaskImage = []
for i in range(len(Mask)):
    MaskImage = Image.fromarray(Mask[i])
    MaskImage.save(str(i)+"_mask.png", "png")
