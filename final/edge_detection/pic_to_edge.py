import cv2
import numpy as np
import sys, os
import scipy.misc

img = cv2.imread(sys.argv[1],0)
edges = cv2.Canny(img,100,200)

filename, file_extension = os.path.splitext(sys.argv[1])
scipy.misc.imsave('{0}_edge.bmp'.format(filename), edges)