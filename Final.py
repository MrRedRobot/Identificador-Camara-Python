# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 12:23:57 2019

"""
from cv2 import *

namedWindow("webcam")
vc = VideoCapture(0);

while True:
    next, frame = vc.read()
    imshow("webcam", frame)
    
    gray = cvtColor(frame, COLOR_BGR2GRAY)
    gauss = GaussianBlur(gray, (7,7), 1.5, 1.5)
    can = Canny(gauss, 0, 30, 3)
    
    cv2.imshow("filtro", can)
    
    if waitKey(50) >= 0:
        destroyAllWindows()
        break;
        