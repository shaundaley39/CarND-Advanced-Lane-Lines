#!/usr/bin/env python3

from camera_calibration import *
import os
import matplotlib.image as mpimg
import glob


test_glob = "test_images/*.jpg"
cal_glob = 'camera_cal/calibration*.jpg'
out_dir = "intermediate/"

c = CameraCalibration(cal_glob)

for fname in (glob.glob(test_glob) + glob.glob(cal_glob)):
    in_img = mpimg.imread(fname)
    undistorted = c.undistort(in_img)
    mpimg.imsave(out_dir + fname.split('/')[-1][:-4] + "_undistorted.jpg",
                 undistorted)
