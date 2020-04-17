#!/usr/bin/env python3

from camera_calibration import *
from birdseye_transform import *
import os
import matplotlib.image as mpimg
import glob

test_glob = 'test_images/*.jpg'
out_dir = 'intermediate/'
c = CameraCalibration('camera_cal/calibration*.jpg')
t = BirdsEyeTransform(c, 'test_images/straight_lines*jpg')
print("Meters per pixel: %s" % str(t.meters_per_pixel))

for fname in glob.glob(test_glob):
    in_img = mpimg.imread(fname)
    name = fname.split('/')[-1][:-4]
    undistorted = c.undistort(in_img)
    mpimg.imsave(out_dir + name + "_undistorted.jpg", undistorted)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    transformed = t.transform(gray)
    mpimg.imsave(out_dir + name + "_transformed.jpg", transformed)
    color_transformed = t.transform(undistorted)
    mpimg.imsave(out_dir + name + "_ctransformed.jpg", color_transformed)
