import glob
import cv2
import numpy as np


class CameraCalibration:
    """
      Helper class for handling camera calibrations and removing distortion.
      E.g.
      c = CameraCalibration('camera_cal/calibration*.jpg')
      c.undistort(image)
      where 'camera_cal/calibration*.jpg' is a glob matching a set of images
      taken by the camera, with 9x6 corner chessboard patterns present in the
      images
    """
    _cal_mtx = None
    _cal_dist = None

    def __init__(self, cal_glob, chess_corners_x=9, chess_corners_y=6):
        x_shape, y_shape = None, None
        objpoints = []
        imgpoints = []
        ret = None
        images = glob.glob(cal_glob)
        objp = np.zeros((chess_corners_y*chess_corners_x, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chess_corners_x,
                               0:chess_corners_y].T.reshape(-1, 2)
        # for all admissible calibration images, accumulate corners
        for fname in images:
            img = cv2.imread(fname)
            # consider only images with the same width and height as the first
            if not x_shape:
                x_shape = img.shape[1]
                y_shape = img.shape[0]
            else:
                if (x_shape != img.shape[1] or y_shape != img.shape[0]):
                    pass
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (chess_corners_x,
                                                            chess_corners_y),
                                                     None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
        # infer mtx and dst from accumulated corners
        if len(imgpoints) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                               imgpoints,
                                                               (x_shape,
                                                                y_shape),
                                                               None, None)
        if ret:
            self._cal_mtx = mtx
            self._cal_dist = dist
        else:
            raise ValueError("Could not calibrate with images matching %s"
                             % cal_glob)

    def undistort(self, image):
        return cv2.undistort(image, self._cal_mtx, self._cal_dist, None,
                             self._cal_mtx)
