import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
from processing_helpers import *


class BirdsEyeTransform:
    """
      Helper class for handling birds' eye view transform of images.
      Infer a transform t using "nice" images of straight road maching a glob:
      t = BirdsEyeTransform('straight_lines*.jpg')
      Use this transform to transform images into birds' eye view:
      transformed = t.transform(image)
      Obtain the meters-per-pixel value for transformed images:
      mpp = t.meters_per_pixel
    """
    DASHED_LANE_LENGTH = 3.0  # meters
    LANE_WIDTH = 3.7  # meters

    meters_per_pixel = None
    _m = None
    _m_inv = None

    def __init__(self, camera_calibration, straight_glob):
        out_dir = "intermediate/"
        rho = 6
        theta = np.pi/180
        threshold = 200
        min_line_len = 300
        max_line_gap = 120
        image_fnames = glob.glob(straight_glob)
        left_arr, right_arr = [], []
        binary_combos = {}
        for fname in image_fnames:
            image = mpimg.imread(fname)
            image = camera_calibration.undistort(image)
            binary_sobelx = sobel_x_filter(image)
            binary_s = s_filter(image)
            binary_combo = np.bitwise_or(binary_sobelx, binary_s)
            imshape = image.shape
            vertices = np.array([[(0, imshape[0]),
                                  (0.5*imshape[1], 0.59*imshape[0]),
                                  (0.5*imshape[1], 0.59*imshape[0]),
                                  (imshape[1], imshape[0])]], dtype=np.int32)

            # Visualize filtered points from undistorted image
            color_binary = np.dstack(( np.zeros_like(binary_sobelx),
                                      binary_sobelx, binary_s)) * 255
            color_binary = region_of_interest(color_binary, vertices)
            mpimg.imsave(out_dir + fname.split('/')[-1][:-4] + 
                         "_t_filtered.jpg", color_binary)

            binary_combo = region_of_interest(binary_combo, vertices)
            lines = cv2.HoughLinesP(binary_combo, rho, theta, threshold,
                                    np.array([]),
                                    minLineLength=min_line_len,
                                    maxLineGap=max_line_gap)
            # Visualize lines on undistorted image
            draw_lines(image, lines)
            mpimg.imsave(out_dir + fname.split('/')[-1][:-4] +
                         "_t_filtered_lined.jpg", image)

            # calculate transform
            i_left, i_right = lines_to_left_right(lines, imshape)
            left_arr.append(i_left)
            right_arr.append(i_right)
            binary_combos[fname] = binary_combo
        left = ((np.mean([e[0][0] for e in left_arr]),
                 np.mean([e[0][1] for e in left_arr])),
                (np.mean([e[1][0] for e in left_arr]),
                 np.mean([e[1][1] for e in left_arr])))
        right = ((np.mean([e[0][0] for e in right_arr]),
                 np.mean([e[0][1] for e in right_arr])),
                 (np.mean([e[1][0] for e in right_arr]),
                 np.mean([e[1][1] for e in right_arr])))
        src = np.float32([right[1], right[0], left[0], left[1]])
        dst = np.float32([[0.6*imshape[1], 0], [0.6*imshape[1], imshape[0]],
                          [0.4*imshape[1], imshape[0]], [0.4*imshape[1], 0]])
        self._m = cv2.getPerspectiveTransform(src, dst)
        self._m_inv = cv2.getPerspectiveTransform(dst, src)
        # second pass - meters per pixels, lateral scaling
        # how many pixels long is a dashed line?
        transformed_left = (self.__transform_point(left[0]),
                            self.__transform_point(left[1]))
        transformed_right = (self.__transform_point(right[0]),
                             self.__transform_point(right[1]))
        meters_per_pixel_arr = []
        for fname in image_fnames:
            transformed_binary_combo = self.transform(binary_combos[fname])
            # self.meters_per_pixel = 0.0416
            meters_per_pixel_arr.append(BirdsEyeTransform.DASHED_LANE_LENGTH /
                                        dash_pixel_length(
                                            transformed_binary_combo,
                                            transformed_left,
                                            transformed_right))
            # Visualize transformed straight lane image
            mpimg.imsave(out_dir + fname.split('/')[-1][:-4] +
                         "_t_filtered_transformed.jpg",
                         transformed_binary_combo)
            transformed = self.transform(image)
            mpimg.imsave(out_dir + fname.split('/')[-1][:-4] +
                         "_t_filtered_lined_transformed.jpg",
                         transformed)

        self.meters_per_pixel = np.mean(meters_per_pixel_arr)
        x_offset = 0.5*BirdsEyeTransform.LANE_WIDTH/self.meters_per_pixel
        xr = 0.5*imshape[1] + x_offset
        xl = 0.5*imshape[1] - x_offset
        dst = np.float32([[xr, 0], [xr, imshape[0]], [xl, imshape[0]],
                          [xl, 0]])
        self._m = cv2.getPerspectiveTransform(src, dst)
        self._m_inv = cv2.getPerspectiveTransform(dst, src)

    def transform(self, image):
        img_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self._m, img_size,
                                   flags=cv2.INTER_AREA)

    def untransform(self, image):
        img_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self._m_inv, img_size,
                                   flags=cv2.INTER_AREA)

    def __transform_point(self, point):
        point = np.asarray([point[0], point[1], 1.0])
        point_t = np.matmul(self._m, point)
        point_t = point_t / point_t[2]
        return point_t[0:2]


def lines_to_left_right(lines, shape):
    lcount, rcount, lgrad, rgrad, loffset, roffset = (0, 0, 0, 0, 0, 0)
    for line in lines:
        for x1, y1, x2, y2 in line:
            try:
                grad = (x2-x1)/(y2-y1)
            except (Exception):
                pass
            offset = x1 - grad * y1
            if grad < -0.2 and grad > -5 and offset < 1400:
                # left lane
                lcount += 1
                lgrad += grad
                loffset += offset
            elif grad > 0.2 and grad < 5 and offset > -200:
                # right lane
                rcount += 1
                rgrad += grad
                roffset += offset
    left, right = None, None
    if lcount > 0:
        lgrad /= lcount
        loffset /= lcount
        y1_left = shape[0]
        x1_left = int((y1_left * lgrad + loffset))
        x2_left = int(shape[1]*0.47)
        y2_left = int((x2_left - loffset) / lgrad)
        left = ((x1_left, y1_left), (x2_left, y2_left))
    if rcount > 0:
        rgrad /= rcount
        roffset /= rcount
        y1_right = shape[0]
        x1_right = int((y1_right * rgrad + roffset))
        x2_right = int(shape[1]*0.53)
        y2_right = int((x2_right - roffset) / rgrad)
        right = (x1_right, y1_right), (x2_right, y2_right)
    return left, right


def dash_pixel_length(binary_image, left_line, right_line,
                      line_dist_threshold=8, distinct_dash_threshold=3):
    nonzero = binary_image.nonzero()
    left_y = []
    right_y = []
    for i in range(len(nonzero[0])):
        p = (nonzero[1][i], nonzero[0][i])
        if (distance_line_to_point(left_line[0], left_line[1], p) <
                line_dist_threshold):
            left_y.append(p[1])
        if (distance_line_to_point(right_line[0], right_line[1], p) <
                line_dist_threshold):
            right_y.append(p[1])
    left_y.sort(reverse=True)
    right_y.sort(reverse=True)
    dash_length = binary_image.shape[1]
    for y_vals in [left_y, right_y]:
        lengths = []
        last_y = y_vals[0]
        current_length = 0
        for y in y_vals:
            if (last_y - y) > distinct_dash_threshold:
                lengths.append(current_length)
                current_length = 0
                last_y = y
            else:
                current_length += (last_y - y)
                last_y = y
        lengths.append(current_length)
        dash_length = min(dash_length, [l for l in lengths if l > 30][0])
    return dash_length


def distance_line_to_point(lp1, lp2, p):
    lp1 = np.asarray(lp1)
    lp2 = np.asarray(lp2)
    p = np.asarray(p)
    return np.linalg.norm(np.cross(lp2-lp1, lp1-p))/np.linalg.norm(lp2-lp1)
