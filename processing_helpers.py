import numpy as np
import cv2


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def s_filter(in_img, s_thresh_min=80, s_thresh_max=255):
    hls = cv2.cvtColor(in_img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    return s_binary


def h_filter(in_img, h_thresh_min=15, h_thresh_max=100):
    hls = cv2.cvtColor(in_img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh_min) & (h_channel <= h_thresh_max)] = 1
    return h_binary


def sobel_x_filter(in_img, thresh_min=20, thresh_max=100):
    gray = cv2.cvtColor(in_img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary


def sobel_xs_filter(in_img, thresh_min=32, thresh_max=200):
    hls = cv2.cvtColor(in_img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary


def sobel_xh_filter(in_img, thresh_min=32, thresh_max=200):
    hls = cv2.cvtColor(in_img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    sobelx = cv2.Sobel(h_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
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
    if lcount > 0:
        lgrad /= lcount
        loffset /= lcount
        y1_left = img.shape[1]
        x1_left = int((y1_left * lgrad + loffset))
        x2_left = int(img.shape[1]*0.49)
        y2_left = int((x2_left - loffset) / lgrad)
        try:
            cv2.line(img, (x1_left, y1_left), (x2_left, y2_left), color,
                     thickness)
        except (Exception):
            pass
    if rcount > 0:
        rgrad /= rcount
        roffset /= rcount
        y1_right = img.shape[1]
        x1_right = int((y1_right * rgrad + roffset))
        x2_right = int(img.shape[1]*0.51)
        y2_right = int((x2_right - roffset) / rgrad)
        try:
            cv2.line(img, (x1_right, y1_right), (x2_right, y2_right), color,
                     thickness)
        except (Exception):
            pass
