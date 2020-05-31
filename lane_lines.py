import cv2
import numpy as np


class LaneLines:
    _lanes_found = False
    _meters_per_pixel = None

    def __init__(self, meters_per_pixel):
        self._meters_per_pixel = meters_per_pixel

    def find_lane_pixels(self, binary_warped):
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        offset = 44
        midpoint = binary_warped.shape[1]//2
        leftx_base = midpoint - offset
        rightx_base = midpoint + offset

        # HYPERPARAMETERS
        # number of sliding windows
        nwindows = 9
        # width of windows set to +/- margin
        margin = 16
        # minimum number of pixels found to recenter window
        minpix = 25

        # height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        stop_left = 0
        stop_right = 0
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) &
                              (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) &
                               (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # recenter next window, if enough pixels found
            if len(good_left_inds) > minpix and stop_left < 2:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix and stop_right < 2:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def update(self, binary_warped):
        bw = binary_warped
        # Obtain lane pixels
        leftx, lefty, rightx, righty, window_img = self.find_lane_pixels(bw)

        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, bw.shape[0] - 1,
                            bw.shape[0])
        try:
            left_fitx = (left_fit[0] * ploty ** 2 +
                         left_fit[1] * ploty + left_fit[2])
            right_fitx = (right_fit[0] * ploty ** 2 +
                          right_fit[1] * ploty + right_fit[2])
        except TypeError:
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        # Color the left and right lane regions blue and red
        window_img[lefty, leftx] = [255, 0, 0]
        window_img[righty, rightx] = [0, 0, 255]

        margin = 5
        left_line_window1 = np.array([np.transpose(
                                np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(
                                np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(
                                    np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack(
                                           [right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,
                                                                ploty])))])
        pts = np.hstack((pts_left, pts_right))

        lane_img = np.zeros_like(window_img)
        # lane centre green
        cv2.fillPoly(lane_img, np.int_([pts]), (0, 255, 0))
        # left lane line red
        cv2.fillPoly(lane_img, np.int_([left_line_pts]), (255, 0, 0))
        # right lane line blue
        cv2.fillPoly(lane_img, np.int_([right_line_pts]), (0, 0, 255))

        # calculate radius of curvature
        mpp = self._meters_per_pixel
        y_eval = np.max(ploty)
        left_fit_cr = np.polyfit(mpp * ploty, mpp * left_fitx, 2)
        left_radius_of_curvature = (((1 + (2 * left_fit_cr[0] * y_eval*mpp +
                                    left_fit_cr[1]) ** 2) ** 1.5) /
                                    np.absolute(2 * left_fit_cr[0]))

        # calculate vehicle displacement from lane centre
        displacement = mpp * ((bw.shape[1] / 2) -
                              ((left_fitx[-1] + right_fitx[-1]) / 2))

        return lane_img, left_radius_of_curvature, displacement, window_img
