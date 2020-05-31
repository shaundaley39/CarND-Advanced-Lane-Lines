#!/usr/bin/env python3

from camera_calibration import *
from birdseye_transform import *
from lane_lines import *
from moviepy.editor import VideoFileClip


test_glob = 'test_images/*.jpg'
# test_glob = 'video_images/*.jpg'


c = CameraCalibration('camera_cal/calibration*.jpg')
t = BirdsEyeTransform(c, 'test_images/straight_lines*jpg')


def process_image(image, fname=None, out_dir=None):
    undistorted = c.undistort(image)
    hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)

    s_thresh = (170, 255)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    sx_thresh = (20, 100)
    sobel_sx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)
    abs_sobel_sx = np.absolute(sobel_sx)
    scaled_sobel_sx = np.uint8(255 * abs_sobel_sx / np.max(abs_sobel_sx))
    sx_binary = np.zeros_like(scaled_sobel_sx)
    sx_binary[(scaled_sobel_sx >= sx_thresh[0]) &
              (scaled_sobel_sx <= sx_thresh[1])] = 1

    x_thresh = (20, 100)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel_x = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))
    x_binary = np.zeros_like(scaled_sobel_x)
    x_binary[(scaled_sobel_x >= x_thresh[0]) &
             (scaled_sobel_x <= x_thresh[1])] = 1

    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1) | (sx_binary == 1) | (x_binary == 1)] = 1
    combined_binary_t = t.transform(combined_binary)

    ll = LaneLines(t.meters_per_pixel)
    lane, left_curverad, displacement, windows = ll.update(combined_binary_t)
    lane_u = t.untransform(lane)

    annotated_img = cv2.addWeighted(undistorted, 1, lane_u, 0.3, 0)
    rc_text = "Radius of Curvature = {0: >5.0f}m".format(left_curverad)
    cv2.putText(annotated_img, rc_text, (50, 50), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 2)
    direction = "right" if displacement >= 0 else "left"
    disp_text = "Vehicle is {0:.2f}m {1} of center".format(abs(displacement),
                                                           direction)
    cv2.putText(annotated_img, disp_text, (50, 100), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 2)

    if fname is not None:
        name = fname.split('/')[-1][:-4]
        out_dir = out_dir + "/"
        color_binary = np.dstack((x_binary, sx_binary, s_binary)) * 255
        color_binary_t = t.transform(color_binary)
        mpimg.imsave(out_dir + name + "_activated_pixels.jpg", color_binary_t)
        mpimg.imsave(out_dir + name + "_lboxed.jpg", windows)
        mpimg.imsave(out_dir + name + "_lined.jpg", lane)
        transformed = t.transform(undistorted)
        mpimg.imsave(out_dir + name + "_trans.jpg", transformed)
        mpimg.imsave(out_dir + name + "_anotated.jpg", annotated_img)
    return annotated_img


def annotate_files(in_glob, out_dir):
    for fname in glob.glob(in_glob):
        if '.mp4' == fname[-4:]:
            clip = VideoFileClip(fname)
            outputf = out_dir + "/" + fname
            try:
                clip = clip.fl_image(process_image)
                clip.write_videofile(outputf, audio=False)
            except Exception:
                pass
        else:
            process_image(mpimg.imread(fname), fname=fname, out_dir=out_dir)


if __name__ == "__main__":
    annotate_files(test_glob, "output_images")
    annotate_files("project_video.mp4", "output_images")
