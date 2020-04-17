## Write-Up

---

**Advanced Lane Finding Project**

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image and Video References)

[image1]: ./camera_cal/calibration4.jpg "Raw chessboard image"
[image2]: ./intermediate/calibration4_undistorted.jpg "Undistorted chessboard image"
[image3]: ./test_images/test6.jpg "Raw road image"
[image4]: ./intermediate/test6_undistorted.jpg "Undistorted road image"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/1966/view) Points

---

### Camera Calibration

#### 1. 

Wherever cameras are to be used for inference, and especially where distances matter, is important to correct for lens distortion. For self driving cars, both robust inference and estimates of distance clearly matter!

To correct for both tangential and radial distortion, we need to infer both a distortion matrix and a camera matrix ([more background theory here](https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html)). The approach taken in this project was to put together a CameraCalibration helper class (implemented in [camera_calibration.py](camera_calibration.py), tested in [test_camera_calibration.py](test_camera_calibration.py)). It is instantiated with a glob referrence to set of chessboard pattern images, makes use of OpenCV to obtain distortion and camera matrices, and each instance provides the "undistort" method to remove lens distortion form images.

See the following examples of lens distortion correction:
- first, a chessboard pattern image (the sort used for calibration)
- second, a road scene - the sort of image we'll be working with (radial distortion correction around the edge of the image is particularly visible)

Raw images | Distortion corrected images
:-------------------------:|:-------------------------:
![raw chessboard][image1] | ![distortion corrected][image2]
![raw road image][image3] | ![distortion corrected road image][image4]

### Birds' Eye View Warping

Here we'll project road images to a birds' eye view.

Right now, that's important for lane position and road curviture (very useful if we want to write a controller to stay in a lane).

Later, this could be very useful for localization, mapping or SLAM. This sort of projection could also give us an estimate of distance from the vehicle in front.

### Pipeline (single images)

Let's run through an example...

#### 1. Distortion correction

Apply distortion correction

#### 2. Birds' eye transform

Apply birds' eye

#### 3. Activated pixel extraction

Sobel operators and color spaces.


#### 4. Lane line inference
- Sliding window lane model with histogram initialization
- memory and fall-back
- polynomial fitting

#### 5. Lane curviture and lateral position

Don't lose control.

#### 6. Visualizing the output

Let's look at this.

---

### Pipeline (video)

#### 1. A more exciting visualization

First, let's look at the [project video](./project_video.mp4). Seems pretty good.

How about a [challenge](./challenge_video.mp4)? Perhaps we're still holding together?

What about the [harder challenge](harder_challenge_video.mp4). This is probably where our flat world assumptions defeat us.

---

### Discussion

#### 1. Where did it all go wrong?

#### 2. Focus for improvement
