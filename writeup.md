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

[image5]: ./intermediate/straight_lines1_undistorted.jpg "Straight 1 - undistorted"
[image6]: ./intermediate/straight_lines1_t_filtered.jpg "Straight 1 - filtered"
[image7]: ./intermediate/straight_lines1_t_filtered_lined.jpg "Straight 1 - lined"
[image8]: ./intermediate/straight_lines1_t_filtered_lined_transformed.jpg "Straight 1 - lined transformed"

[image9]: ./test_images/straight_lines1.jpg
[image10]: ./intermediate/straight_lines1_ctransformed.jpg
[image11]: ./test_images/straight_lines2.jpg
[image12]: ./intermediate/straight_lines2_ctransformed.jpg
[image13]: ./test_images/test2.jpg
[image14]: ./intermediate/test2_ctransformed.jpg

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/1966/view) Points

---

### Camera Calibration

Wherever cameras are to be used for inference, and especially where distances matter, is important to correct for lens distortion. For self driving cars, both robust inference and estimates of distance clearly matter!

To correct for both tangential and radial distortion, we need to infer both a distortion matrix and a camera matrix ([more background theory here](https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html)). The approach taken in this project was to put together a CameraCalibration helper class (implemented in [camera_calibration.py](camera_calibration.py), tested in [test_camera_calibration.py](test_camera_calibration.py)). It is instantiated with a glob referrence to set of chessboard pattern images, makes use of OpenCV to obtain distortion and camera matrices, and each instance provides the "undistort" method to remove lens distortion form images.

See the following examples of lens distortion correction:
- first, a chessboard pattern image (the sort used for calibration)
- second, a road scene - the sort of image we'll be working with (radial distortion correction around the edge of the image is particularly visible)

Raw images | Distortion corrected images
:-------------------------:|:-------------------------:
![raw chessboard][image1] | ![distortion corrected][image2]
![raw road image][image3] | ![distortion corrected road image][image4]

### Birds' Eye View Transformation

The requirement here is to take images from a forward facing video camera from a car, and to project that image onto a plane as though from a birds' eye view. This projection can then be used to calculate road lane curviture and the lateral position of the vehicle in the lane - key objectives of this project, since these numbers can be used in a controller for keeping the vehicle in its lane.

It's worth noting that projections such of this have many further important uses. A birds' eye view transform could be very useful for localization, mapping or SLAM. This sort of projection could also give us an estimate of distance from the vehicle in front, and could be crucial for handling lange changing or merging maneuvers, etc.

For this project, transformation is provided by the BirdsEyeTransform helper class, defined in [birdseye_transform.py](birdseye_transform.py), with an example (and test) in [test_birdseye_transform.py](test_birdseye_transform.py). A BirdsEyeTransform instance is initialized on a glob referrence to "nice" straight line images. By "nice", what we need in these images is a long straight section of road:
- without significant hills
- without shadows or sudden changes in road surface color
- without any other vehicles in the present lane (except in the far distance)
- with the car as close as possible to the centre of the lane (with a large sample size, small deviations average out)

A BirdsEyeTransform object uses these images to infer a transformation matrix. It then provides the transform method for transforming images to a birds' eye view perspective.

Here's a little more detail on the [implementation](test_birdseye_transform.py):

#### 1. First pass: given "nice" straight lane images, infer a birdseye transformation

For each of the straight line images provided, we combine two filters - a Sobel operator for edges with a high x-gradient, and an s-threshold for the HLS color space representation of the image - to create a binary image of activated pixels. From this combination, we extract the region of interest in which a straight lane should be expected, and use a Hough transform to infer lines. These inferred lines are taken to be part of the left lane line or right lane line (bounding the present lane) respectively, based on their gradient. From these inferred left and right lane lines, we infer four points: two close to the horrizon, and two close to the bottom of the image. Since we have multiple images, we take an average across images to get a better estimate for these four points. Viewed from above, these four points define a rectangular area on a plane.

We proceed to project these points onto a rectangle spanning the full height of an image frame, getting a decent first approximation of a birds' eye transform.

Pixels activated by Sobel and S filters | Inferred lane lines used as basis for transform
:-------------------------:|:-------------------------:
![undistorted][image5] | ![lane lines inferred][image7]
![filter activations][image6] | ![first-pass transformed][image8]


#### 2. Second pass: getting meters per pixel consistent across both axes

We have a problem with the first transform: we couldn't control for distance with our rectangular projection, and so each pixel in the x-direction represents a different distance to each pixel in the y-direction. We can correct (approximately) thanks to government regulations: each lane is approximately 3.7 meters wide, and each of the longer dashed lines on the road is approximately 3.0 meters long.

In a second pass through each of our transformed images, we choose activated pixels close to only the left or right lane line respectively, and we consider their sorted y-values. This allows us to infer the pixel-length of dashed lanes in the y-direction. We use this to calculate a meters-per-pixel value; the average (over all images) is then used for the BirdsEyeTransform instance. The transform is adjusted so that the x-axis takes the same meters-per-pixel value as the y-axis.

#### 3. Birds' eye ransformation results

Raw images | Undistorted and transformed images
:-------------------------:|:-------------------------:
![][image9] | ![][image10]
![][image11] | ![][image12]
![][image13] | ![][image14]


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
