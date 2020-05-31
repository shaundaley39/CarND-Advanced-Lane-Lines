## Write-Up

---

**Advanced Lane Finding Project**

This project explores computer vision concepts and tools - using the very practical problem of road lane inference. The goals of this project are the following:

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

[image15]: ./test_images/test5.jpg
[image16]: ./intermediate/test5_undistorted.jpg
[image17]: ./output_images/test5_trans.jpg
[image18]: ./output_images/test5_activated_pixels.jpg
[image19]: ./output_images/test5_lboxed.jpg
[image20]: ./output_images/test5_lined.jpg
[image21]: ./output_images/test5_anotated.jpg

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

The pipeline for processing and annotating an image frame is implemented in [pipeline.py](pipeline.py).

Let's run through an example of a single image frame going through each stage of the pipeline.

#### 1. Distortion correction

As outlined above, the first requirement to process an image frame is to correct the image for distortion. That is handled by code in [camera_calibration.py](camera_calibration.py). Here is an example:

Raw image | Distortion-corrected image
:-------------------------:|:-------------------------:
![][image15] | ![][image16]


#### 2. Birds' eye transform

The next step in the pipeline is to transform the image to a birds' eye perspective - this is more appropriate for fitting to a lane line model. The code for this is in [birdseye_transform.py](birdseye_transform.py). Here is an example:

Distortion-corrected image | Birds' eye transformed image
:-------------------------:|:-------------------------:
![][image16] | ![][image17]

#### 3. Activated pixel extraction

The next step in the pipeline is to extract activated pixels likely to belong to lane line markings. The code for this is entirely in [pipeline.py](pipeline.py). Three filters are cominbed in a logical or operation:
- a Sobel filter (using a custom Sobel operator) in which the x-gradient of grayscale pixels is thresholded in the range [20, 100] \(colored red on the activated pixel image below\)
- a Sobel filter (using a custom Sobel operator) in which the x-gradient of S values is thresholded in the range [20, 100] \(colored green\)
- an S filter, in which the frame is converted to the HLS color space, and pixels are activated only where the S value is in the range [170, 255] \(colored blue\)

Birds' eye transformed image | Activated pixels
:-------------------------:|:-------------------------:
![][image17] | ![][image18]


#### 4. Lane line inference
In the next pipeline step, activated pixels are processed with a sliding window algorithm - some pixels are attributed to the left lane, some are attributed to the right lane and those falling outside a window are discarded entirely. The code for this is in [lane_lines.py](lane_lines.py).

Sliding windows | Inferred lane lines
:-------------------------:|:-------------------------:
![][image19] | ![][image20]


#### 5. Lane curvature and lateral position

The lane line model in [lane_lines.py](lane_lines.py) also calculates the lane curvature. This is implemented using the left lane line polynomial only, and uses the meters per pixel value established when the vehicle's birds' eye transform was calculated.

The lateral displacement of the vehicle within its lane is also calculated here - it assumes the camera is mounted centrally on the front of the vehicle.

#### 6. Visualizing the output

The frame returned by our pipeline consists of the undistorted input image, annotated with lane line markings, the lane's radius of curvature and our vehicle's displacement within the lane:

![Annotated image][image21]

---

### Pipeline (video)

In [pipeline.py](pipeline.py), this same image-by-image processing is applied to each frame in a video feed. The result of this for the project video can be seen here:

[![project video](https://img.youtube.com/vi/ic2hQlMukNU/0.jpg)](https://www.youtube.com/watch?v=ic2hQlMukNU)

---

### Discussion

While this achieved pretty good results with test images and with the project video, this fails with more challenging videos, for a number of reasons:
- the birds' eye transform makes a flat-world assumption. On hilly road surfaces this does not (even approximately) hold, resulting in a distorted and inaccurate birds' eye view
- pixel activation is based on thresholding, which is not sufficiently robust to wide variations in sunshine brightness, shading, overpasses/ bridges, road surface color and lane marking colors. In new untrained environments, there will always be a risk that thresholds break down. And it is likely impossible (at very least a major undertaking) to obtain a set of pixel activation thresholds that work well under all environments and conditions.
- the sliding windows lane line model also has limitations. If there happen to be activated pixels close to a lane marking, the whole model may slide laterally off-track. If lane lines merge or separate (e.g. motorway on-ramps or exits, or narrowing roads), the model will break down. If the vehicle is in the middle of changing lanes, the model will break down. Ideally, a more generic lane model would be robust to all these variations.

A better approach to lane line detection would likely need to improve on all three of these points:
- robustness to variations in road surface topography
- an approach to lane line feature extraction that is easier to implement, easier to scale to more environments, which will improve with more data and which will be more robust. Neural networks may be an appropriate tool.
- a model for lane lines and lane line markings that can represent something close to reality, across most real world situations (lanes merging, lanes separating, lane lines covered in leaves...)
