## Advanced Lane Detection
### Self-Driving Cars Nanodegree @Udacity

###Credits

- Udacity: Self-Driving Car Nano Degree
- OpenCV: http://opencv-python-tutroals.readthedocs.io/en/latest/
- Vivek Yadav: https://goo.gl/r6SalG

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[imagea]: ./examples/chessboard.png "Chessboard"
[image0]: ./examples/original.png "Original"
[image1]: ./examples/undistorted.png "Undistorted"
[image2]: ./examples/transform.png "Road Transformed"
[image3]: ./examples/hlsv.png "HLSV"
[image4]: ./examples/color_mask.png "Color Mask"
[image5]: ./examples/combined_mask.png "Combined Binary"
[image6]: ./examples/first_pass.png "First frame"
[image7]: ./examples/unwarped.png "Result"
[video1]: ./project_video.mp4 "Video"

###Architecture:
A brief overview on the computer vision techniques applied:
![architecture] (https://docs.google.com/drawings/d/17b7_UU4qU-ah_H3Syc5WYYHr3q7R4k_pJ8eMl0hMXKo/pub?w=958&h=1344)
---

###Camera Calibration

####1. Camera Matrix & Distortion Correction:

The code for this step is contained in the fourth and fifth code cell of the IPython notebook located in "./examples/example.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][imagea]

###Pipeline (single images)

####Raw Image:
Raw images like the one below are obtained frame by frame from the video.
![alt text][image0]

####1. Distortion-corrected image.
To demonstrate this step, I apply the distortion correction (code cell: ) to one of the test images like this one:
![alt text][image1]

####2. I then performed a perspective transform to obtain a birds-eye view for the image.

The code for my perspective transform includes a function `warp_image()` (code cell: ) .  The `warp_image()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner:

```
img_size = img.shape
ht_window = np.uint(img_size[0]/1.5)
hb_window = np.uint(img_size[0])
c_window = np.uint(img_size[1]/2)
ctl_window = c_window - .2*np.uint(img_size[1]/2)
ctr_window = c_window + .2*np.uint(img_size[1]/2)
cbl_window = c_window - 1*np.uint(img_size[1]/2)
cbr_window = c_window + 1*np.uint(img_size[1]/2)

src = np.float32([[cbl_window,hb_window],[cbr_window,hb_window],[ctr_window,ht_window],[ctl_window,ht_window]])

dst = np.float32([[0,img_size[0]],[img_size[1],img_size[0]],[img_size[1],0],[0,0]])

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 0, 720      | 0, 720        |
| 1280, 720      | 1280, 720      |
| 768, 480     | 1280, 0      |
| 512, 480      | 0, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####3. Combination of color transforms, gradients or other methods to create a thresholded binary image:
I used a combination of color and gradient thresholds to generate a binary image.  

Example of H, L, S & V color spaces:
![alt text][image3]

Example output of applied color mask:
![alt text][image4]

Example output of combined binary images:
![alt text][image5]


####4. Identified lane-line pixels and fit their positions with a polynomial:

For first frame of image, I produced a histogram for the image and searched for the peaks - to identify lanes. I then broke the image in 9 horizontal sections and searched for lane points section by section.

For consecutive images I reduced my search area to a +/- 100 pixel boundary of existing lane lines.

Outliers: While checking for lines I added the following outlier checks to improve performance:
- check for existence of lane lines (if no lane lines are found, I fall backed to previous detected lines)
- minimum average distance between right lane and left lane (must be positive and greater than 600 pixels)

I then fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]

ps: code cell #19 and #20

####5. Radius of Curvature:
How:
After obtaining a polynomial fit for the two lane lines, I calculated the radius of curvature by using the following formula. I further converted this value to meters (From pixels).

```
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) /
np.absolute(2*left_fit[0])
```

ps: refer code cell #19 and  #20

####5. Vehicle Position:
How: By comparing the screen center to the midpoint between the two lanes.

ps: refer code cell #19 and  #20

####6. Marked lanes plotted back down onto the road:

How:

Here is an example of my result on a test image:
![alt text][image7]

ps: refer code cell #23.  


---

###Pipeline (video)

####Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Problems / issues you faced in your implementation of this project:

####2. Where will your pipeline likely fail?  

####3. What could you do to make it more robust?
