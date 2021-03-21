# Testing Instructions
## Install
To make sure you have all the required packages, you should create a virtual environment, activate it, and run

`pip install -r requirements.txt`
(there might be more packages in there than necessary, and it might not have
packages for files like kinematics.py, oops)

## Important directories/files for testing:
Samples - contains testing images for 10 objects, each with 5 poses, from the
JACQUARD dataset. The 10 subdirectories corresponds to the images for one
object. 

grasptest.py - contains the testing code that uses final.py's grab_points()
grasp prediction function.

## How to test:
`python grasptest.py [dir_with_image_files] [take manual input True or False]`

For each image, the code should display
- What the RGB version of the image looked like after edge detection, along with
grasp points (in pink)
- What the original RGB image looked like, with grasp points

Upon closing the two display windows,
- What the labels from JACQUARD were (orange for grasp points, yellow for grasp
  rectangle)
- And the calculated grasp prediction. (The pink dots are the grasp
  points and the green rectangle represents the grasp rect with different sized
  end effectors)
  
After closing that window,
- the image name and the mode [0 for "rgb", 1 for "d", 2 for "rgd"] will be
  printed, along with the automated validation result (which most of the time is
  not correct, sadly :( )
- You will also be prompted to give manual input on whether the grasp prediction
  was valid.

After validating all the images in the specified directory, an array will be
printed, with accuracy percentages corresponding to ["rgb", "d", "rgd"].

Use Ctrl + c (and close any open windows) to terminate running for any reason.
