import math
import time
from builtins import int, len, range, list, float, sorted, max, min
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import sys
import cv2
import imutils

#  TODO: Fix this freaking virtual environment so we don't have
#   a ton of import statements

#  takes in input in the form of two corners of the bounding box
#  across a diagonal from each other.
#  returns a list of ints in the form of (x, y) of the midpoint
def midpoint(x1, y1, x2, y2):
    return [int((x2 - x1)/2), int((y2 + y1)/2)]


#  takes in input in the form of two corners of the bounding box
#  across a diagonal from each other and the image file
#  returns the image but cropped to the bounding boxes
def crop(x1, y1, x2, y2, image):
    return image.crop((x1, y1, x2, y2))

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


#  applies Gaussian blur and canny edge detection to the image
#  parameters are the image file, the midpoint, and canny edge
#  thresholds
def canny_edge(image_file):
    src = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    #  TODO: Like the threshold values for canny edge, we need
    #   to determine the kernel values for Gaussian blur
    #  apply Gaussian blur on src image

    blurred = cv2.GaussianBlur(image_file, (9, 9), cv2.BORDER_DEFAULT)
    #  apply canny edge detection to the blurred image
    edge = auto_canny(src)
    x = edge.copy()
    cnts = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    l, a, b = cv2.split(blurred)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    cv2.imshow('CLAHE output', cl)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    cv2.imshow('limg', limg)

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imshow('final', final)
    edge = auto_canny(final)
    x = edge.copy()
    cv2.drawContours(x, cnts, -1, (255, 0, 0), 1)
    cv2.imshow("Contours", x)
    cv2.waitKey(0)
    cnts = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:3]
    k = src.copy()
    print("Contours: ", cnts)

    cv2.drawContours(k, [cnts[0]], -1, (255, 255, 0), 1)
    cv2.imshow("Contours", k)
    cv2.waitKey(0)
    return edge

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#  uses the canny edge image and the midpoint to determine the two points
#  that the robot arm needs to grab
def shortest_path(edge, mid, w, h):
    pix_val = []
    half_cols = mid[0]
    half_rows = mid[1]
    total_cols = w
    total_rows = h
    edge = Image.fromarray(edge)
    edge = edge.convert('RGB')
    #  range goes from halfway through the x direction and
    #  the whole way in the y direction
    for i in range(half_cols):
        for j in range(total_rows):
            r, g, b = edge.getpixel((i, j))
            if r == g == b == 255:
                pix_val.append([i, j])

    min_distance = float("inf")

    val_x1, val_y1, val_x2, val_y2 = -1, -1, -1, -1

    for coor in pix_val:
        col, row = coor[0], coor[1]
        theta = math.atan2((half_cols - col), (half_rows - row))
        for radius in range(int(min(total_rows, total_cols)/2)):
            new_col = mid[0] + math.cos(theta)*radius
            new_row = mid[1] + math.sin(theta)*radius
            r, g, b = edge.getpixel((new_col, new_row))
            if r == g == b == 255:
                dist = distance(new_col, new_row, col, row)
                if dist < min_distance:
                    val_x1 = new_col
                    val_y1 = new_row
                    val_x2 = col
                    val_y2 = row
                    min_distance = dist
    
    return "Shortest path: ", val_x1, val_y1, " to ", val_x2, val_y2, " Distance: ", min_distance


#  assumes sys.argv[1] (first argument after the python script call)
#  is a filename; otherwise has a default file
def main():
    image_file = "image.png"
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    with Image.open(image_file) as image:
        width, height = image.size
    #  TODO: Write code with the object detection script to return bounding box coordinates
    #   for now, just assume that the bounding box is the whole image, and no cropping is necessary.
    original_img = Image.open(sys.argv[1])
    opencvimage = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)


    edge_image = canny_edge(opencvimage)
    cv2.imshow("edges", edge_image)
    cv2.waitKey(0)
    #  time.sleep(5)
    print("slept!")
    #  original_img.close()

    #  edge_image = Image.open(edge_image)
    #  edge_image.show()

    mid = midpoint(0, 0, width, height)
    print(shortest_path(edge_image, mid, width, height))
    cv2.destroyAllWindows()

main()
