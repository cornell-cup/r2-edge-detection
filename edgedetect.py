import math
import time
# from builtins import int, len, range, list, float
from PIL import Image
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

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


#  applies Gaussian blur and canny edge detection to the image
#  parameters are the image file, the midpoint, and canny edge
#  thresholds
def canny_edge(image_file, t1, t2):
    src = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    #  TODO: Like the threshold values for canny edge, we need
    #   to determine the kernel values for Gaussian blur
    #  apply Gaussian blur on src image
    blurred = cv2.GaussianBlur(src, (5, 5), cv2.BORDER_DEFAULT)

    #  apply canny edge detection to the blurred image
    edge = cv2.Canny(blurred, t1, t2)
    return edge

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#  uses the canny edge image and the midpoint to determine the two points
#  that the robot arm needs to grab
def shortest_path(edge, mid, w, h):
    pix_val = []
    #  range goes from halfway through the x direction and
    #  the whole way in the y direction
    for i in range(mid[0] - w, mid[0]):
        for j in range(mid[1] - h, mid[1] + h):
            r, g, b = edge.getpixel(i, j)
            if r == g == b == 0:
                pix_val.append([i, j])

    min_distance = float("inf")

    val_x1, val_y1, val_x2, val_y2 = -1
    for coor in pix_val:
        cols, rows = pix_val[0], pix_val[1]
        theta = math.atan2((rows-mid[1]), (cols-mid[0]))
        for radius in range(math.sqrt((w*2)**2+(h)**2)):
            new_col = mid[0] + math.cos(theta)*radius
            new_row = mid[1] + math.sin(theta)*radius
            r, g, b = edge.getpixel(new_col, new_row)
            if r == g == b == 0:
                dist = distance(new_col, new_row, cols, rows)
                if dist < min_distance:
                    val_x1 = new_col
                    val_y1 = new_row
                    val_x2 = cols
                    val_y2 = rows
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
    #   Also need to determine what threshold values we need to use through adjustment, not hard code
    original_img = Image.open(sys.argv[1])
    original_img.show()
    edge_image = canny_edge(image_file, 25, 55)
    # cv2.imshow("edges", edge_image)

    time.sleep(5)
    edge_image = Image.open(edge_image)
    edge_image.show()
    mid = midpoint(0, 0, width, height)
    # print(shortest_path(edge_image, mid, int(width/2), height))

main()
