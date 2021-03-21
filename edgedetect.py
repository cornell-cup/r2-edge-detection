import math
import time
from builtins import int, len, range, list, float, sorted, max, min
# import matplotlib.pyplot as plt
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

def canny_edge(image_file, width, height):
    src = cv2.cvtColor(image_file, cv2.IMREAD_GRAYSCALE)
    #  TODO: Like the threshold values for canny edge, we need
    #   to determine the kernel values for Gaussian blur
    #  apply Gaussian blur on src image

    # TODO: Implement findContours to get the outline of the object:
    #  https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/

    blurred = cv2.GaussianBlur(image_file, (5, 5), cv2.BORDER_DEFAULT)

    #  apply canny edge detection to the blurred image
    edge = auto_canny(src)
    x = edge.copy()
    cnts = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    l, a, b = cv2.split(blurred)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))
    # cv2.imshow('limg', limg)

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # cv2.imshow('final', final)

    edge = auto_canny(final)
    x = edge.copy()
    cv2.drawContours(x, cnts, -1, (255, 0, 0), 5)
    # cv2.imshow("Contours", x)
    # cv2.waitKey(0)
    cnts = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    k = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    # print("Contours: ", cnts)
    cv2.drawContours(k, [cnts[0]], -1, (255, 255, 255), 1)
    # cv2.imshow("Contours", k)
    # cv2.waitKey(0)
    M = cv2.moments(cnts[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return k, [cX, cY]


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#  uses the canny edge image and the midpoint to determine the two points
#  that the robot arm needs to grab


def shortest_path(edge, mid_contour, w, h):
    pix_val = []
    half_cols = w//2
    total_cols = w
    total_rows = h
    #  range goes from halfway through the x direction and
    #  the whole way in the y direction
    for i in range(half_cols):
        for j in range(total_rows):
            g = edge[j][i][1]
            if g == 255:
                pix_val.append([i, j])

    min_distance = float("inf")

    val_x1, val_y1, val_x2, val_y2 = -1, -1, -1, -1

    for coor in pix_val:
        col, row = coor[0], coor[1]
        theta = math.atan2((mid_contour[0] - col), (mid_contour[1] - row))
        for radius in range(int(min(total_rows, total_cols)/2)):
            new_col = min(mid_contour[0] +
                          math.sin(theta)*radius, total_cols-1)
            new_row = min(mid_contour[1] +
                          math.cos(theta)*radius, total_rows-1)
            # print((new_col, new_row))
            g = edge[int(new_row)][int(new_col)][1]
            if g == 255:
                dist = distance(new_col, new_row, col, row)
                if dist < min_distance:
                    val_x1 = col
                    val_y1 = row
                    val_x2 = new_col
                    val_y2 = new_row
                    min_distance = dist

    cv2.circle(edge, (int(val_x1), int(val_y1)), 5, (255, 0, 255), -1)
    cv2.circle(edge, (int(val_x2), int(val_y2)), 5, (255, 0, 255), -1)
    cv2.circle(edge, (int(mid_contour[0]), int(
        mid_contour[1])), 5, (255, 0, 255), -1)

    cv2.imshow("points", edge)
    cv2.waitKey(0)

    return "Shortest path: ", val_x1, val_y1, " to ", val_x2, val_y2, " Distance: ", min_distance


#  assumes sys.argv[1] (first argument after the python script call)
#  is a filename; otherwise has a default file
def main():
    image_file = "IMG_1120.jpg"
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    with Image.open(image_file) as image:
        width, height = image.size

    #  TODO: Write code with the object detection script to return bounding box coordinates
    #   for now, just assume that the bounding box is the whole image, and no cropping is necessary.
    original_img = Image.open(sys.argv[1])
    print(type(original_img))
    opencvimage = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    cv2.imshow("original image", opencvimage)
    cv2.waitKey(0)
    ceRet = canny_edge(opencvimage, width, height)
    edge_image = ceRet[0]
    cv2.imshow("edges", edge_image)
    cv2.waitKey(0)
    #  time.sleep(5)
    #  original_img.close()

    #  edge_image = Image.open(edge_image)
    #  edge_image.show()

    mid = midpoint(0, 0, width, height)
    to_file =  str(ceRet[1][0]/1000) + " " + str(ceRet[1][1]/1000) + " 0"

    with open('/Volumes/alison\'s home/c1c0_arm/coordinates.txt', 'w+') as f:
        f.write(to_file)
        f.close()
    print("Midpoint: ", to_file)
    print(shortest_path(edge_image, ceRet[1], width, height))
    cv2.destroyAllWindows()


main()
