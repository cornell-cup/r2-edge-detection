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

# def grabPointsImage(x1, y1, width_box, height_box, im):
#     grabPoints(x1, y1, width_box, height_box, cv2.imread(im, cv2.COLOR_RGB2BGR))

def grabPoints(x1, y1, width_box, height_box, image):
    def auto_canny(image, sigma=0.33):
        # compute the medisan of the single channel pixel intensities
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

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl,a,b))
        #-----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        edge = auto_canny(final)
        x = edge.copy()
        cv2.drawContours(x, cnts, -1, (255, 0, 0), 5)
        cnts = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:10]
        k = np.zeros(shape=[height, width, 3], dtype=np.uint8)
        cv2.drawContours(k, [cnts[0]], -1, (255, 255, 255), 1)
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
        half_rows = h//2
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
            theta = math.atan2((mid_contour[0] - col), (mid_contour[1] - row))
            for radius in range(int(min(total_rows, total_cols)/2)):
                new_col = min(mid_contour[0] + math.sin(theta)*radius, total_cols-1)
                new_row = min(mid_contour[1] + math.cos(theta)*radius, total_rows-1)
                # print((new_col, new_row))
                r, g, b = edge.getpixel((new_col, new_row))
                if r == g == b == 255:
                    dist = distance(new_col, new_row, col, row)
                    if dist < min_distance:
                        val_x1 = col
                        val_y1 = row
                        val_x2 = new_col
                        val_y2 = new_row
                        min_distance = dist

        # draw = ImageDraw.Draw(edge)
        # draw.ellipse((val_x1-5, val_y1-5, val_x1+5, val_y1+5), fill = 'blue', outline ='blue')
        # draw.ellipse((val_x2-5, val_y2-5, val_x2+5, val_y2+5), fill = 'blue', outline ='blue')
        # # draw.ellipse((half_cols-5, half_rows-5, half_cols+5, half_rows+5), fill = 'blue', outline ='blue')
        # draw.ellipse((mid_contour[0]-5, mid_contour[1]-5, mid_contour[0]+5, mid_contour[1]+5), fill = 'blue', outline ='blue')
        # edge.show()

        # return "Shortest path: ", val_x1, val_y1, " to ", val_x2, val_y2, " Distance: ", min_distance
        return val_x1, val_y1, val_x2, val_y2, min_distance


    #  assumes sys.argv[1] (first argument after the python script call)
    #  is a filename; otherwise has a default file
    raw_img = image
    width, height = width_box, height_box
    cropped_image = raw_img[y1:y1+height_box, x1:x1+width_box]
    # cropped_image = raw_img.crop((x1, y1, x1 + width_box, y1 - height_box))

    opencvimage = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

    ceRet = canny_edge(opencvimage, width, height)
    edge_image = ceRet[0]
    cv2.imshow("edges", edge_image)

    mid = [int((width)/2), int((height)/2)]
    shortest_x1, shortest_y1, shortest_x2, shortest_y2, shortest_dist = shortest_path(edge_image, ceRet[1], width, height)
    # cv2.destroyAllWindows()
    return shortest_x1 + x1, shortest_y1 + y1, shortest_x2 + x1, shortest_y2 + y1, shortest_dist

# image = Image.open(sys.argv[1])
# width, height = image.size
# grabPoints(0, 0, width, height, np.array(image))