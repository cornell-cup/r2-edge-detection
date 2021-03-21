import math
import time
from builtins import int, len, range, list, float, sorted, max, min
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import sys
import cv2
import imutils
from final import grab_points
import os

from typing import Dict
import json

# library needed to read depth maps for BIGBIRD dataset
import h5py
# to print out h5 table
import pandas as pd
# to calculate Intersection over Union for rotated rectangles
from shapely.geometry import Polygon

"""This is a collection of functions for testing grab_points of final.py.
Built for testing with the JACQUARD dataset, and still has some old functions
for the BIGBIRD dataset.
"""


def true_grasps(path, image_name, rgb_line_ending, labels_ending):
    """ Draws grasp rectangles computed from [image_name]'s labels. \n
    [path] is the path to where [image_name] is stored \n
    [rgb_line_ending] and  [labels_ending] are the identifier endings and file
    types of the RGB [image_name] file and the collection of labels for
    [image_name], respectively. """

    img = cv2.imread(path + image_name + rgb_line_ending)
    f = open(path + image_name + labels_ending, "r")
    labels = f.readlines()

    for label in labels:
        label_arr = np.array(label.split(";")).astype(np.float)

        ctr_x, ctr_y = label_arr[0], label_arr[1]
        # maybe opening and jaw size don't need to be halved actually, not sure
        gripper_w = label_arr[3] / 2  # opening
        gripper_h = label_arr[4] / 2  # jaw size
        angle = math.radians(label_arr[2])  # convert to radians
        # print(ctr_x, ctr_y)
        # print(angle)
        rect, _ = calc_label_rect(
            img, gripper_w, gripper_h, ctr_x, ctr_y, angle)
        plot_label_rect(img, rect)
    cv2.imshow("ground truths", img)

    # this commented out code was for seeing if using labels as an outline to
    # grab points from would give a better truth grasp
    # contours = grab_points(0, 0, img.shape[0], img.shape[1], img)

    cv2.waitKey(0)


def calc_label_rect(img, gripper_w, gripper_h, ctr_x, ctr_y, angle):
    # will have to double check that the + and - signs are right
    # NOTE coordinate plane is 0,0 top left pixel.
    # ALso, ANGLE IS MIRRORED OVER HORIZONTAL from the usual
    # clockwise from horizontal not counterclockwise
    dx_w = gripper_w * math.cos(angle)
    dy_w = gripper_w * math.sin(angle)

    dx_h = math.sqrt(gripper_h ** 2 + gripper_w ** 2) * math.sin(angle)
    dy_h = math.sqrt(gripper_h ** 2 + gripper_w ** 2) * math.cos(angle)

    # draw the two gripper dots according to angle and gripper width
    cv2.circle(img, (int(ctr_x + dx_w), int(ctr_y + dy_w)),
               3, (0, 200, 255), -1)
    cv2.circle(img, (int(ctr_x - dx_w), int(ctr_y - dy_w)),
               3, (0, 200, 255), -1)
    # center
    cv2.circle(img, (int(ctr_x), int(ctr_y)), 3, (0, 150, 255), -1)

    rect = np.array([[ctr_x - dx_w - dx_h, ctr_y - dy_w + dy_h],
                     [ctr_x - dx_w + dx_h, ctr_y - dy_w - dy_h],
                     [ctr_x + dx_w + dx_h, ctr_y + dy_w - dy_h],
                     [ctr_x + dx_w - dx_h, ctr_y + dy_w + dy_h]], np.int32)

    ground_truths = Polygon(
        [(ctr_x - dx_w - dx_h, ctr_y - dy_w + dy_h),  # if vals are pos, bottom left
         (ctr_x - dx_w + dx_h, ctr_y - dy_w - dy_h),
         (ctr_x + dx_w + dx_h, ctr_y + dy_w - dy_h),
         (ctr_x + dx_w - dx_h, ctr_y + dy_w + dy_h)])

    return rect, ground_truths  # rect is for plotting, ground_truths is for IoU


def plot_label_rect(img, rect):
    cv2.polylines(img, [rect], True, (0, 255, 255), 1)


def calc_pred_rect(img, points, gripper_h):
    # ================ edge detected grasp point =================
    # gripper_h is taken from labels, just to maintain consistency
    # don't know the specs of the end effector yet so
    x1, x2 = points[0], points[2]
    y1, y2 = points[1], points[3]
    diff_x = abs(x1 - x2)
    diff_y = abs(y1 - y2)
    pred_angle = math.atan2(y2-y1, x2-x1)  # in radians

    # TODO: Is my trig correct? T_T

    gripper_w_pred = math.sqrt(diff_x ** 2 + diff_y ** 2) / 2
    dx_w_pred = gripper_w_pred * math.cos(pred_angle)
    dy_w_pred = gripper_w_pred * math.sin(pred_angle)

    dx_h_pred = math.sqrt(gripper_h ** 2 + gripper_w_pred **
                          2) * math.sin(pred_angle)
    dy_h_pred = math.sqrt(gripper_h ** 2 + gripper_w_pred **
                          2) * math.cos(pred_angle)

    # recalculate center by taking average of two points' components
    ctr_x_pred = (x1 + x2) / 2
    ctr_y_pred = (y1 + y2) / 2

    # draw the two gripper dots again (just in case they got covered)
    cv2.circle(img, (int(x1), int(y1)),
               5, (255, 0, 255), -1)
    cv2.circle(img, (int(x2), int(y2)),
               5, (255, 0, 255), -1)

    prect = np.array([[ctr_x_pred - dx_w_pred - dx_h_pred, ctr_y_pred - dy_w_pred + dy_h_pred],
                      [ctr_x_pred - dx_w_pred + dx_h_pred,
                       ctr_y_pred - dy_w_pred - dy_h_pred],
                      [ctr_x_pred + dx_w_pred + dx_h_pred,
                       ctr_y_pred + dy_w_pred - dy_h_pred],
                      [ctr_x_pred + dx_w_pred - dx_h_pred, ctr_y_pred + dy_w_pred + dy_h_pred]], np.int32)

    predictions = Polygon([(ctr_x_pred - dx_w_pred - dx_h_pred, ctr_y_pred - dy_w_pred + dy_h_pred),
                           (ctr_x_pred - dx_w_pred + dx_h_pred,
                            ctr_y_pred - dy_w_pred - dy_h_pred),
                           (ctr_x_pred + dx_w_pred + dx_h_pred,
                            ctr_y_pred + dy_w_pred - dy_h_pred),
                           (ctr_x_pred + dx_w_pred - dx_h_pred, ctr_y_pred + dy_w_pred + dy_h_pred)])

    # prect for plotting, predictions and angle for calculation IoU
    return prect, predictions, pred_angle


def plot_pred_rect(img, prect):
    cv2.polylines(img, [prect], True, (0, 255, 0), 1)
    # cv2.imshow("pred labels", img)
    # cv2.waitKey(0)


def process_depth(image_path, line_ending):
    """Reads depth image file types and returns corresponding image arrays"""

    if line_ending.endswith(".png") or line_ending.endswith(".jpg"):
        return cv2.imread(image_path+line_ending)
    elif line_ending.endswith(".tiff"):
        return process_tiff(image_path+line_ending)
    elif line_ending.endswith(".h5"):
        process_h5(image_path, line_ending)
        return
    else:
        print("Error: Depth frame file extension not supported")
        return


def process_tiff(full_file_name):
    # Assumes tiff is the same size as rgb image, as with JACQUARD dataset
    img = cv2.imread(full_file_name, cv2.IMREAD_ANYDEPTH)
    img = cv2.normalize(img, None, 0, 255,
                        cv2.NORM_MINMAX, cv2.CV_8U)
    return img


def process_h5(img, depth_ending):
    # Processes h5 files that store depth from the BIGBIRD dataset
    # get depthmap from h5 file
    f = h5py.File(img+depth_ending, "r")
    datasets = [x for x in f.keys()]
    # for x in datasets:
    #     print(x)
    depthframe = f["depth"]
    # print(depthframe)

    d = depthframe[:, :]
    # print(d)
    # print(type(d))

    # get rbg part of img. Assumes rgb images are [img].jpg
    rgb = cv2.imread(img+".jpg")
    print(img.shape)

    # must resize depthmap to match rgb image dimensions
    dims = (rgb.shape[1], rgb.shape[0])
    d = cv2.resize(d, dims)
    # print(resized.shape)

    b, g, r = cv2.split(rgb)
    #print(b.shape, g.shape, r.shape, d.shape)
    #print(b.dtype, g.dtype, r.dtype, d.dtype)

    # convert depth to 8bit representation
    d_scaled = np.zeros(b.shape)
    d_scaled = cv2.normalize(d, d_scaled, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # print(d_scaled.dtype)
    return d_scaled


def eval_pred(img, points, path_to_labels, scale):
    """ Returns 1 if the IoU of the prediction rectangle and label rectangle
    is above a threshold, and grasp angle is within a certain threshold """

    # points is the two points where the grabber goes
    f = open(path_to_labels, "r")
    labels = f.readlines()

    isValid = 0

    for label in labels:  # compare each label to the prediction
        # ============= extract information from labels ================
        # format of label - x;y;theta in degrees;opening;jaw size
        # x, y, opening, and jaw size are in pixels
        # either convert their coordinate system to ours or vice versa
        label = np.array(label.split(";")).astype(np.float)

        # TODO: how to modify the gripper width to be more like real conditions
        # TODO: scale these values according to how the images were scaled
        # halved to more easily compute points
        gripper_w = label[3] * scale / 2  # opening
        gripper_h = label[4] * scale / 2  # jaw size

        ctr_x = label[0] * scale
        ctr_y = label[1] * scale
        angle = math.radians(label[2])  # convert to radians
        # print(angle)

        rect, ground_truths = calc_label_rect(
            img, gripper_w, gripper_h, ctr_x, ctr_y, angle)
        plot_label_rect(img, rect)

        prect, predictions, pred_angle = calc_pred_rect(img, points, gripper_h)
        plot_pred_rect(img, prect)

        intersection = predictions.intersection(ground_truths).area
        union = predictions.union(ground_truths).area
        # print("Intersection: ", intersection, " Union: ", union)
        iou = intersection / union

        # TODO: They also have a simulation online to test prediction points,
        # but I emailed and didn't get a response

        threshold_angle = math.pi/12  # within 15 degrees
        if pred_angle > angle - threshold_angle and pred_angle < angle + threshold_angle and iou > 0.50:
            isValid = 1
    return isValid


def process_images(path, rgb_line_ending, depth_line_ending, manual):
    """ Finds images at the specified [path] and runs the edge detection code to
    generate predictions on RGB, D, and RGD versions of the image.
    Requires: Files of the same object MUST have the same name + an additional
    line ending. The line endings for the rgb and depth frames MUST BE DISTINCT.
    Ex. path="home/images", rgb_line_ending=".png", depth_line_ending="depth.png"
    Each object MUST HAVE BOTH A RGB AND DEPTH IMAGE.
    """
    # TODO: add a toggle to store manual/user input also

    formats = ["rgb", "d", "rgd"]
    # store image names of incorrect preds
    incorrect_preds = [[], [], []]
    manual_incorrect_preds = [[], [], []]
    # store accuracy percentages
    # [rgb, d, rgd] holds sum of pred evals. divide by num preds later
    percentages = [0, 0, 0]
    manual_percentages = [0, 0, 0]

    # stores list of unique image names with line ending removed
    image_names = []
    # checks for image files in immediate directory
    for entry in os.scandir(path):
        if entry.is_file():
            image_name = ""
            if entry.name.endswith(rgb_line_ending):
                # remove line ending
                image_name = entry.name[0:-len(rgb_line_ending)]
                image_names.append(image_name)
            elif entry.name.endswith(depth_line_ending):
                # remove line ending
                image_name = entry.name[0:-len(depth_line_ending)]
                image_names.append(image_name)

    # get unique vals
    image_names = set(image_names)
    # print(image_names)

    # loop over image names
    for image_name in image_names:
        # Generate all the actual images
        # rgb image stored in bgr order because OpenCV is just like that
        bgr = cv2.imread(path+image_name+rgb_line_ending)
        depth = process_depth(path+image_name, depth_line_ending)

        # If you haven't already, do any cropping/scaling necessary (BIGBIRD)

        # resizing to fit on screen when displaying output image
        whratio = bgr.shape[1] / bgr.shape[0]
        # need to scale down labels also - new width/old width
        scale = int(whratio*400) / bgr.shape[1]
        # resize takes parameter order (img, width, height)
        bgr = cv2.resize(bgr, (int(whratio*400), 400))
        depth = cv2.resize(depth, (int(whratio*400), 400))

        b, g, r = cv2.split(bgr)
        dgr = cv2.merge((depth, g, r))

        depth3channel = cv2.merge((depth, depth, depth))

        images = [bgr, depth3channel, dgr]  # iterate over for comparison
        width, height = bgr.shape[1], bgr.shape[0]
        for i in range(3):
            pred_coords = grab_points(0, 0, width, height, images[i])
            # TODO: search image_name.txt file for labels and eval pred
            pred_grasp_valid = eval_pred(
                images[i], pred_coords, path+image_name+"_grasps.txt", scale)
            cv2.imshow("preds", images[i])
            cv2.waitKey(0)
            if pred_grasp_valid:
                print(image_name + " " + str(i) + " was a valid grasp")
                percentages[i] += 1
            else:
                print(image_name + " " + str(i) + " was NOT a valid grasp")
                incorrect_preds[i].append(image_name)

            if manual:
                # user input to confirm whether grasp was correct
                userinput = input("Is the grasp valid? (y/n) \n")
                while userinput != "y" and userinput != "n":
                    print("Invalid input. Please enter y/n")
                    userinput = input("Is the grasp valid? (y/n) \n")

                if userinput == "y":
                    manual_percentages[i] += 1
                else:
                    manual_incorrect_preds[i].append(image_name)

    print("Checked predictions on ", len(image_names), " images.")
    print("Automated rgb, d, rgd percentages: ",
          np.array(percentages) / len(image_names))
    # print incorrectly classified image names, and the channel format
    print("Automatically-identified incorrect preds ", incorrect_preds)

    if manual:
        print("Manual rgb, d, rgd percentages: ",
              np.array(manual_percentages) / len(image_names))
        print("Manually-identified incorrect preds ", manual_incorrect_preds)


class Predictions:
    # Store grasp validation info in a dictionary and calculate accuracy
    # from it. Can also save the dictionary in a json and load it in later.
    filename = None
    accuracies: Dict[str, bool] = {}

    def __init__(self, filename):
        with open(filename, 'r') as f:
            data = f.read()
            # print(type(data))
            # print(data)
            self.accuracies = json.loads(data)
        self.filename = filename

    def get_dict(self):
        return self.accuracies

    def get_accuracy_rate(self):
        return sum(self.accuracies.values()) / len(self.accuracies) * 100

    def length(self):
        return len(self.accuracies)

    """saves a dicitonary to json file"""

    def save(self, dict):
        saver = json.dumps(dict)
        f = open(self.filename, "w")
        f.write(saver)
        f.close()


def cropnscale(img):
    # Cropping for testing with the BIGBIRD dataset
    # cropping
    # y first x second
    cropped = img[220:370, 210:410]
    resized = cv2.resize(cropped, (640, 480))
    # cv2.imshow("cropped bgrd", cropped)
    # cv2.waitKey(0)
    return resized


def detect_img(img):
    """Runs the grasp detection code that uses edge detection on [img] and
    returns the grasping points."""
    img = cv2.imread(img)
    # img = cv2.resize(img, (640, 480))
    height, width, channels = img.shape
    print("width", width, "height", height, "channels", channels)

    return grab_points(0, 0, width, height, img)


def detect_many(images):
    # runs edgedetect on an image set, prints accuracy and processing time
    # shows preds one by one. user inputs if grasps correctly determined or not
    directory = r'images'
    accuracies = {}
    times = []

    for entry in os.scandir(directory):
        if (entry.path.endswith(".jpg")
                or entry.path.endswith(".png")) and entry.is_file():
            print(entry.name)

            # add timer here too
            detect_img(entry.name)
            # times.append

            # user input to confirm whether grasp was correct
            userinput = input("Is the grasp valid? (y/n) \n")
            while userinput != "y" and userinput != "n":
                print("Invalid input. Please enter y/n")
                userinput = input("Is the grasp valid? (y/n) \n")

            validGrasp = False
            if userinput == "y":
                validGrasp = True

            # storing user input
            accuracies[entry.name] = validGrasp

    preds = Predictions("rgb_preds.json")
    preds.save(accuracies)
    # Calculate and print accuracy
    # accuracy = sum(accuracies.values()) / len(accuracies) * 100
    # print("Avg classification time:", "seconds")
    print("Accuracy: ", preds.get_accuracy_rate(), "%")
    print("Accurately predicted grasps for ", sum(
        preds.get_dict().values()), "/", preds.length(), "grasps")
    # more detailed breakdown of which cases failed
    print("The following images were misclassified:\n")
    for img, value in preds.get_dict().items():
        if value == False:
            print(img, ", ")


def main():
    images = "Samples/b1f4459eae651d4a1e4a40c5ca6a557d/"  # default directory
    rgb_ending = "_RGB.png"
    depth_ending = "_stereo_depth.tiff"
    manual_enabled = True
    if len(sys.argv) > 1:
        images = sys.argv[1]
    if len(sys.argv) > 2:
        manual_enabled = sys.argv[2] == "True"

    # detect_many("images")
    # bgrd("images/NP1_0")
    # detect_bgrd_img("images/NP1_0")
    # detect_img("images/NP1_0.jpg")
    # Samples/ffe702c059d0fe5e6617a7fd9720002b/ roun obj
    # Samples/1a9fa4c269cfcc1b738e43095496b061/ grid obj
    # Samples/2f486fc43af412df22ebff72a57eff4/ soccer ball
    # Samples/244cc8707720b66bac7e7b41e1ebe3e4/ spiderman figure
    # Samples/357e8e4114ebc429e1720743367d35c4/ shoes
    # Samples/2206259551e94d263321831d2245cf06/ rack
    # Samples/b1f4459eae651d4a1e4a40c5ca6a557d/ shampoo
    # true_grasps("Samples/2206259551e94d263321831d2245cf06/",
    #            "1_2206259551e94d263321831d2245cf06", "_RGB.png", "_grasps.txt")
    # true_grasps("Samples/1a9fa4c269cfcc1b738e43095496b061/",
    #            "3_1a9fa4c269cfcc1b738e43095496b061", "_RGB.png", "_grasps.txt")
    process_images(images,
                   rgb_ending, depth_ending, manual=manual_enabled)


main()
