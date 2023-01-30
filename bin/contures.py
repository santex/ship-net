import time
#import ffmpeg
import cv2 as cv
import numpy as np
import os
import re
import fnmatch
import datetime
import time
import argparse
import glob
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
import scipy
from PIL import Image
import imagehash
import tempfile
    
from keras import backend as K
from os.path import isfile, join, basename




""" TableDetector """
def detect(self, white_mask):
        open_mask = cv.morphologyEx(white_mask, cv.MORPH_OPEN, np.ones((4, 4), np.uint8))
        close_mask = cv.morphologyEx(open_mask, cv.MORPH_CLOSE, np.ones((20, 20), np.uint8))

        cnts = cv.findContours(close_mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)[-2]
        if len(cnts) >= 2:
            sort = sorted(cnts, key=cv.contourArea, reverse=True)

            cmax = sort[0]
            epsilon = ARC_EPSILON * cv.arcLength(cmax, True)
            self.out_table = cv.approxPolyDP(cmax, epsilon, True)

            cmax2 = sort[1]
            epsilon = ARC_EPSILON * cv.arcLength(cmax2, True)
            self.in_table = cv.approxPolyDP(cmax2, epsilon, True)
        else:
            raise Exception 



""" snippet """
def findSignificantContours (img, sobel_8u):
    image, contours, heirarchy = cv.findContours(sobel_8u, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)

    # From among them, find the contours with large surface area.
    significant = []
    tooSmall = sobel_8u.size * 5 / 100 # If contour isn't covering 5% of total area of image then it probably is too small
    for tupl in level1:
        contour = contours[tupl[0]];
        area = cv.contourArea(contour)
        if area > tooSmall:
            cv.drawContours(img, [contour], 0, (0,255,0),2, cv.LINE_AA, maxLevel=1)
            significant.append([contour, area])

    significant.sort(key=lambda x: x[1])
    return [x[0] for x in significant]; 


""" camera.py """
def _getContours(self, mask):
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        area_rects = []
        areas = []
        for i in xrange(len(contours)):
            area = cv.contourArea(contours[i])
            if area < 2000:
                continue
            center, size, angle = cv.minAreaRect(contours[i])
            if angle < -45:
                size = tuple(reversed(size))
                angle = angle + 90
            area_rects.append((center, size, angle))
            areas.append(area)
        return area_rects, areas 



""" misc.py """
def detect_iris(frame, marks, side='left'):
    """
    return:
       x: the x coordinate of the iris.
       y: the y coordinate of the iris.
       x_rate: how much the iris is toward the left. 0 means totally left and 1 is totally right.
       y_rate: how much the iris is toward the top. 0 means totally top and 1 is totally bottom.
    """
    mask = np.full(frame.shape[:2], 255, np.uint8)
    if side == 'left':
        region = marks[36:42].astype(np.int32)
    elif side == 'right':
        region = marks[42:48].astype(np.int32)
    try:
        cv.fillPoly(mask, [region], (0, 0, 0))
        eye = cv.bitwise_not(frame, frame.copy(), mask=mask)
        # Cropping on the eye
        margin = 4
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        eye = eye[min_y:max_y, min_x:max_x]
        eye = cv.cvtColor(eye, cv.COLOR_RGB2GRAY)

        eye_binarized = cv.threshold(eye, np.quantile(eye, 0.2), 255, cv.THRESH_BINARY)[1]
        contours, _ = cv.findContours(eye_binarized, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv.contourArea)
        moments = cv.moments(contours[-2])
        x = int(moments['m10'] / moments['m00']) + min_x
        y = int(moments['m01'] / moments['m00']) + min_y
        return x, y, (x-min_x-margin)/(max_x-min_x-2*margin), (y-min_y-margin)/(max_y-min_y-2*margin)
    except:
        return 0, 0, 0.5, 0.5 


""" HandRecognition.py """
def hand_contour_find(contours):
    max_area=0
    largest_contour=-1
    for i in range(len(contours)):
        cont=contours[i]
        area=cv.contourArea(cont)
        if(area>max_area):
            max_area=area
            largest_contour=i
    if(largest_contour==-1):
        return False,0
    else:
        h_contour=contours[largest_contour]
        return True,h_contour

# 4. Detect & mark fingers 


""" SudokuExtractor.py """
def find_corners_of_largest_polygon(img):
    """Finds the 4 extreme corners of the largest contour in the image."""
    opencv_version = cv.__version__.split('.')[0]
    if opencv_version == '3':
        _, contours, h = cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Find contours
    else:
        contours, h = cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = sorted(contours, key=cv.contourArea, reverse=True)  # Sort by area, descending
    polygon = contours[0]  # Largest image

    # Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
    # Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

    # Bottom-right point has the largest (x + y) value
    # Top-left has point smallest (x + y) value
    # Bottom-left point has smallest (x - y) value
    # Top-right point has largest (x - y) value
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    # Return an array of all 4 points using the indices
    # Each point is in its own array of one coordinate
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]] 



""" cleanData.py """
def getContours(thresh, orig):
    (cnts, _) = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[-2:]

    # calculate the area of each contour
    area = np.zeros(len(cnts))
    for i, contour in enumerate(cnts):
        area[i] = cv.contourArea(contour)
        # print 'area: ', area[i]

    # filter contours by area
    cnts_filt_indx = [i for i,v in enumerate(area) if v > CONTOUR_AREA_THRESH]

    
    # draw contours on original image
    for cont_i in enumerate(cnts_filt_indx):
        cnt = cnts[cont_i[1]]
        cv.drawContours(orig, [cnt], 0, (0,255,0), 3) 


""" mertracking """
def get_best_contour(imgmask, threshold):
        im, contours, hierarchy = cv.findContours(imgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        best_area = threshold
        best_cnt = None
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > best_area:
                best_area = area
                best_cnt = cnt
        return best_cnt 




""" glass.py """
def get_shape_contour(im):   
    # Now we have to find the contours of the image. This is the money-maker
    # in that this gives us the ultimate points that we will use to fit the
    # spline regression. Hopefully, all the thresholding/morphology work above
    # should give us one giant coherant shape, and so the contour should be
    # straightforward, smooth, and should give us the shape we want.
    # NOTE: For some reason the curve points don't match the image exactly, and
    # so the offset parameter corrects that. Thus far it seems consistent across
    # images, and the offset appears to be a function of the morphology parameters
    # More validation and testing could figure out how to remove or better
    # compensate for this variation.
    image, contours, hierarchy = cv.findContours(im, cv.RETR_TREE,
                                                  cv.CHAIN_APPROX_SIMPLE,
                                                  offset=(-10,-15))
    # Now that we have the contours, find the one with the biggest area
    # This should be the overall shape.
    areas = [] #list to hold all areas
    for contour in contours:
        ar = cv.contourArea(contour)
        areas.append(ar)
    max_area = max(areas)
    max_area_index = areas.index(max_area) #index of the list element with largest area
    biggest_contour = contours[max_area_index] #largest area contour
    # This function draws a useful image that we'll need later for debugging
    # So return it for now.
    # TODO: fix this so it's in some kind of "verbose" or "debug" tag, etc.
    img = cv.drawContours(image, contours, max_area_index, (0,255,0), 3)
    cv.imshow('img',img)
    #iplot(img)
    return biggest_contour,img 


""" contours """
def filter_min_area(contours, min_area):
    """
    Filter a list of contours, requiring
    a polygon to have an area of >= min_area
    to pass the filter.

    Uses OpenCV's contourArea for fast area computation

    Returns a list of contours.
    """
    return list(filter(lambda cnt: cv.contourArea(cnt) >= min_area,
        contours)) 


""" contours """
def filter_max_area(contours, max_area):
    """
    Filter a list of contours, requiring
    a polygon to have an area of <= max_area
    to pass the filter.

    Uses OpenCV's contourArea for fast area computation

    Returns a list of contours.
    """
    return list(filter(lambda cnt: cv.contourArea(cnt) <= max_area,
        contours)) 


""" contours """
def sort_by_area(contours, reverse=True):
    """
    Sort a list of contours by area
    """
    return sorted(contours, key=cv.contourArea, reverse=True) 


""" motion.py """
def filter_prediction(self, output, image):
        if len(output) < 2:
            return pd.DataFrame()
        else:
            df = pd.DataFrame(output)
            df = df.assign(
                    area=lambda x: df[0].apply(lambda x: cv.contourArea(x)),
                    bounding=lambda x: df[0].apply(lambda x: cv.boundingRect(x))
                    )
            df = df[df['area'] > MIN_AREA]
            df_filtered = pd.DataFrame(
                    df['bounding'].values.tolist(), columns=['x1', 'y1', 'w', 'h'])
            df_filtered = df_filtered.assign(
                    x1=lambda x: x['x1'].clip(0),
                    y1=lambda x: x['y1'].clip(0),
                    x2=lambda x: (x['x1'] + x['w']),
                    y2=lambda x: (x['y1'] + x['h']),
                    label=lambda x: x.index.astype(str),
                    class_name=lambda x: x.index.astype(str),
                    )
            return df_filtered 


""" detection """
def detect_motion(bgsub, frame):
    threshold = 127
    blurk = 13
    min_area = 100

    fgmask = bgsub.apply(frame)
    # cv.imshow('Mask', fgmask) # cv.resize(fgmask, (0,0), fx = 0.5, fy = 0.5))
    cv.GaussianBlur(fgmask, (blurk, blurk), 0, dst = fgmask)
    # cv.imshow('MaskBlur', fgmask) # cv.resize(fgmask, (0,0), fx = 0.5, fy = 0.5))
    cv.threshold(fgmask, threshold, 255, cv.THRESH_BINARY, dst = fgmask)
    # cv.imshow('Thresh', fgmask) # cv.resize(fgmask, (0,0), fx = 0.5, fy = 0.5))

    cnts = cv.findContours(fgmask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1]
    cnts = [c for c in cnts if cv.contourArea(c) > min_area]
    return cnts 


""" preprocess """
def img_preprocess(img):
    img_rgb = cv.resize(img, (1000, 1600))



    img_save = img_change(img_rgb)
    gray_b = cv.split(img_save)[0]
    ret, binary = cv.threshold(gray_b, 180, 255, cv.THRESH_BINARY)
    ele = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    eroded = cv.erode(binary, ele)
    ele = cv.getStructuringElement(cv.MORPH_RECT, (60, 60))
    diated = cv.dilate(eroded, ele)
    image, contours, hierarchy = cv.findContours(diated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)
        # areas.append(area)
        if area < 400000:
            continue
        rect = cv.minAreaRect(cnt)
        img_crop = crop_minAreaRect(img_rgb, rect)

    img_crop = cv.resize(img_crop, (1600, 1000))
    return img_crop 


""" document.py """
def detect_edge(self, image, enabled_transform = False):
        dst = None
        orig = image.copy()

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edged = cv.Canny(blurred, 0, 20)
        _, contours, _ = cv.findContours(edged, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        contours = sorted(contours, key=cv.contourArea, reverse=True)

        for cnt in contours:
            epsilon = 0.051 * cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4:
                target = approx
                cv.drawContours(image, [target], -1, (0, 255, 0), 2)

                if enabled_transform:
                    approx = rect.rectify(target)
                    # pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]])
                    # M = cv.getPerspectiveTransform(approx,pts2)
                    # dst = cv.warpPerspective(orig,M,(800,800))
                    dst = self.four_point_transform(orig, approx)
                break

        return image, dst 



def combine_detections(frame, first_frame, gray):
        detected = False
        frame_delta = cv.absdiff(first_frame, gray)
        text = self.text_detected_false()
        thresh = cv.threshold(frame_delta, 25, 255, cv.THRESH_BINARY)[1]
        thresh = cv.dilate(thresh, None, iterations=2)

        (_, contours, _) = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for c in sort_contours(contours):
            if cv.contourArea(c) < self.min_area:
                continue
            # detected
            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = self.text_detected()
            detected = True

        cv.putText(frame, "Status: {}".format(text), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return frame, detected 

""" camera.py """
def frame_draw_detections(self, frame, first_frame, gray):
        detected = False
        frame_delta = cv.absdiff(first_frame, gray)
        text = self.text_detected_false()
        thresh = cv.threshold(frame_delta, 25, 255, cv.THRESH_BINARY)[1]
        thresh = cv.dilate(thresh, None, iterations=2)

        (_, contours, _) = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv.contourArea(c) < self.min_area:
                continue
            # detected
            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = self.text_detected()
            detected = True

        cv.putText(frame, "Status: {}".format(text), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return frame, detected 


""" dista """
def find_marker(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(gray, 255, 255)
    (cnts, _) = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key = cv.contourArea)

    # compute the bounding box of the of the paper region and return it
    return cv.minAreaRect(c) 


""" split_pdf.py """
def find_box_using_opencv(image, min_width, min_height, max_width, max_height, debug):    #find a slide/box in an image (should only pass images that contain a single slide)
    lower_bound_pixel = 0 #values used in colour thresholding
    upper_bound_pixel = 5
    opencv_image = numpy.array(image) # convert to open cv image

    #open cv tings start
    grayscale_img = cv.cvtColor(opencv_image, cv.COLOR_RGB2GRAY)

    mask = cv.inRange(grayscale_img, lower_bound_pixel,upper_bound_pixel) #find black elements (assuming black boxes)
    contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]#find contours in mask

    if len(contours) > 0:
        slide_coords = max(contours, key = cv.contourArea) #get the biggest contour
        if len(slide_coords) == 4: #ensure precisely 4 coords, slides are squares

                if debug:
                    cv.drawContours(opencv_image, slide_coords, -1, (0,255,0), 3)
                    cv.imwrite("contour.png", opencv_image)

                slide_found_width = (slide_coords[2][0])[0] - (slide_coords[0][0])[0] #get width and heihgt
                slide_found_height = (slide_coords[2][0])[1] - (slide_coords[0][0])[1]

                #ensure found width and height is between allowed bounds
                if(slide_found_width > min_width and slide_found_width < max_width and 
                    slide_found_height > min_height and slide_found_height < max_height):
                    return slide_coords
    else:
        return None

#used when finding 4 slides 


""" split_pdf.py """
def get_largest_contours(contours):
    
    three_largest_contours = []
    indexes_with_more_than_four_points = []

    # get the indexes of contours with more than 4 points, since they are not slides! 
    for i in range(0,len(contours)-1):
        if len(contours[i]) > 3:
            indexes_with_more_than_four_points.append(i)

    return indexes_with_more_than_four_points


#return the contour coordinates of all 3 slides in the left half of the image 


""" split_pdf.py """
def find_left_slides_using_opencv(image, min_width, min_height, max_width, max_height):
    
    lower_bound_pixel = 0 #values used in colour thresholding
    upper_bound_pixel = 5
    opencv_image = numpy.array(image) # convert to open cv image
    second_image = opencv_image.copy()

    min_slide_area = min_width * min_height
    max_slide_area = max_width * max_height

    #open cv tings start
    grayscale_img = cv.cvtColor(opencv_image, cv.COLOR_RGB2GRAY)

    mask = cv.inRange(grayscale_img, lower_bound_pixel,upper_bound_pixel) #find black elements (assuming black boxes)
    contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]#find contours in mask

    if len(contours) > 2: #make sure at least 3 contours were found in the image
        three_largest_contours = get_three_largest_contours(contours)  #get the three largest contours
        

        #check that the returned contours are inside the allowed bounds UGLY HACK
        slides_inside_bounds = (cv.contourArea(three_largest_contours[0]) > min_slide_area and cv.contourArea(three_largest_contours[0]) < max_slide_area)
        slides_inside_bounds = slides_inside_bounds and (cv.contourArea(three_largest_contours[1]) > min_slide_area and cv.contourArea(three_largest_contours[1]) < max_slide_area)
        slides_inside_bounds = slides_inside_bounds and (cv.contourArea(three_largest_contours[2]) > min_slide_area and cv.contourArea(three_largest_contours[2]) < max_slide_area)
        
        if slides_inside_bounds:
            return three_largest_contours
        else:
            return None
    else:
        return None #return empty list


#used when finding 6 slides total, returns the coordinates of all slides in the left half of the image 


""" TripletSubmit.py """
def extract_largest(mask, n_objects):
    contours, _ = cv.findContours(
        mask.copy(), cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE
    )
    areas = [cv.contourArea(c) for c in contours]
    contours = np.array(contours)[np.argsort(areas)[::-1]]
    background = np.zeros(mask.shape, np.uint8)
    choosen = cv.drawContours(
        background, contours[:n_objects],
        -1, (255), thickness=cv.FILLED
    )
    return choosen 


""" TripletSubmit.py """
def remove_smallest(mask, min_contour_area):
    contours, _ = cv.findContours(
        mask.copy(), cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv.contourArea(c) > min_contour_area]

    background = np.zeros(mask.shape, np.uint8)
    choosen = cv.drawContours(
        background, contours,
        -1, (255), thickness=cv.FILLED
    )
    return choosen 


""" autocapture """
def getMask(self, img1, img2):
        # Subtract two images showing inverted pattern. This blacks out non-display areas that only have diffuse light.
        imgdiff1 = cv.subtract(img1, img2)
        self.putImage(imgdiff1, 'imgdiff1')
        imgdiff2 = cv.subtract(img2, img1)
        self.putImage(imgdiff2, 'imgdiff2')

        # Add two diff images together to fill visible area, then threshold it to create mask
        imgdiff = cv.add(imgdiff1, imgdiff2)
        self.putImage(imgdiff, 'imgdiff')
        img = cv.cvtColor(imgdiff, cv.COLOR_BGR2GRAY)
        img = cv.medianBlur(img, 13)
        ret, mask = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        self.putImage(mask, 'mask')

        # Detect closed contours in the image, then pick the second largest which should be the projector's main visible area
        # The largest contour is a border around the entire picture
        border = 10
        maskb = cv.copyMakeBorder(mask, top=border, bottom=border, left=border, right=border, borderType=cv.BORDER_CONSTANT, value=[255, 255, 255] )

        (cnts, x) = cv.findContours(maskb, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        logger.info("Found %s contours in mask image", len(cnts))
        cnts = sorted(cnts, key = cv.contourArea, reverse = True)[1:2]
        #cv.drawContours(mask2, cnts, -1, (255,0,0), 2)

        mask2 = np.full(maskb.shape, 255, np.uint8)
        cv.fillPoly(mask2, pts=cnts, color=(0,0,0))
        mask2 = mask2[border:-border, border:-border]
        self.putImage(mask2, 'mask2')

        return mask2 


""" nms_rotate.py """
def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):

    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            int_pts = cv.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv.convexHull(int_pts, returnPoints=True)

                int_area = cv.contourArea(order_pts)

                inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + cfgs.EPSILON)

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64) 


""" BallDetector """
def inside_detect(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_white, upper_white)
        cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            cmax = max(cnts, key=cv.contourArea)
            M = cv.moments(cmax)
            if M["m00"] > 0:
                center = (M["m10"] / M["m00"], M["m01"] / M["m00"])
        return center 


""" demo_segmentation """
def draw_bounding_box(img, mask, area_threshold=100):
    b, c = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in b:
        area = cv.contourArea(cnt)
        if area > area_threshold:
            hull = cv.convexHull(cnt)
            cv.drawContours(img, [hull], 0, (50, 128, 30), -1)

    return img 


""" snippet """
def get_contours(im, min_area=0, max_area=None):
    _im = im.copy()
    contours, _ = cv.findContours(_im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    areas = [cv.contourArea(c) for c in contours]
    if max_area is None:
        return [c for c, a in zip(contours, areas) if min_area <= a]
    return [c for c, a in zip(contours, areas) if min_area <= a <= max_area] 


""" size_detector.py """
def _find_size_candidates(self, image):
        binary_image = self._filter_image(image)

        _, contours, _ = cv.findContours(binary_image,
                                          cv.RETR_LIST,
                                          cv.CHAIN_APPROX_SIMPLE)

        size_candidates = []
        for contour in contours:
            bounding_rect = cv.boundingRect(contour)
            contour_area = cv.contourArea(contour)
            if self._is_valid_contour(contour_area, bounding_rect):
                candidate = (bounding_rect[2] + bounding_rect[3]) / 2
                size_candidates.append(candidate)

        return size_candidates 


""" bbox """
def bbox_overlaps(boxes, query_boxes):
    """ Calculate IoU(intersection-over-union) and angle difference for each input boxes and query_boxes. """

    N = boxes.shape[0]
    K = query_boxes.shape[0]
    boxes = np.round(boxes, decimals=2)
    query_boxes = np.round(query_boxes, decimals=2)
    overlaps = np.reshape(np.zeros((N, K)), (N,K))
    delta_theta = np.reshape(np.zeros((N, K)), (N,K))

    for k in range(K):
        rect1 = ((query_boxes[k][0], query_boxes[k][1]),
                 (query_boxes[k][2], query_boxes[k][3]),
                 query_boxes[k][4])
        for n in range(N):
            rect2 = ((boxes[n][0], boxes[n][1]),
                     (boxes[n][2], boxes[n][3]),
                     boxes[n][4])
            # can check official document of opencv for details
            num_int, points = cv.rotatedRectangleIntersection(rect1, rect2)
            S1 = query_boxes[k][2] * query_boxes[k][3]
            S2 = boxes[n][2] * boxes[n][3]
            if num_int == 1 and len(points) > 2:
                s = cv.contourArea(cv.convexHull(points,returnPoints=True))
                overlaps[n][k] = s / (S1 + S2 - s)
            elif num_int == 2:
                overlaps[n][k] = min(S1, S2) / max(S1, S2)
            delta_theta[n][k] = np.abs(query_boxes[k][4] - boxes[n][4])
    return overlaps, delta_theta 


""" motion_detector.py """
def update(self, image):
        # init set of locations, where motion is detected
        locs = []
        self.status = "Secure"

        # if background has not been initialized, init it
        if self.avg is None:
            self.avg = np.asfarray(np.copy(image))
            return locs

        # accumulate weighted average between current frame and avg. frame
        # get pixel-wise differences between current frame and avg. frame
        cv.accumulateWeighted(image, self.avg, self.accumWeight)
        frameDelta = cv.absdiff(image, cv.convertScaleAbs(self.avg))

        # threshold the delta image and apply a series of dilations
        thresh = cv.threshold(frameDelta, self.deltaThresh, 255,
            cv.THRESH_BINARY)[1]
        thresh = cv.dilate(thresh, None, iterations=2)

        # find contours in the dilated image
        (_, cnts, _) = cv.findContours(np.copy(thresh), cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE)

        # if area of contour is bigger than min area - it's movement
        for c in cnts:
            if cv.contourArea(c) > self.minArea:
                self.status = "Breached"
                locs.append(c)

        return locs 


""" detection.py """
def validate_smoke(frame, recognized_smoke, validation_picture, photo_taken):
    validated_smoke = []

    # whether a frame or the geotagged image is used
    if (photo_taken):
        image = cv.imread(validation_picture)
    else:
        image = validation_picture
    # cv.imwrite("outputs/07_validation.jpg", image)  # vis

    # resizing contours and/or frame
    # resized_image = cv.resize(frame, (image.shape[1], image.shape[0]))
    res_contours = resize_contours(frame.shape, recognized_smoke, image.shape)

    # detecting the contours in the validation image
    grey_photo, color_seg_contours_photo = color_segmentation(image)
    overlapping_contours_v = check_overlap(
        color_seg_contours_photo, grey_photo, [res_contours])

    index = 0
    for c_v in overlapping_contours_v:
        area_d = cv.contourArea(res_contours)
        area_v = cv.contourArea(c_v)
        print("Areas of validated smoke {}: {}; of detected smoke - {}".format(
            index, area_v, area_d))
        if (area_v > area_d):
            validated_smoke.append(c_v)
            cv.drawContours(image, [c_v], -1, (255, 125, 0), 15)
            cv.drawContours(
                image, [res_contours], -1, (0, 125, 255), 15)
            name = "../outputs/08_cnt_valid_reshaped {}.jpg".format(area_v)
            cv.imwrite(name, image)  # visualization

    return res_contours, validated_smoke 


""" squares """
def find_squares(img):
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            contours, hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares 


""" optimize_alignment """
def get_contour_path(image,
                     contour_dilate_ksz=3,
                     contour_dilate_itr=20):
    ''''''


    ### convert to binary (unexplored == occupied)
    ret, bin_img = cv.threshold(image, 200, 255 , cv.THRESH_BINARY)

    ### dilating to expand the contour
    dilate_k = np.ones((contour_dilate_ksz,contour_dilate_ksz),np.uint8)
    bin_img = cv.dilate(bin_img, dilate_k, iterations=contour_dilate_itr)

    ### find contours
    im2, contours, hierarchy = cv.findContours(bin_img, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
    contours_pts = [c[:,0,:] for c in contours]

    ### sort contours according to thier sizes
    areas = [cv.contourArea(c) for c in contours_pts] # [c.shape[0] for c in contours_pts]
    contours_pts = [c for (a,c) in sorted(zip(areas,contours_pts),reverse=True)]

    ### selecting the biggest contour
    # max_idx = [c.shape[0] for c in contours].index(max([c.shape[0] for c in contours]))
    # contour_pts = contours[max_idx]

    # creating one path per contour
    contours_path = [mapali._create_mpath(c) for c in contours_pts]

    return contours_pts, contours_path

################################################################################ 


""" polyIO """
def _reject_small_contours(contours, min_vert, min_area):
    '''
    '''
    for idx in range(len(contours)-1, -1, -1):
        small = False if min_area is None else np.abs(cv.contourArea(contours[idx])) <= float(min_area)
        short = False if min_vert is None else contours[idx].shape[0] <= float(min_vert)
        if small or short: contours.pop(idx)

    return contours


################################################################################ 


""" extract_lines """
def find_items(maze_image):
    # Preprocessing to find the contour of the shapes
    h, w = maze_image.shape[0], maze_image.shape[1]
    dim = (h+w)//2
    b_and_w = cv.cvtColor(maze_image, cv.COLOR_BGR2GRAY)
    edges = cv.GaussianBlur(b_and_w, (11, 11), 0)
    edges = cv.adaptiveThreshold(edges, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 2)

    cv.rectangle(edges,(0, 0),(w-1,h-1),(255,255,255),16)
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    cv.imshow('d', edges)

    items = []

    if contours:
        item_mask = np.zeros(edges.shape, np.uint8)
        conts = sorted(contours, key=lambda x: cv.contourArea(x), reverse=False)

        for cnt in conts:

            if cv.contourArea(cnt) > 0.35*dim:
                return items, item_mask

            elif cv.contourArea(cnt) > 0.05*dim:
                d = np.mean(cnt, axis=0)
                d[0][0], d[0][1] = int(round(d[0][0])), int(round(d[0][1]))

                # TODO adjust the size here?
                if cv.contourArea(cnt) < 0.1*dim:
                    items.append((d, 'smol'))
                    cv.drawContours(item_mask, [cnt], -1, (255,255,255), -1)
                else:
                    items.append((d, 'big'))
                    cv.drawContours(item_mask, [cnt], -1, (255,255,255), -1)

    return items, item_mask 


""" targetDetect.py """
def findLargestContour(source):
    contours, hierarchy = cv.findContours(source, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        ordered = sorted(contours, key = cv.contourArea, reverse = True)[:1]
        return ordered[0] 


""" ball_detect.py """
def findLargestContour(source):
    contours, hierarchy = cv.findContours(source, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        ordered = sorted(contours, key = cv.contourArea, reverse = True)[:1]
        return ordered[0] 


""" john-test.py """
def findLargestContour(source):
    contours, hierarchy = cv.findContours(source, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        ordered = sorted(contours, key = cv.contourArea, reverse = True)[:1]
        return ordered[0] 


""" vision.py """
def findTarget(raw, params): 
    mask = filterHue(raw, params["hue"], params["hueWidth"], params["low"], params["high"])
    #cv.imshow("mask", mask)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        ordered = sorted(contours, key = cv.contourArea, reverse = True)[:1]
        largestContour = ordered[0]
        if largestContour != None and cv.contourArea(largestContour) > params["countourSize"]:
            return largestContour 


def save_video(cap,saving_file_name,fps=64):
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(3600,3600))
        .output(saving_file_name, pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    return process

def natural_sort(l): 
  convert = lambda text: int(text) if text.isdigit() else text.lower() 
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
  return sorted(l, key = alphanum_key)
def recursive_glob(rootdir='.', pattern='jpg'):
  frame_list = [os.path.join(looproot, filename)
    for looproot, _, filenames in os.walk(rootdir)
      for filename in filenames
        if fnmatch.fnmatch(filename, pattern)]
  
  reture = []
  for frame in frame_list:
    reture.append(os.path.dirname(frame))
    
  return np.unique(reture)
def missing_indexes(seq):
  seq.append(0)
  res = [ele for ele in range(max(seq)+1) if ele not in seq] 
  return res
  
def fs_sort(files):
  files = natural_sort(files)
  return files

def search_frames(path):
  files = []
  for r, d, f in os.walk(path):
    print(r)
    return recursive_glob(r,'jpg')

def extract_from_img(file_name):
  try: 
    image=cv.imread(file_name)
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    return gray
  except:
    print("problem '%s'") 

def resize_img(img, scale):
    return scipy.ndimage.zoom(img, scale, order=1)
    
def add_corners(image):
  imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  ret, thresh = cv.threshold(imgray, 0, 25, 0)
  contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
  cnt = contours
  cv.drawContours(image, [cnt], 0, (0,255,0), 3)
  
  #cv.drawContours(image, cnt, -1, (0,255,0), 1)
  return image 
 
def extract_corners(image , file_name,ROI_number):
    frame = image
    base_dir,base_name = os.path.split(file_name)#.replace(".jpg","")
        
    #frame = resize_img(image, 0.5)
    imgray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnts = contours
#    ROI_number = 0
    
    #clipps = args.get('clipps')
    for c in cnts:
        (x, y, w, h) = cv.boundingRect(c)
        ROI = frame[y:y+h, x:x+w]
        go = h * w
        check = y+h        
        # if the contour is too small, ignore it
        if go < 50:
            continue
            
        ROI_number += 1
        
        #cv.drawContours(image, [cnt], 0, (0,255,0), 3)
        cv.drawContours(image, cnts, -1, (0,255,0), 1)
        
      #  image = resize_img(image,2)
        
    return image ,ROI_number

# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]
    # Return original image if no need to resize
    if width is None and height is None:
        return image
    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # Return the resized image
    return cv.resize(image, dim, interpolation=inter)

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    #numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    #(hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
    # normalize the histogram, such that it sums to one
    #hist = hist.astype("float")
    #hist /= hist.sum()
 
    # return the histogram
    return clt

def drawHistogram(img,color=True,windowName='drawHistogram'):
    h = numpy.zeros((300,256,3))
     
    bins = numpy.arange(256).reshape(256,1)
    
    if color:
            channels =[ (255,0,0),(0,255,0),(0,0,255) ];
    else:
            channels = [(255,255,255)];
    
    for ch, col in enumerate(channels):
            hist_item = cv.calcHist([img],[ch],None,[256],[0,255])
            #cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
            hist=numpy.int32(numpy.around(hist_item))
            pts = numpy.column_stack((bins,hist))
            #if ch is 0:
            cv.polylines(h,[pts],False,col)
     
    h=numpy.flipud(h)
     
    cv.imshow(windowName,h); 

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
 
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX
    
    # return the bar chart
    return bar    

def rescale_by_height(image, target_height, method=cv.INTER_LANCZOS4):
    """Rescale `image` to `target_height` (preserving aspect ratio)."""
    w = int(round(target_height * image.shape[1] / image.shape[0]))
    return cv.resize(image, (w, target_height), interpolation=method)

def rescale_by_width(image, target_width, method=cv.INTER_LANCZOS4):
    """Rescale `image` to `target_width` (preserving aspect ratio)."""
    h = int(round(target_width * image.shape[0] / image.shape[1]))
    return cv.resize(image, (target_width, h), interpolation=method)
        
def count_colours(src):
    unique, counts = np.unique(src, return_counts=True)
    print(counts.size)
    return counts.size
    
def cccSat(c , frame, file_name,ROI_number,messure,bg,slots,i):
    (x, y, w, h) = cv.boundingRect(c)
    image_width = image.shape[0]
    image_height = image.shape[1]
    
    ROI = frame[y:y+h, x:x+w]
    go = h * w
    check = y+h        
    check2 = x
    n_red_pix = np.sum(ROI == 180)
    n_white_pix = np.sum(ROI == 255)
    
    wp = go / 100
    wp = n_white_pix / wp
    if (y > 900 and x < 380):
        print(0)
    else:
        ROIgray = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)
        
        
        if(go>100):
            #ret, thresh = cv.threshold(imgray, 125, 255, 0)
            clipps = args.get('clipps')
            base_dir,base_name = os.path.split(file_name)#.replace(".jpg","")
            date = base_name.split("_")[0]
            time = base_name.split("_")[1]
            sat  = base_name.split("_")[2]
            res  = base_name.split("_")[3].replace('.jpg','')
            #print(base_name.split("_"))
                    
            path = os.path.join(args.get('clipps'),sat,date) 
            
            im = '{}/{}_{}_{}_{}_{}_{}.jpg'.format(clipps,base_name,ROI_number,x,y,w,h)
            cv.imwrite(im, ROI)
            hash = imagehash.average_hash(Image.open(im))
            os.remove(im)
            im2 = '{}/{}_{}_{}_{}_{}_{}_{}.jpg'.format(path,hash,base_name,ROI_number,x,y,w,h)
            cv.imwrite(im2,ROI)
            
            #ROI = rescale_by_width( ROI,200)
            
      
      
            #hist=ROI   
            if ROI is not None and go > messure:
                        
                hist=drawHistogram(ROI)
                hist = cv.resize(hist, (700,300))    
                ROI = cv.resize(ROI, (500,400))    
                n = count_colours(ROI)
        
                if i == 1:
            
                    bg = image_info(i,ROI, bg,image.shape[0]+160,25)
                    #slots += 1
                    #bg = image_info(i,hist,bg,bg.shape[0]-60,103)
                    cv.rectangle(frame, (x, y),(x + w, y + h), (0, 255, 0), 1)
                    messure=go
                    slot[0]=0
                    slot[1]=1
                
                elif i == 2:
            
                    bg = image_info(i,ROI, bg,image.shape[0]+740,25)
                    #slots += 1
                    #bg = image_info(i,hist,bg,bg.shape[0]-60,103)
                    cv.rectangle(frame, (x, y),(x + w, y + h), (0, 255, 0), 1)
                    messure=go
                    slot[1]=0
                    slot[2]=1
                
                elif i == 3:
            
                    bg = image_info(i,ROI, bg,image.shape[0]+160,710)
                    #slots += 1
                    #bg = image_info(i,hist,bg,bg.shape[0]-60,103)
                    cv.rectangle(frame, (x, y),(x + w, y + h), (0, 255, 0), 1)
                    messure=go
                    slot[2]=0
                    slot[3]=1
                
                else:
            
                    bg = image_info(i,ROI, bg,image.shape[0]+740,710)
                    #slots += 1
                    #bg = image_info(i,hist,bg,bg.shape[0]-60,103)
                    cv.rectangle(frame, (x, y),(x + w, y + h), (0, 255, 0), 1)
                    messure=go
                    slot[3]=0
                    slot[0]=1
                    
                
                
            #cv.rectangle(frame, (image_width+130,90),(image_width+470,390), (255, 255, 255), 0,1)
        
                    
    return frame,bg,slot,ROI

def between(n,fr ,to):
    if n > fr and n < to:
        return 1
    else:
        return 0
        

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))


    #print(boundingBoxes)
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

    def gradient_ascent(image, file_name,ROI_number,bg,slot,sety):
        
        
        
#        imagse = maintain_aspect_ratio_resize(image,1050,1050)
        #image = rescale_by_width(image,1340)
        frame = image
        base_dir,base_name = os.path.split(file_name)#.replace(".jpg","")
        date = base_name.split("_")[0]
        time = base_name.split("_")[1]
        sat  = base_name.split("_")[2]
        res  = base_name.split("_")[3].replace('.jpg','')
        #print(base_name.split("_"))
                
        path = os.path.join(args.get('clipps'),sat,date) 
          
        try: 
            os.makedirs(path, exist_ok = True) 
            #print("Directory '%s' created successfully" %path)         
            
            #imutils.resize(image, width=1050, height=1050)
            imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(imgray, 125, 255, 0)
        except:
            print("Directory '%s' can not be created") 
 
 
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv.dilate(thresh, None, iterations=1)
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        

        # sort the contours according to the provided method
        (cnts, boundingBoxes) = sort_contours(cnts, "top-to-bottom")
        
        clipps = args.get('clipps')
# loop over the frames of the video
        # loop over the contours
      #  cv.putText(bg, " {} / {} {}".format(date,time,ROI_number), (int(image.shape[0]/4),image.shape[1]+100),1, 1, (0, 0, 255), 1)
        n = 0
        for (i, c) in enumerate(cnts):
            
            epsilon = 0.01*cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, epsilon, True)
            # Position for writing text
            (x, y, w, h) = cv.boundingRect(c)  
   
            go = w * h
            
            if go<100:
                continue
            
            #if between(x ,350 ,600) != 0 and between(y, 400 ,600) != 0 :
            #    continue
            

            ROI_number = ROI_number + 1
            if sat == "c3" or sat == "c2": 
                frame,bg,slot,ROI = cSat(c, frame,file_name,ROI_number,messure,bg,slot,i)
                sety.append(ROI)
                if type(frame) == 'int':
                    continue
                    
                else:
                    n +=1
            
            
            
        
        
        cv.putText(frame, "Sat: {}".format(sat), (6,30),0, 1, ( 255,  255, 255), 2)
        
     #   cv.putText(bg, " {}".format(n), (int(frame.shape[0]/4)-100,frame.shape[1]+100),1, 1, (0, 0, 255), 2)
        
        return frame,ROI_number,bg,slot,sety
        
def find(path, expr):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file and fnmatch.fnmatch(file,expr):
                files.append(os.path.join(r, file))
    for f in files:
        print(f)
    return np.sort(files)
    
def process(files):
    for i,file_name in enumerate(files):
        image=cv.imread(file_name)
        image = gradient_ascent(image, file_name)
        #image = add_corners(image)
    #    process.stdin.write(cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.uint8).tobytes())        
        
        #time.sleep(0.1)
        
        
        cv.namedWindow(sat)
        cv.resizeWindow(sat, 1600, 1600)
        cv.imshow(sat,image)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
                
def drawHistogram(img,color=True,windowName='drawHistogram'):
    h = np.zeros((300,256,3))
     
    bins = np.arange(256).reshape(256,1)
    
    if color:
            channels =[ (255,0,0),(0,255,0),(0,0,255) ];
    else:
            channels = [(255,255,255)];
    
    for ch, col in enumerate(channels):
            hist_item = cv.calcHist([img],[ch],None,[256],[0,255])
            #cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
            hist=np.int32(np.around(hist_item))
            pts = np.column_stack((bins,hist))
            #if ch is 0:
            cv.polylines(h,[pts],False,col)
    
    cv.normalize(hist,hist,0,255,cv.NORM_MINMAX)
    
    h=np.flipud(h)
    
    return h
def logoOverlay(image,logo,alpha=1.0,x=0, y=0, scale=1.0):
    (h, w) = image.shape[:2]
    image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
    #verlay = cv.resize(logo, None,fx=scale,fy=scale)
    return image
    
def put4ChannelImageOn4ChannelImage(back, fore, x, y):
    rows, cols, channels = fore.shape    
    trans_indices = fore[...,3] != 0 # Where not transparent
    overlay_copy = back[y:y+rows, x:x+cols] 
    overlay_copy[trans_indices] = fore[trans_indices]
    back[y:y+rows, x:x+cols] = overlay_copy
def image_info(post_id,hist, image,x,y):
         
    x_offset=x
    y_offset=y
    image[y_offset:y_offset+hist.shape[0], x_offset:x_offset+hist.shape[1]] = hist
    
    return image
    
def image_add(post_id,hist , image,x,y):
    
    x_offset=x
    y_offset=y
    image[y_offset:y_offset, x_offset:x_offset] = hist
    return image
