# ----------------------------------
# create data in this file
# ----------------------------------

# organize imports
import cv2
import imutils
import numpy as np

# global variables

bg = None  # if background is first frame

# ------------------------------------------------
# Function to find the running average over the background
# running average between background model and current frame (30 frames)
# running average of those frames is Background
# ------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:  # if the bg is first frame, then initialize it with current frame
        bg = image.copy().astype("float")
        return

    # compute running average over bg model and current frame
    # compute weighted average, accumulate it and update the background
    # formula dst(x,y) = (1-a).dst(x,y) + a.src(x,y)
    # src(x,y) source/input img (1 or 3 channel, 8 or 32-bit floating point)
    # dst(x,y) destination/output img
    # a weight of  the source/input img
    cv2.accumulateWeighted(image, bg, aWeight)

# -----------------------------------------------
# To segment the region of the hand in the image
# takes 2 parameters: current frame and threshold (to make black & white the difference image)
# -----------------------------------------------
def segment(image, threshold=25):
    global bg
    # 1st : find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # 2nd : threshold the diff image so that we get the foreground (reveal only hand region)
    thresholded = cv2.threshold(diff, threshold, 225, cv2.THRESH_BINARY)[1]

    # 3rd : perform contours extraction on thresholded (black & white) image
    # to take the contour with the largest area (our hand)
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
    # return thresholded image and segmented image as tuple
    # tuple? to store multiple items in a single variable
    # is a collection which is ordered and unchangeable
    # Python 4 built-in data types: tuple, list, set and dictionary
    # if x(n) = {1, if n>=threshold; 0, if n<threshold}
    # x(n) pixel intensity of input image
    # threshold: decides how to segment/threshold image into binary img


# --------------
# MAIN FUNCTION
# --------------
if __name__ == "__main__":
    # initialize weight for running average
    # if you set a lower value, the running avg over larger amount
    # of previous frames and vice-versa
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # instead of recognizing gesture from whole video sequence
    # just minimize the area that the system has to look for hand region
    # to highlight his region, we use cv2.rectangle()
    # this function needs the following ROI coordinates
    # this function will be used after while loop

    # region of interest (ROI) coordinates 10, 350, 225, 590
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    # to keep track of frame count
    num_frames = 0
    element = 10
    num_imgs_taken = 0

    # we start infinite loop and
    # keep looping, until interrupted
    while (True):

        # read frame from webcam using camera.read()
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize input frame to a fixed width
        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame (current frame)
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI = recognizing zone
        # using simple NumPy slicing
        roi = frame[top:bottom, right:left]

        # convert the ROI to grayscale and blur it
        # use blur to minimize the high frequency components in the image
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Hand Gray-Scale", gray)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        cv2.imshow("Hand GaussianBlur", gray)


        # to get the background, keep looking till a threshold is reached
        # till all image is covered white pixels (hand) black pixels (not hand)
        if num_frames < 30:
            # until we get past 30 frames
            # we keep adding input frame to our run_avg()
            # and update background model
            # NOTE: keep camera without any motion
            # or else, entire algo fails
            cv2.putText(clone, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2)
            run_avg(gray, aWeight)


        else:

            # after updating bg model,
            # current input frame is passed into segment()
            # segment the hand
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                # thresholded and segmented images are returned
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                # segmented contour is drawn over the frame using cv2.drawContours()
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 225))
                # thresholded output is shown using cv2.imshow()
                #cv2.imshow("Thresholded Hand", thresholded)
                if num_imgs_taken <= 300:
                    thresholded, hand_segment = hand

                    cv2.imshow("Thesholded Hand Image", thresholded)

                    #thresholded = cv2.resize(thresholded, (64, 64))
                    #thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                    #thresholded = np.reshape(thresholded, (1, thresholded.shape[0], thresholded.shape[1], 3))

                    #cv2.imshow("Thresholded Hand", thresholded)
                    # cv2.imwrite(r"D:\\gesture\\train\\"+str(element)+"\\" + str(num_imgs_taken+300) + '.jpg', thresholded)
                     #cv2.imwrite(r"C:\\Users\\HadilAbdulhakim\\PycharmProjects\\Hand Tracking\\code\\gesture\\train\\11\\" + "\\" + str(
                      #      num_imgs_taken) + '.jpg', thresholded)
                else:
                    break
                num_imgs_taken += 1


        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 225, 0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand region in the current frame
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break;

# free up memory
camera.release()
cv2.destroyAllWindows()