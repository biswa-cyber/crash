import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def frame_capture(path):
    vid_obj = cv2.VideoCapture(path)
    count = 0
    success = 1
    while success:
        success, image = vid_obj.read()
        cv2.imwrite("frame%d.jpg" % count, image)
        count += 1


def detect_humans_from_frames():
    for imagePath in paths.list_images("frames"):
        # load the image and resize it to
        # (1) reduce detection time
        # (2) improve detection accuracy
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=min(400, image.shape[1]))
        orig = image.copy()
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image,
                                                winStride=(4, 4),
                                                padding=(8, 8),
                                                scale=1.05)
        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # show some information on the number of bounding boxes
        filename = imagePath[imagePath.rfind("/") + 1:]
        print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))
        # show the output images
        cv2.imshow("Before NMS", orig)
        cv2.imshow("After NMS", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    #frame_capture("video.mp4")
    detect_humans_from_frames()
