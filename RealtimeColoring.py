# import the necessary packages
import collections
from imutils.video import VideoStream
from imutils import face_utils
from imutils import paths
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import os




FACIAL_LANDMARKS_IDXS = collections.OrderedDict([
    ("mouth", (48,68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-w","--watermark",required=True,
 #               help="path to watermark image (assumed to be transparent PNG)")
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
                help="whether or not the Raspberry Pi camera should be used")

args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# load the watermark image, making sure we retain the 4th channel
# which contains the alpha transparency


# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        i=0
        cv2.putText(frame, "Hearts #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        cv2.ellipse(frame, (324, 174), (25, 12), 0, 0, 360, (210, 205, 255), 2)
        cv2.ellipse(frame, (451, 174), (25, 12), 0, 0, 360, (210, 205, 255), 2)

        cv2.ellipse(frame, (330, 228), (25, 12), 30, 0, 360, (210, 205, 255), 2)
        cv2.ellipse(frame, (445, 228), (25, 12), -30, 0, 360, (210, 205, 255), 2)


        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(frame,str(i+1),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255),1)

            (j,k)=FACIAL_LANDMARKS_IDXS["jaw"]

            #(col) jaw 9 to nose 28
            pts=shape[j:k]
            #(row) jaw 4 to jaw 14
           # print("(row)Jaw landmark 4 :"+pts+"Jaw landmark 14 :"+pts)


    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

