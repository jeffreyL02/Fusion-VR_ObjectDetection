"""Play a video and saves a frame as a jpg file when input is pressed.

Play the video found in the path and press either d or f to save the frame.
d and f are arbitary choices to map to left and right. Purpose of this script is 
to easily collect images from VR_Videos to create test/train dataset. Use these
images in LabelImg GUI to label classes and retrain model. 
"""

import cv2 as cv
from time import sleep
from uuid import uuid4

VIDEO_PATH = 'D:/work-stuffs/Fusion-VR/vision/VR_videos/BEAM/V007_BEAM.mp4'

cap = cv.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    key = cv.waitKey(1)
    cv.imshow('Frame',frame)

    # Press d to save the frame showing 
    if key == ord('d'):
        _, img = cap.read()
        cv.imwrite("./Custom_OD_BEAM/images/collectedimages/left/{}.jpg".format(str(uuid4())), img)


    if key == ord('f'):
        _, img = cap.read()
        cv.imwrite("./Custom_OD_BEAM/images/collectedimages/right/{}.jpg".format(str(uuid4())), img)

    if key == 27:
        break

    sleep(1/30)

cap.release()
cv.destroyAllWindows()