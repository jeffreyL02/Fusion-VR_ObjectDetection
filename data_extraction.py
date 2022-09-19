"""Exports object coordinates frame by frame from a video file to a csv file.

Loads the saved custom object detection model for avatar and arrow detection.
BEAM detection model extracts locations of avatar and locations, color,and
orienation of arrow. N-BACK detection model extracts locations of avatar and 
locations and orientation of arrow. Parses video file frame by frame and writes
coordinates to exported csv file. 
"""

import csv
import cv2 as cv
import tensorflow as tf
from object_detection.utils import label_map_util
from utils import load_image_into_numpy_array
from utils import get_object_detection

# Arbitary choice; may change
MIN_CONF_THRESH = 0.6
PATH_TO_SAVED_MODEL = r'D:\work-stuffs\final\Custom_OD_arrowavatarv2\exported_model\saved_model'
PATH_TO_LABELS = r'D:\work-stuffs\final\Custom_OD_arrowavatarv2\annotations\label_map.pbtxt'
PATH_TO_VIDEO = r'D:\work-stuffs\Fusion-VR\vision\VR_videos\BEAM\Recording3beam.mp4'
CSV_FILENAME = 'video.csv'

if __name__ == "__main__":
  detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
  category_index = label_map_util.create_category_index_from_labelmap(
      PATH_TO_LABELS, use_display_name=True)
  
  cap = cv.VideoCapture(PATH_TO_VIDEO)
  frame_count = 0

  with open(CSV_FILENAME, 'wt', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', 
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    # Iterate through every frame of the video and perform inference on each
    # frame. Write the coords of each obj found above MIN_CONF_THRESH to .csv
    while True:
      ret, frame = cap.read()
      frame_count += 1

      if not ret:
        filewriter.close()
        print("Done")

      image_np = load_image_into_numpy_array(frame)
      detections, _ = get_object_detection(detect_fn, image_np, 
                                           category_index, MIN_CONF_THRESH)

      # Find which boxes are above the confidence threshold and write their 
      # locations as a new row to the csv file
      boxes = detections['detection_boxes']
      scores = detections['detection_scores']
      for i in range(boxes.shape[0]):
          if scores is None or scores[i] > MIN_CONF_THRESH:
              # boxes[i] is the coordinates of the box to draw
              print(str(frame_count))
              class_name = category_index[detections['detection_classes'][i]]['name']
              filewriter.writerow([frame_count, class_name, boxes[i]])
