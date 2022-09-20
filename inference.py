"""Extract object coordinates from batch of images and saves bounding boxes.

Loads the saved custom object detection model for avatar and arrow detection.
BEAM detection model extracts locations of avatar and locations, color,and
orienation of arrow. N-BACK detection model extracts locations of avatar and 
locations and orientation of arrow. Parses given directory for .jpg files and 
performs inference on each one. Draws bounding boxes on the image and saves the
results for each image. 
"""

import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from uuid import uuid4
from object_detection.utils import label_map_util
from utils import load_image_into_numpy_array
from utils import get_object_detection

# Arbitary choice; may change
MIN_CONF_THRESH = 0.6
PATH_TO_SAVED_MODEL = r'D:\work-stuffs\final\Custom_OD_BEAM\exported_model\saved_model'
PATH_TO_LABELS=r'D:\work-stuffs\final\Custom_OD_BEAM\annotations\label_map.pbtxt'
PATH_TO_IMAGES=r'D:\work-stuffs\final\Custom_OD_BEAM\images\collectedimages\left'

if __name__ == "__main__":
  print('Loading model...', end='')
  start_time = time.time()
  detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
  end_time = time.time()
  elapsed_time = end_time - start_time
  print('Done! Took {} seconds'.format(elapsed_time))

  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

  # Go through all jpg files and save a separate jpg with bounding boxes
  for root, dirs, files in os.walk(PATH_TO_IMAGES):
    for file in files:
      if(file.endswith(".jpg")):
        image_np = load_image_into_numpy_array(os.path.join(root,file))
        _, detection_img = get_object_detection(detect_fn, image_np, 
                                                category_index, MIN_CONF_THRESH)

        plt.figure()
        plt.imshow(detection_img)
        print(file + 'Done')
        plt.savefig("{}.png".format(str(uuid4())))
        plt.clf()
  plt.close()