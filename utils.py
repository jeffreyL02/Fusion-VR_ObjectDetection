import numpy as np
import cv2 as cv
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils

def load_image_into_numpy_array(path):
  """Load a frame into a numpy array.
  
  Puts frame into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    frame: single frame from video

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  return np.array(cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB))

def get_object_detection(detect_fn, image_np, category_index, conf_thresh):
    """Perform inference on an image and return extracted locations

    Takes in an image and performs inference using the loaded in model. Based on
    classes and confidence threshold the function will return the detection 
    results (locations, classes found, and confidence scores). 

    Args:
    detect_fn: Saved model loaded in
    image_np: Single frame with type np array 
    category_index: Dict mapping ints to a dict containing categories
    conf_thresh: Min confidence score 

    Returns:
    Detection results from inference as a dict
    Image with bounding box results as np array
    """
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=conf_thresh,
        agnostic_mode=False)

    return detections, image_np_with_detections