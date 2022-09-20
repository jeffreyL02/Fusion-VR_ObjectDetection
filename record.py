""" Generate TF Record files for model training.

Given a label map, directory to training images, and directory to test images,
generaet the TF Record files for model training. Developed and Modified from the
following source: 
https://medium.com/analytics-vidhya/tensorflow-2-object-detection-api-using-custom-dataset-745f30278446
"""

# importing required libraraies
import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

# Set the folder name for the source annotated XML files and folder #to store the TFRecord Record file
LABEL_MAP_FILE=r'Custom_OD_BEAM/annotations/label_map.pbtxt'
TRAIN_XML_FILE=r'Custom_OD_BEAM/images/Train'
TRAIN_TF_RECORD_DIR=r'Custom_OD_BEAM/Annotations/train.record'
TEST_XML_FILE=r'Custom_OD_BEAM/images/test'
TEST_TF_RECORD_DIR=r'Custom_OD_BEAM/Annotations/test.record'

#Create a dictionary for the labels  or objects
label_map = label_map_util.load_labelmap(LABEL_MAP_FILE)
label_map_dict = label_map_util.get_label_map_dict(label_map)

#convert the object annotation from XML file to a dataframe
def xml_to_df(path):
    """Iterates through all .xml files conatining Annotation in a given 
    directory and combines them in a single Pandas dataframe.
    
    Args:
    path: The path containing the .xml files
    
    Returns:
    The produced dataframe
    """
    ct = 0
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        print (ct)
        ct += 1
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

#pass the label and get its equivalent integer
def class_label_to_int(row_label):
    return label_map_dict[row_label]

#Split filename and the annotations details for all the xml files
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
    
# Create the TF Record files
def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
        
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_label_to_int(row['class']))
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),        
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

#Generating the train TFRecord
writer = tf.python_io.TFRecordWriter(TRAIN_TF_RECORD_DIR)
train_xml = xml_to_df(TRAIN_XML_FILE)
test_xml = xml_to_df(TEST_XML_FILE)
grouped_train = split(train_xml, 'filename')
grouped_test = split(test_xml, 'filename')
for group in grouped_train:
    tf_example = create_tf_example(group, TRAIN_XML_FILE)
    writer.write(tf_example.SerializeToString())
writer.close()
print('Successfully created the TFRecord file: {}'.format(TRAIN_TF_RECORD_DIR))

# Generating the test TFRecord 
writer = tf.python_io.TFRecordWriter(TEST_TF_RECORD_DIR)
for group in grouped_test:
    tf_example = create_tf_example(group, TEST_XML_FILE)
    writer.write(tf_example.SerializeToString())
writer.close()
print('Successfully created the TFRecord file: {}'.format(TEST_TF_RECORD_DIR))