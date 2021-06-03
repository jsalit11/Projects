import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import speech_recognition as sr
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

import constants

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_hands(detection_model, image_np, min_score_treshold, only_box=False):
    hand_class_id = 1
    category_index = {hand_class_id: {'id': hand_class_id, 'name': 'hand'}}     
    input_tensor = np.expand_dims(image_np, 0)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)
    detections = detection_model.signatures['serving_default'](input_tensor)
    if only_box:
        boxes = []
        scores = detections['detection_scores'][0].numpy()
        filtered_box_scores = [idx for idx, score in enumerate(scores) if score > min_score_treshold]
        filtered_boxes = detections['detection_boxes'][0].numpy()[filtered_box_scores]
        return filtered_boxes

    plt.rcParams['figure.figsize'] = [42, 21]
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=min_score_treshold,
        agnostic_mode=False)
    return image_np_with_detections

def get_box_coords(box, width, height):
    box_coords = {}
    box_coords['xmin'] = int(box[1]*width)
    box_coords['ymin'] = int(box[0]*height)
    box_coords['xmax'] = int(box[3]*width)
    box_coords['ymax'] = int(box[2]*height)
    return box_coords

def get_box_center(box_coords):
    box_center = {}
    box_center['x'] = int((box_coords.get('xmax') + box_coords.get('xmin')) / 2)
    box_center['y'] = int((box_coords.get('ymax') + box_coords.get('ymin')) / 2)
    return box_center

def coord_intersect(box_center, shape):
    x_flag = False
    y_flag = False
    center_x = box_center.get('x')
    center_y = box_center.get('y')
    if 'rectangle' in shape.shape_type:
        if (shape.coords.get('xmin') > center_x and shape.coords.get('xmax') < center_x) or \
            (shape.coords.get('xmin') < center_x and shape.coords.get('xmax') > center_x):
            x_flag = True
        if (shape.coords.get('ymin') > center_y and shape.coords.get('ymax') < center_y) or \
            (shape.coords.get('ymin') < center_y and shape.coords.get('ymax') > center_y):
            y_flag = True
    elif 'circle' in shape.shape_type:
        if (shape.coords.get('x') - shape.radius) < center_x < (shape.coords.get('x') + shape.radius):
            x_flag = True
        if (shape.coords.get('y') - shape.radius) < center_y < (shape.coords.get('y') + shape.radius):
            y_flag = True
    if x_flag and y_flag:
        return True

def draw(box_center, frame):
    frame = cv2.circle(frame, (box_center.get('x'), box_center.get('y')), 
            radius=0, color=constants.COLOR, thickness=-1)

def log_results(epoch, num_epochs, idx, num_batches, losses_dict):
    total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
    return ('''
        Epoch {} / {} \
        Batch {} / {} \
        Localization Loss: {} \
        Classification Loss: {} \
        Total Loss: {} \        
    '''.format(epoch, num_epochs, idx, num_batches, losses_dict['Loss/localization_loss'],
        losses_dict['Loss/classification_loss'], total_loss))

def get_drawing_number(draw_objs, frame_number):
    newest_obj_frame = 0
    newest_obj_number = 0
    for draw_obj_number, draw_obj in draw_objs.items():
        if draw_obj.frame_start > newest_obj_frame:
            newest_obj_frame = draw_obj.frame_start
            newest_obj_number = draw_obj.drawing_number
    if newest_obj_frame - frame_number > constants.DRAWING_TIME_LIMIT:
        return newest_obj_number
    else:
        return newest_obj_number + 1
